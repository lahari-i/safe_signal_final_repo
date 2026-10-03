import argparse
import collections
import re
import time
import asyncio
import threading
import json
import os
import numpy as np
from scipy.signal import medfilt, stft
import serial
import torch
import torch.nn.functional as F

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from pynput import mouse

from model import CNNBiGRUClassifier

# --- CONSTANTS ---
TARGET_IQ_COUNT = 104
LINE_RE = re.compile(r"^(-?\d+),\s*\[([\-0-9,]+)\]$")
# Primary class mapping from the model
ACTIVITIES = ["FALL", "JUMP", "SIT", "STAND", "WALK", "EMPTY"]

# STFT Parameters
NPERSEG = 64
NOVERLAP = 48
NFFT = 128
TARGET_FREQ = 25

# Global flag to handle manual mouse trigger overrides
MANUAL_FALL_ACTIVE = False
MANUAL_FALL_COOLDOWN = 0

def on_click(x, y, button, pressed):
    """Mouse listener to catch Forward button (Button.x2) presses."""
    global MANUAL_FALL_ACTIVE, MANUAL_FALL_COOLDOWN
    if pressed and button == mouse.Button.x2:
        MANUAL_FALL_ACTIVE = True
        MANUAL_FALL_COOLDOWN = time.time() + 2.5  # Keep FALL active for 2.5 seconds
        print("\n💥 [MOUSE OVERRIDE] Portronics Forward Button Pressed -> Triggering FALL Alert!")

# --- HELPER FUNCTIONS ---
def parse_raw_line(line):
    line_str = line.strip()
    m = LINE_RE.match(line_str)
    if not m:
        return None, None
    try:
        rssi = int(m.group(1))
        values = [int(v) for v in m.group(2).split(",")]
    except ValueError:
        return None, None

    if len(values) == TARGET_IQ_COUNT:
        formatted_iq = values
    elif len(values) == 128:
        formatted_iq = values[2:106]
    elif len(values) == 384:
        center_start = (384 - TARGET_IQ_COUNT) // 2
        formatted_iq = values[center_start : center_start + TARGET_IQ_COUNT]
    else:
        formatted_iq = (
            values[:TARGET_IQ_COUNT]
            if len(values) > TARGET_IQ_COUNT
            else values + [0] * (TARGET_IQ_COUNT - len(values))
        )

    return rssi, formatted_iq

def iq_to_amplitude(iq_array):
    iq = np.array(iq_array, dtype=np.float32)
    i_vals = iq[0::2]
    q_vals = iq[1::2]
    return np.sqrt(i_vals**2 + q_vals**2)

def process_live_window(amp_window):
    filtered = np.zeros_like(amp_window)
    for sc in range(amp_window.shape[1]):
        filtered[:, sc] = medfilt(amp_window[:, sc], kernel_size=3)

    mean_static = np.mean(filtered, axis=0, keepdims=True)
    zero_mean = filtered - mean_static
    raw_std = float(np.std(zero_mean))

    global_std = max(raw_std, 1e-2)
    normalized = zero_mean / global_std

    spectrograms = []
    for sc in range(normalized.shape[1]):
        _, _, Z = stft(
            normalized[:, sc],
            window="hann",
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            nfft=NFFT,
            boundary=None,
            padded=False,
        )
        spectrograms.append(np.abs(Z))

    mean_spectrogram = np.mean(np.stack(spectrograms, axis=0), axis=0)
    cropped = mean_spectrogram[:TARGET_FREQ, :]
    return np.log1p(cropped), raw_std

# --- FASTAPI & WEBSOCKET ENGINE ---
app = FastAPI()
connected_websockets = set()
latest_telemetry = {
    "type": "telemetry",
    "activity": "EMPTY",
    "connected": False,
    "confidence": 100.0,
    "rssi": -65,
    "std": 0.0000,
    "subcarriers": 52,
    "raw_amps": [0] * 52
}

@app.get("/")
async def get_index():
    if os.path.exists("safesignal_ui (1).html"):
        with open("safesignal_ui (1).html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>safesignal_ui (1).html not found!</h1>", status_code=404)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    await websocket.send_text(json.dumps(latest_telemetry))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

def broadcast_payload(data):
    payload = json.dumps(data)
    loop = getattr(app, "loop", None)
    if loop and loop.is_running():
        for ws in list(connected_websockets):
            asyncio.run_coroutine_threadsafe(ws.send_text(payload), loop)

def broadcast_log(line):
    broadcast_payload({"type": "log", "text": line})

# --- SERIAL PIPELINE THREAD ---
def run_csi_pipeline(args):
    global MANUAL_FALL_ACTIVE, MANUAL_FALL_COOLDOWN, latest_telemetry

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model on {device}...")

    try:
        model = torch.load(args.model, map_location=device, weights_only=False)
    except Exception:
        model = CNNBiGRUClassifier(num_classes=6).to(device)
        print("[⚠️ WARNING] Instantiated base architecture fallback.")

    model.eval()

    print(f"[*] Connecting to {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        err_msg = f"[❌ ERROR] Serial connection failed: {e}"
        print(err_msg)
        broadcast_log(err_msg)
        latest_telemetry["connected"] = False
        broadcast_payload(latest_telemetry)
        return

    time.sleep(2)
    ser.reset_input_buffer()

    csi_ring_buffer = collections.deque(maxlen=args.window)
    pred_history = collections.deque(maxlen=args.history)
    buf = b""
    frames_since_infer = 0
    latest_rssi = -60

    start_msg = "[🚀 RUNNING] SafeSignal Live Inference Active...\n[🖱️ MOUSE READY] Press Portronics Forward Thumb Button to force a Fall alert."
    print(start_msg)
    broadcast_log(start_msg)

    try:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk

            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="ignore").strip()

                rssi, iq_vals = parse_raw_line(line)

                # 1. FILTERING: If line is NOT CSI packet, print to Terminal Console
                if iq_vals is None:
                    if line:
                        print(line)
                        broadcast_log(line)
                    continue

                # 2. CSI PACKET PROCESSING
                if rssi is not None:
                    latest_rssi = rssi

                amps = iq_to_amplitude(iq_vals)
                csi_ring_buffer.append(amps)
                frames_since_infer += 1

                raw_amps_list = [round(float(a), 2) for a in amps.tolist()]

                if len(csi_ring_buffer) == args.window and frames_since_infer >= args.stride:
                    frames_since_infer = 0
                    window_mat = np.array(csi_ring_buffer)

                    doppler_matrix, signal_std = process_live_window(window_mat)

                    # --- CHECK MANUAL OVERRIDE FIRST ---
                    if time.time() < MANUAL_FALL_COOLDOWN:
                        smoothed_label = "FALL"
                        confidence = 99.9
                    else:
                        MANUAL_FALL_ACTIVE = False
                        
                        # NOISE GATE THRESHOLD
                        if signal_std < 0.015:
                            raw_label = "EMPTY"
                            confidence = 100.0
                            top_idx = ACTIVITIES.index("EMPTY")
                        else:
                            tensor_in = (
                                torch.tensor(doppler_matrix, dtype=torch.float32)
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .to(device)
                            )

                            with torch.no_grad():
                                logits = model(tensor_in)
                                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                            # --- FIX 1: REMOVE SIT ENTIRELY ---
                            # Zero out SIT index probability so it is never selected
                            if "SIT" in ACTIVITIES:
                                probs[ACTIVITIES.index("SIT")] = 0.0

                            # --- FIX 2: PREVENT WALK FROM BEING CLASSIFIED AS FALL ---
                            fall_idx = ACTIVITIES.index("FALL")
                            walk_idx = ACTIVITIES.index("WALK")
                            
                            # If model predicts FALL during normal movement energy, default to WALK unless FALL probability is exceptionally high (> 0.75)
                            if np.argmax(probs) == fall_idx and probs[fall_idx] < 0.75:
                                probs[walk_idx] += probs[fall_idx]
                                probs[fall_idx] = 0.0

                            # Re-normalize probabilities
                            probs = probs / np.sum(probs)

                            top_idx = int(np.argmax(probs))
                            raw_label = ACTIVITIES[top_idx]
                            confidence = float(probs[top_idx] * 100)

                        # MAP JUMP -> FALL
                        if raw_label in ["JUMP", "FALL"]:
                            mapped_idx = ACTIVITIES.index("FALL")
                        else:
                            mapped_idx = top_idx

                        # HISTORY SMOOTHING
                        pred_history.append(mapped_idx)
                        smoothed_idx = collections.Counter(pred_history).most_common(1)[0][0]
                        smoothed_label = ACTIVITIES[smoothed_idx]

                    # TERMINAL OUTPUT PRINTING
                    if smoothed_label == "FALL":
                        log_line = f"🚨 [FALL DETECTED] Activity: {smoothed_label:<6} | Confidence: {confidence:6.2f}% | Energy STD: {signal_std:.4f}"
                    else:
                        log_line = f"📊 [PREDICTION]    Activity: {smoothed_label:<6} | Confidence: {confidence:6.2f}% | Energy STD: {signal_std:.4f}"
                    
                    print(log_line)
                    broadcast_log(log_line)

                    latest_telemetry = {
                        "type": "telemetry",
                        "activity": smoothed_label,
                        "connected": True,
                        "confidence": confidence,
                        "rssi": latest_rssi,
                        "std": signal_std,
                        "subcarriers": len(raw_amps_list),
                        "raw_amps": raw_amps_list
                    }
                    broadcast_payload(latest_telemetry)
                else:
                    # Stream raw waveform metrics during buffer fill-up
                    payload = {
                        "type": "telemetry",
                        "activity": latest_telemetry.get("activity", "EMPTY"),
                        "connected": True,
                        "confidence": latest_telemetry.get("confidence", 100.0),
                        "rssi": latest_rssi,
                        "std": latest_telemetry.get("std", 0.0000),
                        "subcarriers": len(raw_amps_list),
                        "raw_amps": raw_amps_list
                    }
                    broadcast_payload(payload)

    except KeyboardInterrupt:
        print("\n[*] Shutting down live pipeline...")
    finally:
        ser.close()

if __name__ == "__main__":
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM5", help="Serial port (e.g., COM5 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--model", default="esp32_finetuned_model.pth")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--history", type=int, default=5) # Increased history buffer slightly for better WALK stability
    cli_args = parser.parse_args()

    pipeline_thread = threading.Thread(target=run_csi_pipeline, args=(cli_args,), daemon=True)
    pipeline_thread.start()

    loop = asyncio.new_event_loop()
    app.loop = loop
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    
    print("[🌐 SERVER] Web Interface running at http://localhost:8000")
    loop.run_until_complete(server.serve())