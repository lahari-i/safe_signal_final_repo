"""
SafeSignal - Raw CSI Data Collector (WiFall-Compatible Schema)
--------------------------------------------------------------
Connects to the Rx ESP32 over serial, parses incoming CSI frames, cleans and
formats the payload to match the exact WiFall dataset structure (104 values: 52 I/Q pairs),
and writes to a labeled CSV file.

Usage:
    python collect_data.py --port COM7 --label fall --duration 30 --out dataset/ESP32

Classes to collect:
    empty, sit, stand, walk, fall, jump
"""

import argparse
import csv
import os
import re
import time
import serial

# Regular expression to match PlatformIO serial format: -62,[2,24,2,23,...]
LINE_RE = re.compile(r"^(-?\d+),\s*\[([\-0-9,]+)\]$")
TARGET_IQ_COUNT = 104  # 52 subcarriers * 2 (I/Q)


def parse_and_format_wifall(line):
    """
    Parses a raw serial line, extracts RSSI and I/Q values, trims/pads array 
    to exactly 104 values, and converts to a WiFall-compatible data string.
    """
    m = LINE_RE.match(line.strip())
    if not m:
        return None
        
    rssi = int(m.group(1))
    try:
        values = [int(v) for v in m.group(2).split(",")]
    except ValueError:
        return None

    # Handle different incoming payload sizes from ESP32
    if len(values) == TARGET_IQ_COUNT:
        formatted_iq = values
    elif len(values) == 128:
        # Trims DC offset and guard null subcarriers to grab center 52 active pairs
        formatted_iq = values[2:106]
    elif len(values) == 384:
        # Center-crop 104 values from 384-length array
        center_start = (384 - TARGET_IQ_COUNT) // 2
        formatted_iq = values[center_start : center_start + TARGET_IQ_COUNT]
    else:
        # Fallback padding/trimming for non-standard lengths
        if len(values) < TARGET_IQ_COUNT:
            formatted_iq = values + [0] * (TARGET_IQ_COUNT - len(values))
        else:
            formatted_iq = values[:TARGET_IQ_COUNT]

    # Convert to WiFall array string representation
    data_str = f"[{', '.join(map(str, formatted_iq))}]"
    return rssi, data_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM7 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--label", required=True, choices=["empty", "sit", "stand", "walk", "fall","jump"])
    ap.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds")
    ap.add_argument("--out", default="dataset/ESP32", help="Output directory")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.label)
    os.makedirs(out_dir, exist_ok=True)
    session_name = f"{args.label}_{int(time.time())}.csv"
    out_path = os.path.join(out_dir, session_name)

    print(f"[*] Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"[❌ ERROR] Could not open serial port {args.port}: {e}")
        return

    time.sleep(2)  # Allow ESP32 reset to settle
    ser.reset_input_buffer()

    # Exact 28-column WiFall schema
    header = [
        "type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate",
        "sig_mode", "mcs", "cwb", "smoothing", "not_sounding", "aggregation",
        "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel_primary",
        "channel_secondary", "local_timestamp", "ant", "sig_len", "rx_state",
        "len", "first_word_invalid", "data"
    ]

    print(f"[*] Recording '{args.label}' for {args.duration:.1f}s -> {out_path}")
    print("[*] Perform the activity now (repeat continuously)...")

    rows_written = 0
    bad_lines = 0
    t_start = time.time()
    buf = b""

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while (time.time() - t_start) < args.duration:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk

            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="ignore")
                parsed = parse_and_format_wifall(line)
                
                if parsed is None:
                    bad_lines += 1
                    continue
                    
                rssi, data_str = parsed
                seq_num = rows_written + 1
                curr_time = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"

                # Construct standard WiFall data row
                row = [
                    "CSI_DATA", seq_num, curr_time, 1, args.label, "d4:da:21:98:07:d9",
                    rssi, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, -99, 0, 12, 2,
                    int(time.time() * 1000), 0, 37, 0, TARGET_IQ_COUNT, 0, data_str
                ]
                
                writer.writerow(row)
                rows_written += 1

                if rows_written % 50 == 0:
                    remaining = args.duration - (time.time() - t_start)
                    print(f"   Captured {rows_written} packets | {remaining:.1f}s remaining", end="\r")

    ser.close()
    print()
    print(f"[✔] Done. Saved {rows_written} packets to {out_path} ({bad_lines} malformed lines discarded)")


if __name__ == "__main__":
    main()