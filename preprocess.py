import os
import ast
import numpy as np
import pandas as pd

from scipy.signal import medfilt

# =====================================================
# Configuration
# =====================================================

DATASET_PATH = "dataset/WiFall"
OUTPUT_PATH = "processed_dataset"

EXPECTED_LENGTH = 104
RSSI_THRESHOLD = -90

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =====================================================
# Activity Labels
# =====================================================

LABELS = {
    "fall": 0,
    "Jump": 1,
    "sit": 2,
    "stand": 3,
    "walk": 4
}

# =====================================================
# Find every CSV
# =====================================================

def get_all_csv():

    files = []

    for person in sorted(os.listdir(DATASET_PATH)):

        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for activity in sorted(os.listdir(person_path)):

            activity_path = os.path.join(person_path, activity)

            if not os.path.isdir(activity_path):
                continue

            for file in sorted(os.listdir(activity_path)):

                if file.endswith(".csv"):

                    files.append({
                        "person": person,
                        "activity": activity,
                        "path": os.path.join(activity_path, file)
                    })

    return files

# =====================================================
# Clean Packets
# =====================================================

def clean_packets(df):

    df = df[df["first_word_invalid"] == 0]
    df = df[df["rssi"] >= RSSI_THRESHOLD]

    packets = []

    for row in df["data"]:

        try:

            csi = ast.literal_eval(row)

            if len(csi) != EXPECTED_LENGTH:
                continue

            csi = np.array(csi, dtype=np.float32)

            if np.isnan(csi).any():
                continue

            packets.append(csi)

        except:
            continue

    return np.array(packets)

# =====================================================
# Amplitude
# =====================================================

def compute_amplitude(csi):

    I = csi[:,0::2]
    Q = csi[:,1::2]

    return np.sqrt(I**2 + Q**2)

# =====================================================
# Median Filter
# =====================================================

def median_filter(amplitude):

    filtered = np.zeros_like(amplitude)

    for sc in range(amplitude.shape[1]):

        filtered[:,sc] = medfilt(
            amplitude[:,sc],
            kernel_size=3
        )

    return filtered

# =====================================================
# Remove Static Environment
# =====================================================

def remove_static(amplitude):

    return amplitude - np.mean(
        amplitude,
        axis=0,
        keepdims=True
    )

# =====================================================
# Normalize
# =====================================================

def normalize(amplitude):

    mean = np.mean(
        amplitude,
        axis=0,
        keepdims=True
    )

    std = np.std(
        amplitude,
        axis=0,
        keepdims=True
    )

    std[std == 0] = 1e-8

    return (amplitude - mean) / std

# =====================================================
# Process One File
# =====================================================

def process_file(csv_path):

    df = pd.read_csv(csv_path)

    packets = clean_packets(df)

    if len(packets) == 0:
        return None

    amplitude = compute_amplitude(packets)

    amplitude = median_filter(amplitude)

    amplitude = remove_static(amplitude)

    amplitude = normalize(amplitude)

    return amplitude

# =====================================================
# Main
# =====================================================

def main():

    files = get_all_csv()

    print("="*60)
    print("WiFall Preprocessing")
    print("="*60)

    print("Files found:", len(files))
    print()

    metadata = []

    processed = 0
    skipped = 0

    for i, sample in enumerate(files,1):

        activity = sample["activity"]

        save_folder = os.path.join(
            OUTPUT_PATH,
            activity
        )

        os.makedirs(save_folder, exist_ok=True)

        processed_data = process_file(
            sample["path"]
        )

        if processed_data is None:

            skipped += 1
            continue

        filename = os.path.basename(
            sample["path"]
        ).replace(".csv",".npy")

        save_path = os.path.join(
            save_folder,
            filename
        )

        np.save(
            save_path,
            processed_data
        )

        metadata.append({

            "person": sample["person"],
            "activity": activity,
            "label": LABELS[activity],
            "file": save_path,
            "packets": processed_data.shape[0],
            "subcarriers": processed_data.shape[1]

        })

        processed += 1

        print(f"[{i}/{len(files)}] Saved {filename}")

    metadata = pd.DataFrame(metadata)

    metadata.to_csv(

        os.path.join(
            OUTPUT_PATH,
            "metadata.csv"
        ),

        index=False

    )

    print()

    print("="*60)
    print("Finished")
    print("="*60)

    print("Processed :", processed)
    print("Skipped   :", skipped)
    print("Metadata  :", os.path.join(OUTPUT_PATH,"metadata.csv"))

if __name__ == "__main__":

    main()