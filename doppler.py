import os
import numpy as np
from scipy.signal import stft

# =====================================================
# Configuration
# =====================================================

INPUT_PATH = "processed_dataset"
OUTPUT_PATH = "doppler_dataset"

os.makedirs(OUTPUT_PATH, exist_ok=True)

ACTIVITIES = [
    "fall",
    "Jump",
    "sit",
    "stand",
    "walk"
]

# STFT Parameters
NPERSEG = 64
NOVERLAP = 48
NFFT = 128

# =====================================================
# Compute Doppler Spectrogram
# =====================================================

def compute_doppler(amplitude):

    spectrograms = []

    for sc in range(amplitude.shape[1]):

        signal = amplitude[:, sc]

        _, _, Z = stft(
            signal,
            window="hann",
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            nfft=NFFT,
            boundary=None,
            padded=False
        )

        spectrograms.append(np.abs(Z))

    spectrograms = np.stack(spectrograms)

    # Average over all 52 subcarriers
    doppler = np.mean(spectrograms, axis=0)

    # Log compression
    doppler = np.log1p(doppler)

    return doppler.astype(np.float32)

# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("Generating Doppler Spectrograms")
    print("=" * 60)

    total = 0

    for activity in ACTIVITIES:

        input_folder = os.path.join(
            INPUT_PATH,
            activity
        )

        output_folder = os.path.join(
            OUTPUT_PATH,
            activity
        )

        os.makedirs(output_folder, exist_ok=True)

        for file in sorted(os.listdir(input_folder)):

            if not file.endswith(".npy"):
                continue

            amplitude = np.load(
                os.path.join(input_folder, file)
            )

            doppler = compute_doppler(amplitude)

            np.save(
                os.path.join(output_folder, file),
                doppler
            )

            total += 1

            print(f"[{total}] {activity}/{file}")

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)
    print("Total Doppler files:", total)

if __name__ == "__main__":

    main()