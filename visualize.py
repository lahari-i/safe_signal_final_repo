import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

DATASET = "processed_dataset"

FILES = {
    "Fall": os.path.join(
        DATASET,
        "fall",
        os.listdir(os.path.join(DATASET, "fall"))[0]
    ),

    "Jump": os.path.join(
        DATASET,
        "Jump",
        os.listdir(os.path.join(DATASET, "Jump"))[0]
    ),

    "Sit": os.path.join(
        DATASET,
        "sit",
        os.listdir(os.path.join(DATASET, "sit"))[0]
    ),

    "Stand": os.path.join(
        DATASET,
        "stand",
        os.listdir(os.path.join(DATASET, "stand"))[0]
    ),

    "Walk": os.path.join(
        DATASET,
        "walk",
        os.listdir(os.path.join(DATASET, "walk"))[0]
    )
}

# =====================================================
# Motion Signature Visualization
# =====================================================

for activity, path in FILES.items():

    data = np.load(path)

    print(f"{activity}: {data.shape}")

    # -------------------------------------------------
    # Motion Energy
    # -------------------------------------------------

    # Difference between consecutive packets
    diff = np.diff(data, axis=0)

    # Average change across all subcarriers
    motion = np.mean(np.abs(diff), axis=1)

    # Smooth slightly
    window = 5
    kernel = np.ones(window) / window

    motion = np.convolve(
        motion,
        kernel,
        mode="same"
    )

    # Normalize between 0 and 1
    motion = (
        motion - motion.min()
    ) / (
        motion.max() - motion.min() + 1e-8
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(figsize=(12,4))

    plt.plot(
        motion,
        linewidth=3
    )

    plt.title(f"{activity} Motion Signature")

    plt.xlabel("Packet Number")

    plt.ylabel("Normalized Motion")

    plt.grid(True)

    plt.tight_layout()

    plt.show()