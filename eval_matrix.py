import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import CNNBiGRUClassifier


# =====================================================
# Configuration
# =====================================================

DOPPLER_PATH = os.path.join("doppler_dataset", "ESP32")
FINE_TUNED_MODEL_PATH = "esp32_finetuned_model.pth"

LABEL_MAP = {
    "fall": 0,
    "jump": 1,
    "sit": 2,
    "stand": 3,
    "walk": 4,
    "empty": 5
}

CLASS_NAMES = [
    "fall",
    "jump",
    "sit",
    "stand",
    "walk",
    "empty"
]


# =====================================================
# Dataset
# =====================================================

class CSISequenceDataset(Dataset):

    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        file_path = os.path.join(
            DOPPLER_PATH,
            row["subject"],
            row["activity"],
            row["file"]
        )

        # Load spectrogram
        spectrogram = np.load(file_path).astype(np.float32)

        # Add channel dimension
        #
        # Original:
        #     [frequency, time]
        #
        # After:
        #     [1, frequency, time]
        #
        spectrogram = np.expand_dims(
            spectrogram,
            axis=0
        )

        label = LABEL_MAP[row["activity"]]

        return (
            torch.tensor(spectrogram, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# =====================================================
# Confusion Matrix
# =====================================================

def plot_confusion_matrix():

    # -------------------------------------------------
    # Load metadata
    # -------------------------------------------------

    meta_path = os.path.join(
        DOPPLER_PATH,
        "metadata.csv"
    )

    if not os.path.exists(meta_path):
        print(
            f"[ERROR] Metadata file not found: {meta_path}"
        )
        return

    df = pd.read_csv(meta_path)

    print(f"Total samples: {len(df)}")

    # -------------------------------------------------
    # Exact 80/20 stratified validation split
    # -------------------------------------------------

    _, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["activity"]
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    # -------------------------------------------------
    # Dataset and DataLoader
    # -------------------------------------------------

    val_dataset = CSISequenceDataset(val_df)

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load trained model
    # -------------------------------------------------

    if not os.path.exists(FINE_TUNED_MODEL_PATH):

        print(
            f"[ERROR] Model not found: "
            f"{FINE_TUNED_MODEL_PATH}"
        )

        return

    print(
        f"Loading model from "
        f"{FINE_TUNED_MODEL_PATH}..."
    )

    model = torch.load(
        FINE_TUNED_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = model.to(device)

    print("Model loaded successfully.")

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    model.eval()

    all_preds = []
    all_targets = []

    print("\nRunning evaluation...")

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(
            val_loader
        ):

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Print first batch shape for debugging
            if batch_idx == 0:
                print(
                    f"Input shape: {inputs.shape}"
                )

            outputs = model(inputs)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES)))
    )

    print(
        "\nConfusion Matrix (Raw Counts):"
    )

    print(cm)

    # -------------------------------------------------
    # Class-wise accuracy
    # -------------------------------------------------

    print("\nClass-wise Results:")

    for i, class_name in enumerate(CLASS_NAMES):

        total = cm[i].sum()
        correct = cm[i, i]

        if total > 0:
            accuracy = (
                correct / total
            ) * 100
        else:
            accuracy = 0.0

        print(
            f"{class_name:>6}: "
            f"{correct}/{total} "
            f"({accuracy:.2f}%)"
        )

    # -------------------------------------------------
    # Overall accuracy
    # -------------------------------------------------

    total_correct = np.trace(cm)
    total_samples = cm.sum()

    overall_accuracy = (
        total_correct / total_samples
    ) * 100

    print(
        f"\nOverall Accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    disp.plot(
        cmap=plt.cm.Blues,
        ax=ax,
        values_format="d"
    )

    plt.title(
        "ESP32 Fall & Activity Detection - "
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    plot_confusion_matrix()