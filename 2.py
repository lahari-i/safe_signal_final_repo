"""
SafeSignal - Fine-Tuned Model Evaluation Engine
-----------------------------------------------
Evaluates fine-tuned model performance across all 6 classes.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

# Import model architecture to register custom class globals
from model import CNNBiGRUClassifier


class ESP32DopplerDataset(Dataset):
    def __init__(self, doppler_root):
        self.samples = []
        self.labels = []
        self.label_map = {"fall": 0, "jump": 1, "sit": 2, "stand": 3, "walk": 4, "empty": 5}

        for root, _, files in os.walk(doppler_root):
            for file in files:
                if file.endswith(".npy") and not file.startswith("."):
                    act_name = os.path.basename(root)
                    if act_name in self.label_map:
                        fpath = os.path.join(root, file)
                        self.samples.append(np.load(fpath))
                        self.labels.append(self.label_map[act_name])

        print(f"[*] Evaluation Dataset Loaded: {len(self.samples)} samples across {len(set(self.labels))} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.expand_dims(self.samples[idx], axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    # Load complete PyTorch model object with weights_only=False
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support the weights_only flag
        model = torch.load(model_path, map_location=device)

    model.eval()

    dataset = ESP32DopplerDataset(data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    target_names = ["Fall", "Jump", "Sit", "Stand", "Walk", "Empty"]
    
    print("\n" + "=" * 65)
    print("FINE-TUNED ESP32 EVALUATION RESULTS")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print("\nCONFUSION MATRIX:")
    print(cm_df)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="esp32_finetuned_model.pth", help="Path to fine-tuned model file")
    parser.add_argument("--data", default="doppler_dataset/ESP32", help="Path to doppler_dataset directory")
    args = parser.parse_args()

    evaluate(args.model, args.data)