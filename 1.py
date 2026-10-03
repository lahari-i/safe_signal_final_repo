"""
SafeSignal - Full CNN-BiGRU Domain Adaptation & Fine-Tuning
-----------------------------------------------------------
Adapts base model weights to ESP32 data for 6 classes:
[fall, jump, sit, stand, walk, empty]
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Correct import name matching model.py
from model import CNNBiGRUClassifier


class ESP32DopplerDataset(Dataset):
    def __init__(self, doppler_root):
        self.samples = []
        self.labels = []
        
        # 6-class mapping matching preprocess_esp32.py
        self.label_map = {
            "fall": 0,
            "jump": 1,
            "sit": 2,
            "stand": 3,
            "walk": 4,
            "empty": 5
        }

        for root, _, files in os.walk(doppler_root):
            for file in files:
                if file.endswith(".npy") and not file.startswith("."):
                    act_name = os.path.basename(root)
                    if act_name in self.label_map:
                        fpath = os.path.join(root, file)
                        arr = np.load(fpath)  # Target matrix shape: (25, 13)
                        self.samples.append(arr)
                        self.labels.append(self.label_map[act_name])

        print(f"[*] Dataset Loaded: {len(self.samples)} doppler slices across {len(set(self.labels))} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = np.expand_dims(x, axis=0)  # Convert (25, 13) -> (1, 25, 13)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def fine_tune(model_path, data_dir, output_path, epochs=30, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # 1. Instantiate the Python Model Class (defaulting to 5 classes initially for weight loading)
    model = CNNBiGRUClassifier(num_classes=5)

    # 2. Safely load checkpoint or state_dict
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and not isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Filter out final classifier layer weights (fc.3) due to shape mismatch (5 -> 6 classes)
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("[*] Successfully loaded state_dict weights into CNNBiGRUClassifier.")
    else:
        model = checkpoint
        print("[*] Successfully loaded full model object.")

    # 3. Update the final linear layer (index 3 of self.fc Sequential) to 6 classes
    # self.fc structure: [0: Linear(256, 64), 1: ReLU, 2: Dropout, 3: Linear(64, num_classes)]
    print(f"[*] Modifying Classifier Head: Changing output layer from 5 to 6 classes.")
    model.fc[3] = nn.Linear(64, 6)
    
    model.to(device)

    # 4. Unfreeze all parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # 5. Set up Optimizer with Differential Learning Rates
    optimizer = torch.optim.Adam([
        {"params": model.cnn.parameters(), "lr": 1e-5},
        {"params": model.bigru.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ])

    criterion = nn.CrossEntropyLoss()
    dataset = ESP32DopplerDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 6. Fine-Tuning Loop
    model.train()
    print("\n" + "=" * 60)
    print("STARTING CNN-BiGRU FINE-TUNING")
    print("=" * 60)
    
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Accuracy: {acc:.2f}%")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save the complete model object
    torch.save(model, output_path)
    print("=" * 60)
    print(f"[✔] Fine-tuned model saved successfully to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="wifall_cnn_bigru.pth", help="Path to base model file")
    parser.add_argument("--data", default="doppler_dataset/ESP32", help="Path to doppler_dataset directory")
    parser.add_argument("--out", default="esp32_finetuned_model.pth", help="Output path for saved model")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    fine_tune(args.model, args.data, args.out, args.epochs)