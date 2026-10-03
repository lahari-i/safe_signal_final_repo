import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import WiFallDopplerDataset
from model import CNNBiGRUClassifier

METADATA_PATH = "processed_dataset/metadata.csv"
DOPPLER_DIR = "doppler_dataset"
MODEL_SAVE_PATH = "wifall_cnn_bigru.pth"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 60)
    print(f"CNN-BiGRU Pre-training Loop Initialized on Device: {DEVICE}")
    print("=" * 60)

    full_dataset = WiFallDopplerDataset(metadata_csv=METADATA_PATH, doppler_dir=DOPPLER_DIR)
    dataset_size = len(full_dataset)
    
    if dataset_size == 0:
        print("[!] Error: No targets parsed in metadata. Build dataset matrices first.")
        return

    # Random split partition settings (80/20 train/validation split)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNBiGRUClassifier(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (correct_val / total_val) * 100

        print(f" >> [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% "
              f"|| [Val] Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # Save weights only if validation metrics hit an absolute performance spike
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" ==> Weight Saved! Updated best matrix parameters: {best_val_acc:.2f}%\n")
        else:
            print()

    print("=" * 60)
    print(f"Pre-training terminated. Core weights exported to: {MODEL_SAVE_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()