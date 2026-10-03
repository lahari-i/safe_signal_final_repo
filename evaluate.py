import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset import WiFallDopplerDataset
from model import CNNBiGRUClassifier

# =====================================================
# Configuration
# =====================================================
METADATA_PATH = "processed_dataset/metadata.csv"
DOPPLER_DIR = "doppler_dataset"
MODEL_PATH = "models/wifall_cnn_bigru.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVITIES = ["Fall", "Jump", "Sit", "Stand", "Walk"]

def main():
    print("[*] Initializing Evaluation Routine...")
    
    # 1. Recreate the exact same dataset validation split
    full_dataset = WiFallDopplerDataset(metadata_csv=METADATA_PATH, doppler_dir=DOPPLER_DIR)
    dataset_size = len(full_dataset)
    
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load the network layout and inject the saved weights
    model = CNNBiGRUClassifier(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    # 3. Predict metrics without calculating gradients
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    # =====================================================
    # 4. Generate & Display Performance Reports
    # =====================================================
    print("\n" + "="*50)
    print("         CNN-BiGRU CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=ACTIVITIES))
    print("="*50)

    # Calculate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot the matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ACTIVITIES, yticklabels=ACTIVITIES)
    plt.title('WiFall Pre-training Confusion Matrix (CNN-BiGRU)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Activity Class', fontsize=12)
    plt.ylabel('True Activity Class', fontsize=12)
    plt.tight_layout()
    
    # Save chart asset for your project review presentation documentation
    plt.savefig("wifall_evaluation_matrix.png", dpi=300)
    print("[*] Confusion matrix plot exported successfully to 'wifall_evaluation_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()