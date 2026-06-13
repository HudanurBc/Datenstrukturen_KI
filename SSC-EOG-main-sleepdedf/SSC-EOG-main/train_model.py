# train_model.py
# Task: Pre-train the ResNet-Transformer model on Sleep-EDF data.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add path to project source
sys.path.append(os.getcwd())

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from src.train_function import train

# 1. Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10 # Slightly more epochs as we have less data to train on
LR = 0.001
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
CLASS_NAMES = ["Non-REM", "REM"]

def run_experiment(run_name, epochs, lr, rem_boost):
    # 2. Load data
    data_path = os.path.join("..", "SLEEP_data", "numpy_subjects", "pretext.pt")
    if not os.path.exists(data_path):
        data_path = os.path.join("SLEEP_data", "numpy_subjects", "pretext.pt")

    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found!")
        return

    print(f"Loading data from: {data_path}")
    data_dict = torch.load(data_path)
    samples = data_dict["samples"].float()
    labels = data_dict["labels"].long()

    if len(samples.shape) != 3:
        samples = samples.view(samples.shape[0], 1, -1)

    full_dataset = TensorDataset(samples, labels)
    
    # Split (80% Training, 20% Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

    print(f"Dataset split: {train_size} epochs for training, {test_size} epochs for testing.")

    # 3. Initialize model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)

    # 4. Training setup with Dynamic Class Weighting (against class imbalance)
    train_labels = []
    for _, l in train_dataset:
        train_labels.append(l.item() if hasattr(l, 'item') else l)
    train_labels = torch.tensor(train_labels)
    class_counts = torch.bincount(train_labels, minlength=2)
    print(f"Training class distribution: Non-REM={class_counts[0]}, REM={class_counts[1]}")
    
    # Calculate weighting (inversely proportional to frequency)
    class_weights = len(train_labels) / (2.0 * class_counts.float())
    class_weights[1] = class_weights[1] * rem_boost
    class_weights = class_weights.to(DEVICE)
    print(f"Class weights for loss function (incl. REM boost {rem_boost:.2f}): Non-REM={class_weights[0]:.4f}, REM={class_weights[1]:.4f}")

    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Training
    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, epoch, DEVICE)
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:5.4f}")

    # 6. Evaluation (Blind test)
    print("\n--- STARTING EVALUATION ON UNSEEN TEST DATA ---")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output[:, 0, :].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = (all_preds == all_targets).mean()
    print(f"\nTEST DATA EVALUATION RESULT:")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # 7. Create Confusion Matrix
    # Convert matrix to percentages (each row sums to 100%)
    cm_percent = confusion_matrix(all_targets, all_preds, labels=[0, 1], normalize='true') * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    # Add a percent sign to the cell annotations
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
 
    plt.xlabel('Predicted (AI)')
    plt.ylabel('Actual (Expert)')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.2f}%)')
    
    output_dir = os.path.join("output", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "final_confusion_matrix.png"))
    print(f"\nConfusion matrix saved as '{os.path.join(output_dir, 'final_confusion_matrix.png')}'")
    
    # 8. Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model_pretrained.pth"))
    print(f"Model successfully saved as '{os.path.join(output_dir, 'model_pretrained.pth')}'")

    # 9. Save classification report
    report_path = os.path.join(output_dir, "classification_report.txt")
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report successfully saved to '{report_path}'")

    # 10. Document hyperparameters
    param_path = os.path.join(output_dir, "parameters.txt")
    with open(param_path, "w") as f:
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"REM Boost: {rem_boost}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"CNN Layers: {CNN_LAYERS}\n")
        f.write(f"Transformer Layers: {nLAYERS}\n")
        f.write(f"Embedding Size: {EMB_SIZE}\n")
        f.write(f"Attention Heads: {nHEADS}\n")
        f.write(f"Hidden Size: {D_HID}\n")
    print(f"Parameters successfully saved to '{param_path}'")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Name of the training run")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--rem_boost", type=float, default=1.2, help="Loss weight multiplier for REM class")
    args = parser.parse_args()
    
    if args.run_name is None:
        args.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    run_experiment(args.run_name, args.epochs, args.lr, args.rem_boost)
