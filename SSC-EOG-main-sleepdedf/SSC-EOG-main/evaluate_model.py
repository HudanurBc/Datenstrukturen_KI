# evaluate_model.py
# Task: Run evaluation on a trained Sleep-EDF model without training again.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add path to project source
sys.path.append(os.getcwd())

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
CLASS_NAMES = ["Non-REM", "REM"]

def evaluate(model_path, data_path, seed=42):
    print(f"\n================ EVALUATION MANAGER FOR SLEEP-EDF ================")
    print(f"Device: {DEVICE}")
    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_path}")

    # 1. Load dataset
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found!")
        return
        
    data_dict = torch.load(data_path)
    samples = data_dict["samples"].float()
    labels = data_dict["labels"].long()

    if len(samples.shape) != 3:
        samples = samples.view(samples.shape[0], 1, -1)

    full_dataset = TensorDataset(samples, labels)
    
    # 2. Re-create the 80% / 20% split using a fixed seed for reproducibility
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Using torch.Generator with a seed to get a reproducible split
    generator = torch.Generator().manual_seed(seed)
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    pin_memory = True if torch.cuda.is_available() else False
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
    
    print(f"Test split successfully loaded: {len(test_dataset)} epochs (reproduced using seed {seed}).")

    # 3. Load model architecture and weights
    if not os.path.exists(model_path):
        print(f"Error: Model weights file {model_path} not found!")
        return

    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    if "pos_encoder.pe" in state_dict:
        checkpoint_shape = state_dict["pos_encoder.pe"].shape
        model_shape = model.pos_encoder.pe.shape
        if checkpoint_shape != model_shape:
            print(f"Warning: Size mismatch for pos_encoder.pe ({checkpoint_shape} vs {model_shape}). Using model's default buffer.")
            del state_dict["pos_encoder.pe"]
            
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 4. Perform prediction
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

    # 5. Create Confusion Matrix
    cm_percent = confusion_matrix(all_targets, all_preds, labels=[0, 1], normalize='true') * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    # Add a percent sign to the cell annotations
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
 
    plt.xlabel('Vorhergesagt (KI)')
    plt.ylabel('Echt (Arzt)')
    plt.title(f'Konfusionsmatrix (Accuracy: {accuracy*100:.2f}%)')
    
    # Save the output in the same directory as the model
    output_dir = os.path.dirname(model_path)
    if not output_dir:
        output_dir = "."
        
    cm_save_path = os.path.join(output_dir, "final_confusion_matrix.png")
    plt.savefig(cm_save_path)
    print(f"\nConfusion matrix saved as: {cm_save_path}")
    
    # 6. Save Classification Report
    report_save_path = os.path.join(output_dir, "eval_classification_report.txt")
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES)
    
    print("\nDetailed Classification Report:")
    print(report)
    
    with open(report_save_path, "w") as f:
        f.write(report)
    print(f"Classification report saved as: {report_save_path}")
    print("==================================================================")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pth file")
    parser.add_argument("--data_path", type=str, default="SLEEP_data/numpy_subjects/pretext.pt", help="Path to dataset file")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducible test split")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.data_path, args.seed)
