# analyze_patient.py
# Task: Analyze Sleep-EDF patient data and print model accuracy details per sleep class.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

# Add path to project source
sys.path.append(os.getcwd())

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel

# --- CONFIGURATION (Must match your model parameters) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
CLASS = 5
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

def analyze_data(data_path, model_path):
    print(f"\n--- PATIENT DATA ANALYSIS ---")
    
    # 1. Load data
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found!")
        return

    print(f"Loading patient data from: {data_path}")
    data_dict = torch.load(data_path)
    samples = data_dict["samples"].float()
    labels = data_dict["labels"].long()
    
    total_epochs = len(labels)
    print(f"Total number of 30-second epochs: {total_epochs}")

    # Count sleep class distribution
    print("\nSleep phase distribution (Ground Truth):")
    label_counts = torch.bincount(labels, minlength=5)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {label_counts[i].item()} epochs")

    # 2. Load model
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_path} not found. Only data analysis is possible.")
        return

    print(f"\nLoading model from: {model_path} for predictions...")
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Generate predictions
    dataloader = DataLoader(TensorDataset(samples, labels), batch_size=1, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output[:, 0, :].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = labels.numpy()

    # 4. Print detailed report per class
    print("\nRESULTS REPORT (Model Predictions):")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        # Find indices where actual class is i
        class_indices = np.where(all_targets == i)[0]
        total_class = len(class_indices)
        
        if total_class > 0:
            correct_class = np.sum(all_preds[class_indices] == i)
            accuracy_class = (correct_class / total_class) * 100
            print(f"{name:5}: {total_class:3} epochs present | {correct_class:3} correctly recognized ({accuracy_class:5.1f}%)")
        else:
            print(f"{name:5}: No epochs present in the data.")
    print("-" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find the dataset (in the inner or outer folder)
    MY_DATA = os.path.join(BASE_DIR, "SLEEP_data", "numpy_subjects", "pretext.pt")
    if not os.path.exists(MY_DATA):
        MY_DATA = os.path.join(BASE_DIR, "..", "SLEEP_data", "numpy_subjects", "pretext.pt")
        
    MY_MODEL = os.path.join(BASE_DIR, "model_final.pth")
    
    analyze_data(MY_DATA, MY_MODEL)
