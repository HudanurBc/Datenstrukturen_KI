# train_dreams.py
# Person 2: AI Architect
# Task: Initialize ResNet-Transformer with pretrained weights and fine-tune on DREAMS patients.

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import paths for src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from src.train_function import train
from preprocess_dreams import load_active_learning_dataset

# 1. SETTINGS (Adjust training parameters here!)
PATIENTS = "1,2,3,4,5"  # Patients for training (e.g. "1,2,3,4,5" or "1,2,3,4,5,6")
EPOCHS = 30            # Number of epochs for fine-tuning
BATCH_SIZE = 32         # Batch size (smaller = more stable learning for small datasets)
LR = 0.0001             # Learning rate (very small = gentler fine-tuning of pretrained model)
REM_BOOST = 0.2         # REM_BOOST < 1.0 -> Model becomes more conservative with REM predictions
                        # (Training has ~38% REM, test has less -> avoids false alarms!)
OUTPUT_MODEL = "dreams_model.pth"  # Output model filename
PRETRAINED_PATH = ""    # Optional: Path to a specific pretrained model (leave empty for auto-search)

# Hyperparameters (must match the pretraining hyperparameters!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1

def main():
    # Parse patient numbers
    patient_list = [int(p) for p in PATIENTS.split(",")]
    print(f"Starting DREAMS model training on patients: {patient_list}")
    
    # 1. Load data
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    try:
        X, y = load_active_learning_dataset(preprocessed_dir, patient_list)
    except FileNotFoundError as e:
        print(f"Error loading preprocessed data: {e}")
        print("Please run the Data Engineering Pipeline (preprocess_dreams.py) first!")
        return
        
    print(f"Data successfully loaded. Shapes: X={X.shape}, Y={y.shape}")
    
    # Check signal shape (expected [N, 1, 3000])
    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)
        
    dataset = TensorDataset(X.float(), y.long())
    pin_memory = True if torch.cuda.is_available() else False
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
    
    # 2. Initialize model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    # 3. Load pretrained weights (Transfer Learning)
    pretrained_file = PRETRAINED_PATH
    if not pretrained_file:
        # Search in the output folders of the sleepedf project directory (where bug-fixed runs are located)
        import glob
        search_pattern = os.path.join(project_root, "..", "SSC-EOG-main -sleepdedf", "SSC-EOG-main", "output", "*", "model_pretrained.pth")
        found_files = glob.glob(search_pattern)
        if found_files:
            # Select the most recently modified file
            pretrained_file = max(found_files, key=os.path.getmtime)
        else:
            # Fallback if no output folders exist
            pretrained_file = os.path.join(project_root, "..", "SSC-EOG-main -sleepdedf", "SSC-EOG-main", "model_pretrained.pth")
            if not os.path.exists(pretrained_file):
                # Local fallback
                pretrained_file = os.path.join(project_root, "model_pretrained.pth")
            
    if os.path.exists(pretrained_file):
        print(f"Loading pretrained weights from: {pretrained_file}")
        try:
            pretrained_dict = torch.load(pretrained_file, map_location=DEVICE)
            model_dict = model.state_dict()
            
            # Filter weights: load only if key exists and shape matches
            # This loads the entire feature extractor (ResNet + Transformer) and skips the final linear layer
            filtered_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            skipped_keys = [k for k in model_dict.keys() if k not in filtered_dict]
            print(f"Transfer Learning: loaded {len(filtered_dict)} layer weights.")
            print(f"The following layers will be re-initialized for 2-class classification: {skipped_keys}")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print("Pretrained weights (ResNet + Transformer) successfully loaded!")
        except Exception as e:
            print(f"Warning loading weights: {e}")
            print("Training will start with random weights.")
    else:
        print(f"No pretrained model file found at: {pretrained_file}")
        print("Training will start with random weights (from scratch).")
        
    # 4. Handle class imbalance (Weighted Cross Entropy)
    class_counts = torch.bincount(y.long(), minlength=2)
    print(f"Class distribution: Non-REM={class_counts[0]}, REM={class_counts[1]}")
    
    # If a class is missing (e.g. training only Patient 1 -> 100% REM),
    # set equal weights to avoid division by zero
    if class_counts[0] == 0 or class_counts[1] == 0:
        class_weights = torch.tensor([1.0, 1.0]).to(DEVICE)
    else:
        class_weights = len(y) / (2.0 * class_counts.float())
        class_weights[1] = class_weights[1] * REM_BOOST
        class_weights = class_weights.to(DEVICE)
    print(f"Class weights for loss function (incl. REM boost {REM_BOOST:.2f}): Non-REM={class_weights[0]:.4f}, REM={class_weights[1]:.4f}")
    
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training
    print("Starting fine-tuning loop...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, loader, criterion, optimizer, epoch, DEVICE)
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:5.4f}")
        
    # 6. Save model
    output_dir = os.path.join(project_root, "output", f"dreams_patients_{PATIENTS.replace(',', '_')}")
    os.makedirs(output_dir, exist_ok=True)
    
    out_model_path = os.path.join(output_dir, OUTPUT_MODEL)
    torch.save(model.state_dict(), out_model_path)
    print(f"\nFine-tuned model successfully saved to: {out_model_path}")
 
    # 7. Document hyperparameters
    param_path = os.path.join(output_dir, "parameters.txt")
    with open(param_path, "w") as f:
        f.write(f"Patients: {PATIENTS}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"REM Boost: {REM_BOOST}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"CNN Layers: {CNN_LAYERS}\n")
        f.write(f"Transformer Layers: {nLAYERS}\n")
        f.write(f"Embedding Size: {EMB_SIZE}\n")
        f.write(f"Attention Heads: {nHEADS}\n")
        f.write(f"Hidden Size: {D_HID}\n")
    print(f"Parameters successfully saved as '{param_path}'.")
    print("=== DREAMS FINE-TUNING COMPLETED ===")

if __name__ == "__main__":
    main()
