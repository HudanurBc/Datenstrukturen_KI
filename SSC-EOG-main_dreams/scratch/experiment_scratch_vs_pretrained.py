import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# Add paths for model import
project_root = r"c:\Users\gamah1\Desktop\SSC-EOG-main\SSC-EOG-main_dreams"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))
sys.path.append(os.path.join(project_root, "dreams_pipeline", "2_model_architecture"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from src.train_function import train
from preprocess_dreams import load_active_learning_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1

def train_and_eval(use_pretrained=True, epochs=25, lr=0.0001, rem_boost=1.5):
    print(f"\n==================================================")
    print(f"EXPERIMENT: Pretrained={use_pretrained} | Epochs={epochs} | LR={lr} | Boost={rem_boost}")
    print(f"==================================================")
    
    # 1. Load training data (Patients 1-7)
    train_patients = [1, 2, 3, 4, 5, 6, 7]
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    X_train, y_train, stages_train = load_active_learning_dataset(preprocessed_dir, train_patients)
    
    # Filter out Wake stages (stage == 5) from training
    non_wake_idx = (stages_train != 5)
    X_train = X_train[non_wake_idx]
    y_train = y_train[non_wake_idx]
    
    if len(X_train.shape) != 3:
        X_train = X_train.view(X_train.shape[0], 1, -1)
        
    train_dataset = TensorDataset(X_train.float(), y_train.long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Training set: {len(X_train)} epochs (excluding Wake).")
    
    # 2. Initialize Model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    # 3. Load Pretrained Weights if requested
    if use_pretrained:
        import glob
        search_pattern = os.path.join(project_root, "..", "SSC-EOG-main-sleepdedf", "SSC-EOG-main", "output", "*", "model_pretrained.pth")
        found_files = glob.glob(search_pattern)
        if found_files:
            pretrained_file = max(found_files, key=os.path.getmtime)
            print(f"Loading pretrained weights from: {pretrained_file}")
            pretrained_dict = torch.load(pretrained_file, map_location=DEVICE)
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
        else:
            print("Warning: No pretrained weights found! Training from scratch.")
    else:
        print("Training model from scratch (no transfer learning).")
        
    # 4. Loss & Optimizer
    class_counts = torch.bincount(y_train.long(), minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts.float())
    class_weights[1] = class_weights[1] * rem_boost
    class_weights = class_weights.to(DEVICE)
    print(f"Class weights: Non-REM={class_weights[0]:.4f}, REM={class_weights[1]:.4f}")
    
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.
        for batch_data, batch_tgt in train_loader:
            batch_data, batch_tgt = batch_data.to(DEVICE), batch_tgt.to(DEVICE)
            pred = model(batch_data)
            custom_pred = pred[:, 0, :] # middle epoch/seq_len=1 index
            loss = criterion(custom_pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")
            
    # 6. Evaluate on Patient 9 (unseen)
    print("\n--- Evaluating on unseen Patient 9 ---")
    model.eval()
    X_val, y_val, stages_val = load_active_learning_dataset(preprocessed_dir, [9])
    if len(X_val.shape) != 3:
        X_val = X_val.view(X_val.shape[0], 1, -1)
    
    val_dataset = TensorDataset(X_val.float(), y_val.long())
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(DEVICE)
            output = model(data)
            probs = torch.exp(output[:, 0, :])
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    y_val_np = y_val.numpy()
    stages_val_np = stages_val.numpy()
    
    rem_probs = all_probs[:, 1]
    
    # Test different configurations (thresholds and masks)
    for mask_nrem in [True, False]:
        mask_stages = [0, 1, 2, 5] if mask_nrem else [5]
        for threshold in [0.5, 0.2, 0.1]:
            preds = (rem_probs > threshold).astype(int)
            # Apply stage masking
            for stage in mask_stages:
                preds[stages_val_np == stage] = 0
                
            prec = precision_score(y_val_np, preds, zero_division=0)
            rec = recall_score(y_val_np, preds, zero_division=0)
            f1 = f1_score(y_val_np, preds, zero_division=0)
            
            mask_str = "NREM+Wake Mask" if mask_nrem else "Wake Mask Only"
            print(f"  {mask_str:14s} | Threshold: {threshold:.2f} | Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1: {f1*100:5.2f}%")
            
    # Return trained model state dict for saving if needed
    return model.state_dict()

if __name__ == "__main__":
    # Run Experiment 1: With Pretraining
    state_pretrained = train_and_eval(use_pretrained=True, epochs=15, lr=0.0002, rem_boost=2.0)
    
    # Run Experiment 2: From Scratch
    state_scratch = train_and_eval(use_pretrained=False, epochs=15, lr=0.0002, rem_boost=2.0)
