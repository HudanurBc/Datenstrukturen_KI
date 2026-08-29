import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Import paths for src modules
project_root = r"c:\Users\gamah1\Desktop\SSC-EOG-main\SSC-EOG-main_dreams"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from preprocess_dreams import load_active_learning_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1

def evaluate_patient(patient_num, model_path, threshold=0.5, mask_stages=[0, 1, 2, 5]):
    # 1. Load patient
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    X, y, stages = load_active_learning_dataset(preprocessed_dir, [patient_num])
    
    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)
        
    dataset = TensorDataset(X.float(), y.long())
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 2. Initialize model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    model.load_state_dict(torch.load(os.path.join(project_root, model_path), map_location=DEVICE))
    model.eval()
    
    # 3. Predict probabilities
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            output = model(data) # output shape: [batch, 1, 2]
            # output is log-softmax, so exp(output) gives probabilities
            probs = torch.exp(output[:, 0, :])
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    stages_np = stages.numpy()
    
    # Get probability for class 1 (REM event)
    rem_probs = all_probs[:, 1]
    
    # Apply threshold
    preds = (rem_probs > threshold).astype(int)
    
    # Apply masking for non-REM stages
    masked_preds = preds.copy()
    for stage in mask_stages:
        masked_preds[stages_np == stage] = 0
        
    prec = precision_score(all_targets, masked_preds, zero_division=0)
    rec = recall_score(all_targets, masked_preds, zero_division=0)
    f1 = f1_score(all_targets, masked_preds, zero_division=0)
    
    # Count of positive predictions before and after masking
    total_pos_before = np.sum(preds == 1)
    total_pos_after = np.sum(masked_preds == 1)
    
    print(f"Patient {patient_num} | Threshold: {threshold:.2f} | Mask Stages: {mask_stages}")
    print(f"  Before Mask: {total_pos_before} predicted REMs")
    print(f"  After Mask:  {total_pos_after} predicted REMs")
    print(f"  Precision:   {prec*100:.2f}%")
    print(f"  Recall:      {rec*100:.2f}% (Found {np.sum((masked_preds == 1) & (all_targets == 1))} / {np.sum(all_targets == 1)})")
    print(f"  F1-Score:    {f1*100:.2f}%")
    print("-" * 50)

model_path = "output/dreams_patients_1_2_3_4_5/dreams_model.pth"

print("=== EVALUATION ON PATIENT 6 (unseen test patient) ===")
evaluate_patient(6, model_path, threshold=0.5, mask_stages=[5]) # Wake only
evaluate_patient(6, model_path, threshold=0.5, mask_stages=[0, 1, 2, 5]) # NREM 0,1,2 + Wake
evaluate_patient(6, model_path, threshold=0.2, mask_stages=[0, 1, 2, 5]) # Lower threshold + NREM/Wake mask
evaluate_patient(6, model_path, threshold=0.1, mask_stages=[0, 1, 2, 5]) # Lower threshold + NREM/Wake mask

print("\n=== EVALUATION ON PATIENT 7 (unseen test patient) ===")
evaluate_patient(7, model_path, threshold=0.5, mask_stages=[5])
evaluate_patient(7, model_path, threshold=0.5, mask_stages=[0, 1, 2, 5])
evaluate_patient(7, model_path, threshold=0.2, mask_stages=[0, 1, 2, 5])
evaluate_patient(7, model_path, threshold=0.1, mask_stages=[0, 1, 2, 5])

print("\n=== EVALUATION ON PATIENT 9 (unseen test patient) ===")
evaluate_patient(9, model_path, threshold=0.5, mask_stages=[5])
evaluate_patient(9, model_path, threshold=0.5, mask_stages=[0, 1, 2, 5])
evaluate_patient(9, model_path, threshold=0.2, mask_stages=[0, 1, 2, 5])
evaluate_patient(9, model_path, threshold=0.1, mask_stages=[0, 1, 2, 5])

print("\n=== EVALUATION ON PATIENT 3 (highly active REM sleep training patient) ===")
evaluate_patient(3, model_path, threshold=0.5, mask_stages=[5])
evaluate_patient(3, model_path, threshold=0.5, mask_stages=[0, 1, 2, 5])
evaluate_patient(3, model_path, threshold=0.2, mask_stages=[0, 1, 2, 5])
evaluate_patient(3, model_path, threshold=0.1, mask_stages=[0, 1, 2, 5])
