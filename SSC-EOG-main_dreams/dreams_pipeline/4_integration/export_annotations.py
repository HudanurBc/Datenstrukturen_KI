# export_annotations.py
# Person 4: Integration Engineer
# Task: Export AI predictions for the EDF browser (Group 3) and import corrected feedback files back.

import os
import sys
import argparse
import numpy as np
import torch

# Import paths for src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from preprocess_dreams import load_active_learning_dataset

# 1. SETTINGS (Adjust Action, Patient, and Paths here!)
ACTION = "export"        # Either "export" (create predictions) or "import" (read feedback)
PATIENT = 6              # Patient number, e.g. 6
MODEL_PATH = "output/dreams_patients_1_2_3_4_5/dreams_model.pth"  # Model path (only for "export")
FEEDBACK_FILE = "DatabaseREMs/Visual_scoring1_excerpt6.txt"        # Feedback file path (only for "import")

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1

def export_predictions_for_group3(patient_num, model_path, output_dir):
    """
    Takes the model predictions for a patient and exports them into the
    EDF Browser format (start time in seconds, duration in seconds).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load patient data
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    X, _ = load_active_learning_dataset(preprocessed_dir, [patient_num])
    
    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)
        
    dataset = torch.utils.data.TensorDataset(X.float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 2. Initialize and load model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    model_file = os.path.join(project_root, model_path)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model not found at {model_file}!")
        
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    
    # 3. Generate predictions
    all_preds = []
    with torch.no_grad():
        for (data,) in loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output[:, 0, :].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            
    # 4. Convert to EDF Browser format (1 line per REM event)
    out_path = os.path.join(output_dir, f"prediction_excerpt{patient_num}.txt")
    
    with open(out_path, 'w') as f:
        f.write("[predicted_REMs/EOGs]\n")
        for i, pred in enumerate(all_preds):
            if pred == 1: # REM predicted
                start_sec = i * 30.0
                duration_sec = 30.0
                f.write(f"   {start_sec:.4f}\t    {duration_sec:.4f}\n")
                
    print(f"AI predictions for Patient {patient_num} successfully exported!")
    print(f"File ready for Group 3 at: {out_path}")
    return out_path

def import_human_feedback(feedback_path, patient_num, output_preprocessed_dir):
    """
    Reads the corrected file from Group 3 and updates the
    labels for the corresponding patient in the preprocessed/.npz file.
    """
    if not os.path.exists(feedback_path):
        raise FileNotFoundError(f"Feedback file not found at {feedback_path}!")
        
    print(f"Reading corrected annotation file: {feedback_path}")
    
    # 1. Load original patient data to get signal shape and epoch count
    patient_path = os.path.join(output_preprocessed_dir, f"patient_{patient_num}.npz")
    if not os.path.exists(patient_path):
        raise FileNotFoundError(f"Original preprocessed Patient {patient_num} not found!")
        
    original_data = np.load(patient_path)
    x = original_data["x"]
    fs = original_data["fs"]
    num_epochs = len(x)
    
    # 2. Initialize new label array (Default: all Non-REM = 0)
    updated_labels = np.zeros(num_epochs, dtype=np.int32)
    
    # 3. Read file and overwrite epochs
    with open(feedback_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            
            # Parse start time and duration
            parts = line.split()
            if len(parts) >= 2:
                try:
                    start_time = float(parts[0])
                    duration = float(parts[1])
                    
                    # Compute which 30-second epochs are touched by this event
                    start_epoch = int(start_time // 30)
                    end_epoch = int((start_time + duration) // 30)
                    
                    # Mark epochs in array (Class 1 = REM)
                    for epoch_idx in range(start_epoch, min(end_epoch + 1, num_epochs)):
                        updated_labels[epoch_idx] = 1
                except ValueError:
                    continue
                    
    # 4. Overwrite the patient's preprocessed NPZ file
    np.savez(patient_path, x=x, y=updated_labels, fs=fs)
    
    print(f"Active Learning: Labels for Patient {patient_num} successfully updated!")
    print(f"New REM distribution: {np.sum(updated_labels)} / {num_epochs} epochs.")

if __name__ == "__main__":
    output_dir = os.path.join(project_root, "DatabaseREMs", "predictions")
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    
    if ACTION == "export":
        export_predictions_for_group3(PATIENT, MODEL_PATH, output_dir)
    elif ACTION == "import":
        if not FEEDBACK_FILE:
            print("Error: FEEDBACK_FILE is required for import!")
        else:
            import_human_feedback(FEEDBACK_FILE, PATIENT, preprocessed_dir)
