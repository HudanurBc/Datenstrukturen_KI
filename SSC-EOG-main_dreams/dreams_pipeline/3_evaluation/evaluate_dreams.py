# evaluate_dreams.py
# Person 3: Data Scientist
# Task: Evaluate predictions on unseen test patient, compute metrics, and visualize confusion matrix.

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import paths for src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from preprocess_dreams import load_active_learning_dataset

# 1. SETTINGS (Choose the test patient and the model path here!)
PATIENT = 6                                             # Patient to evaluate
MODEL_PATH = "output/dreams_patients_1_2_3_4_5/dreams_model.pth"  # Path to the trained model file

# Hyperparameters (must match train_dreams.py!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
CLASS_NAMES = ["Non-REM", "REM"]

def main():
    print(f"Starting DREAMS evaluation for Patient: {PATIENT}")
    
    # 1. Load test patient
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    try:
        X, y = load_active_learning_dataset(preprocessed_dir, [PATIENT])
    except FileNotFoundError as e:
        print(f"Error: Patient {PATIENT} data not found: {e}")
        return
        
    print(f"Test data loaded. Shape: X={X.shape}, Y={y.shape}")
    
    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)
        
    dataset = TensorDataset(X.float(), y.long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    # 3. Load trained weights
    model_file = os.path.join(project_root, MODEL_PATH)
    if not os.path.exists(model_file):
        print(f"Error: Trained DREAMS model {model_file} does not exist!")
        print("Please run train_dreams.py first!")
        return
        
    print(f"Loading fine-tuned model from: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    
    # 4. Generate predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            output = model(data)
            # Handle Transformer output shape (Seq, Batch, Classes -> [0] is the classification token)
            pred = output[:, 0, :].argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 5. Compute metrics
    accuracy = (all_preds == all_targets).mean()
    # Compute metrics robustly (with zero_division=0 to handle cases with no REM epochs predicted/present)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS FOR PATIENT {PATIENT}:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*40)
    
    # Generate and print classification report
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save classification report
    model_dir = os.path.dirname(model_file)
    if model_dir != project_root:
        report_path = os.path.join(model_dir, f"classification_report_patient_{PATIENT}.txt")
    else:
        report_path = os.path.join(project_root, f"classification_report_patient_{PATIENT}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # 6. Visualize and save confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    cm_percent = confusion_matrix(all_targets, all_preds, labels=[0, 1], normalize='true') * 100
    
    plt.figure(figsize=(7, 5))
    # Render confusion matrix heatmap with counts and percentages
    labels = np.array([
        [f"{cm[0,0]}\n({cm_percent[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_percent[0,1]:.1f}%)"],
        [f"{cm[1,0]}\n({cm_percent[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_percent[1,1]:.1f}%)"]
    ])
    
    sns.heatmap(cm_percent, annot=labels, fmt="", cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.xlabel('Predicted (AI)')
    plt.ylabel('Actual (Expert)')
    plt.title(f'DREAMS Test Results (Patient {PATIENT})\nAccuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%')
    
    # If the model_path is inside an output directory, save the plot there as well
    model_dir = os.path.dirname(model_file)
    if model_dir != project_root:
        plot_path = os.path.join(model_dir, f"dreams_confusion_matrix_patient_{PATIENT}.png")
    else:
        plot_path = os.path.join(project_root, f"dreams_confusion_matrix_patient_{PATIENT}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
