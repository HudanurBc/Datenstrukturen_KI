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

# 1. SETTINGS (Choose the test patient and evaluation settings here!)
FROM_SCRATCH = True                                      # True -> evaluate scratch model; False -> transfer model
PATIENT = 8                                              # Patient to evaluate
THRESHOLD = 0.50                                         # Probability threshold for predicting REM event (class 1)
MASK_STAGES = [0, 1, 2, 3]                               # Stages to mask out (Stadium 5 / Wake auskommentiert -> wird NICHT mehr maskiert)

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
    sub_dir = "scratch" if FROM_SCRATCH else "transfer"
    print(f"Starting DREAMS evaluation for Patient: {PATIENT} using {sub_dir} model")
    
    # 1. Load test patient
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    try:
        X, y, stages = load_active_learning_dataset(preprocessed_dir, [PATIENT])
    except FileNotFoundError as e:
        print(f"Error: Patient {PATIENT} data not found: {e}")
        return
        
    print(f"Test data loaded. Shape: X={X.shape}, Y={y.shape}, Stages={stages.shape}")
    
    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)
        
    dataset = TensorDataset(X.float(), y.long())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize model
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    
    # 3. Load trained weights dynamically from the partition folder
    import glob
    search_pattern = os.path.join(project_root, "output", sub_dir, "*", "dreams_model.pth")
    found_models = glob.glob(search_pattern)
    if found_models:
        model_file = max(found_models, key=os.path.getmtime)
    else:
        model_file = os.path.join(project_root, "output", sub_dir, "dreams_patients_1_2_3_4_5_6_7", "dreams_model.pth")
        
    if not os.path.exists(model_file):
        print(f"Error: Trained DREAMS model {model_file} does not exist!")
        print("Please run train_dreams.py first!")
        return
        
    print(f"Loading trained model from: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    
    # 4. Generate predictions and probabilities
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            output = model(data)
            # output is log-softmax, so exp gives probabilities
            probs = torch.exp(output[:, 0, :])
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    stages_np = stages.numpy()
    
    # Extract probability of class 1 (REM event)
    rem_probs = all_probs[:, 1]
    
    # Apply Threshold
    all_preds = (rem_probs > THRESHOLD).astype(int)
    
    # Apply sleep stage masking (force prediction to 0 for masked stages)
    masked_preds = all_preds.copy()
    forced_zero_count = 0
    for stage in MASK_STAGES:
        stage_mask = (stages_np == stage)
        forced_zero_count += np.sum((masked_preds == 1) & stage_mask)
        masked_preds[stage_mask] = 0
    print(f"Stage Masking applied for stages {MASK_STAGES}: forced {forced_zero_count} REM predictions to 0.")
    
    # 5. Compute metrics
    accuracy = (masked_preds == all_targets).mean()
    precision = precision_score(all_targets, masked_preds, zero_division=0)
    recall = recall_score(all_targets, masked_preds, zero_division=0)
    f1 = f1_score(all_targets, masked_preds, zero_division=0)
    
    # Compute ROC Curve and AUC
    # Mask the probability array similarly for ROC calculation
    masked_probs = rem_probs.copy()
    for stage in MASK_STAGES:
        masked_probs[stages_np == stage] = 0.0
        
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(all_targets, masked_probs)
    auc_score = roc_auc_score(all_targets, masked_probs)
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS FOR PATIENT {PATIENT} ({sub_dir.upper()}):")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"ROC AUC:   {auc_score:.4f}")
    print("="*40)
    
    # Generate and print classification report
    report = classification_report(all_targets, masked_preds, target_names=CLASS_NAMES, zero_division=0)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save classification report
    model_dir = os.path.dirname(model_file)
    report_path = os.path.join(model_dir, f"classification_report_patient_{PATIENT}.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nROC AUC: {auc_score:.4f}\n")
    print(f"Classification report saved to: {report_path}")
    
    # 6. Visualize and save ROC curve
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Patient {PATIENT} ({sub_dir.capitalize()} Model)\nAUC: {auc_score:.4f}')
    plt.legend(loc="lower right")
    
    roc_plot_path = os.path.join(model_dir, f"dreams_roc_curve_patient_{PATIENT}.png")
    plt.savefig(roc_plot_path, bbox_inches='tight')
    plt.close()
    print(f"ROC Curve plot saved to: {roc_plot_path}")
    
    # 7. Visualize and save confusion matrix
    cm = confusion_matrix(all_targets, masked_preds, labels=[0, 1])
    cm_percent = confusion_matrix(all_targets, masked_preds, labels=[0, 1], normalize='true') * 100
    
    plt.figure(figsize=(7, 5))
    labels = np.array([
        [f"TN: {cm[0,0]}\n({cm_percent[0,0]:.1f}%)", f"FP: {cm[0,1]}\n({cm_percent[0,1]:.1f}%)"],
        [f"FN: {cm[1,0]}\n({cm_percent[1,0]:.1f}%)", f"TP: {cm[1,1]}\n({cm_percent[1,1]:.1f}%)"]
    ])
    
    sns.heatmap(cm_percent, annot=labels, fmt="", cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.xlabel('Predicted (AI)')
    plt.ylabel('Actual (Expert)')
    plt.title(f'DREAMS Test Results (Patient {PATIENT})\nAccuracy: {accuracy*100:.2f}% | F1: {f1*100:.2f}%')
    
    plot_path = os.path.join(model_dir, f"dreams_confusion_matrix_patient_{PATIENT}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix plot saved to: {plot_path}")
    print("=== DREAMS EVALUATION COMPLETED ===")

if __name__ == "__main__":
    main()
