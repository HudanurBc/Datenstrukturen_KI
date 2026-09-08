"""Evaluate the human-feedback model and create reports and plots."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from preprocess_dreams import load_active_learning_dataset
from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from settings import (
    ATTENTION_HEADS, CLASS_COUNT, CNN_LAYERS, DROPOUT, EMBEDDING_SIZE,
    EVALUATION_BATCH_SIZE, FINETUNE_OUTPUT_MODEL, HIDDEN_SIZE,
    MASK_STAGES, REM_THRESHOLD, TRANSFORMER_LAYERS,
)

PATIENT = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Non-REM", "REM"]


def main():
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    X, y, stages = load_active_learning_dataset(preprocessed_dir, [PATIENT])
    model_path = os.path.join(project_root, "output", "feedback", FINETUNE_OUTPUT_MODEL)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Feedback-Modell nicht gefunden: {model_path}")

    model_cnn = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS_COUNT, EMBEDDING_SIZE, ATTENTION_HEADS, HIDDEN_SIZE,
                             TRANSFORMER_LAYERS, model_cnn, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    loader = DataLoader(TensorDataset(X.float(), y.long()),
                        batch_size=EVALUATION_BATCH_SIZE, shuffle=False)
    probabilities = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data.to(DEVICE))
            probabilities.extend(torch.exp(output[:, 0, :]).cpu().numpy())
            targets.extend(target.numpy())

    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    stages = stages.numpy()
    rem_probs = probabilities[:, 1]
    predictions = (rem_probs > REM_THRESHOLD).astype(int)
    for stage in MASK_STAGES:
        predictions[stages == stage] = 0
        rem_probs[stages == stage] = 0.0

    accuracy = float(np.mean(predictions == targets))
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc_score = roc_auc_score(targets, rem_probs)
    output_dir = os.path.join(project_root, "output", "feedback")
    report_path = os.path.join(output_dir, f"classification_report_patient_{PATIENT}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(classification_report(targets, predictions, target_names=CLASS_NAMES, zero_division=0))
        handle.write(f"\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nROC AUC: {auc_score:.4f}\n")

    fpr, tpr, _ = roc_curve(targets, rem_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Feedback-Modell - Patient {PATIENT}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"feedback_roc_curve_patient_{PATIENT}.png"), bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(targets, predictions, labels=[0, 1], normalize="true") * 100
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Feedback-Modell - Patient {PATIENT} | F1: {f1:.4f}")
    plt.savefig(os.path.join(output_dir, f"feedback_confusion_matrix_patient_{PATIENT}.png"), bbox_inches="tight")
    plt.close()
    print(f"Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} AUC={auc_score:.4f}")
    print(f"Ergebnisse gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()
