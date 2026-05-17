import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

# Pfade zum Projekt-Source hinzufügen
sys.path.append(os.getcwd())

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel

# --- KONFIGURATION (Muss zu deinem Modell passen) ---
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
    print(f"\n--- ANALYSE DER PATIENTENDATEN ---")
    
    # 1. Daten laden
    if not os.path.exists(data_path):
        print(f"Fehler: Datei {data_path} nicht gefunden!")
        return

    print(f"Lade Patientendaten von: {data_path}")
    data_dict = torch.load(data_path)
    samples = data_dict["samples"].float()
    labels = data_dict["labels"].long()
    
    total_epochs = len(labels)
    print(f"Gesamtzahl der 30-Sekunden Abschnitte (Epochen): {total_epochs}")

    # Verteilung der Phasen zählen
    print("\nVerteilung der echten Schlafphasen (Ground Truth):")
    label_counts = torch.bincount(labels, minlength=5)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {label_counts[i].item()} Epochen")

    # 2. Modell laden
    if not os.path.exists(model_path):
        print(f"Warnung: Modell {model_path} nicht gefunden. Nur Daten-Analyse möglich.")
        return

    print(f"\nLade Modell: {model_path} für Vorhersage...")
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Vorhersage treffen
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

    # 4. Detail-Report pro Klasse
    print("\nERGEBNIS-REPORT (Was hat das Modell erkannt?):")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        # Indices finden, wo die echte Klasse i ist
        class_indices = np.where(all_targets == i)[0]
        total_class = len(class_indices)
        
        if total_class > 0:
            correct_class = np.sum(all_preds[class_indices] == i)
            accuracy_class = (correct_class / total_class) * 100
            print(f"{name:5}: {total_class:3} Epochen vorhanden | {correct_class:3} richtig erkannt ({accuracy_class:5.1f}%)")
        else:
            print(f"{name:5}: Keine Epochen in den Daten vorhanden.")
    print("-" * 50)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Versuche den Datensatz zu finden (im inneren oder äußeren Ordner)
    MY_DATA = os.path.join(BASE_DIR, "SLEEP_data", "numpy_subjects", "pretext.pt")
    if not os.path.exists(MY_DATA):
        MY_DATA = os.path.join(BASE_DIR, "..", "SLEEP_data", "numpy_subjects", "pretext.pt")
        
    MY_MODEL = os.path.join(BASE_DIR, "model_final.pth")
    
    analyze_data(MY_DATA, MY_MODEL)
