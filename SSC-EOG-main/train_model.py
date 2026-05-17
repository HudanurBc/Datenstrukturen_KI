import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Pfade zum Projekt-Source hinzufügen
sys.path.append(os.getcwd())

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from src.train_function import train

# 1. Hyperparameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10 
LR = 0.001
CLASS = 5
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

def run_experiment():
    # 2. Daten laden
    data_path = os.path.join("..", "SLEEP_data", "numpy_subjects", "pretext.pt")
    if not os.path.exists(data_path):
        data_path = os.path.join("SLEEP_data", "numpy_subjects", "pretext.pt")

    if not os.path.exists(data_path):
        print(f"Fehler: Datei {data_path} nicht gefunden!")
        return

    print(f"Lade Daten von: {data_path}")
    data_dict = torch.load(data_path)
    samples = data_dict["samples"].float()
    labels = data_dict["labels"].long()

    if len(samples.shape) != 3:
        samples = samples.view(samples.shape[0], 1, -1)

    full_dataset = TensorDataset(samples, labels)
    
    # Split (80% Training, 20% Test) 
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Datensatz aufgeteilt: {train_size} Epochen zum Lernen, {test_size} Epochen zum Testen.")

    # 3. Modell initialisieren
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)

    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. Training
    print(f"Starte Training auf {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, epoch, DEVICE)
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:5.4f}")

    # 6. EVALUIERUNG 
    print("\n--- STARTE EVALUIERUNG AUF UNGESEHENEN DATEN ---")
    model.eval()
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
    print(f"\nERGEBNIS AUF TEST-DATEN:")
    print(f"Genauigkeit (Accuracy): {accuracy*100:.2f}%")

    # 7. Konfusionsmatrix erstellen
    cm_percent = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4], normalize='true') * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    # Füge ein Prozentzeichen zu den Annotationen hinzu
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")

    plt.xlabel('Vorhergesagt (KI)')
    plt.ylabel('Echt (Arzt)')
    plt.title(f'Echter Blind-Test in Prozent (Accuracy: {accuracy*100:.2f}%)')
    
    plt.savefig("final_confusion_matrix.png")
    print("\nKonfusionsmatrix wurde als 'final_confusion_matrix.png' gespeichert.")
    plt.show()

if __name__ == "__main__":
    run_experiment()
