from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from rem_detection.utils.config import load_yaml

def main(config_path: str = "configs/training.yaml"):
    cfg = load_yaml(config_path)
    reports_dir = Path(cfg["reports_dir"])
    model_dir = Path(cfg["model_dir"])
    
    # 1. Daten laden (Wir nutzen den Output von predict.py!)
    predictions_file = reports_dir / "predictions_for_review.csv"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Konnte {predictions_file} nicht finden. Führe zuerst predict.py aus!")
        
    df = pd.read_csv(predictions_file)
    
    # Aus der metadata.csv wissen wir, dass die echten Labels in der Spalte 'label' stehen
    y_true = df["label"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    # 2. Terminal-Output generieren (landet in der evaluation.txt)
    print("--- EVALUATION REPORT ---")
    print("\nConfusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Nicht-REM-Event", "REM-Event"], zero_division=0))

    # 3. JSON-Export aktualisieren
    result = {
        "confusion_matrix": cm.tolist(),
        "threshold": cfg["threshold"],
    }
    with (model_dir / "evaluation_all_windows.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # 4. VISUALISIERUNG (Dein Beitrag als Evaluator!)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['KI: Nicht-REM', 'KI: REM'], 
                yticklabels=['Echt: Nicht-REM', 'Echt: REM'])
    plt.title(f"Confusion Matrix (Threshold: {cfg['threshold']})")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predictions')
    
    # Bild im reports-Ordner speichern
    plot_path = reports_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nVisualisierung gespeichert unter: {plot_path}")

if __name__ == "__main__":
    main()