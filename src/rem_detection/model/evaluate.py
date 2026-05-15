from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from rem_detection.utils.config import load_yaml
from rem_detection.model.architecture import SimpleEOGCNN


def main(config_path: str = "configs/training.yaml"):
    cfg = load_yaml(config_path)
    processed_dir = Path(cfg["processed_dir"])
    model_dir = Path(cfg["model_dir"])

    X = np.load(processed_dir / "windows.npy").astype(np.float32)
    y = np.load(processed_dir / "labels.npy").astype(int)

    model = SimpleEOGCNN(window_samples=X.shape[1])
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X).unsqueeze(1)
        probs = torch.sigmoid(model(X_t)).squeeze().numpy()
        preds = (probs >= cfg["threshold"]).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds, target_names=["Nicht-REM-Event", "REM-Event"], zero_division=0))

    result = {
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "threshold": cfg["threshold"],
    }
    with (model_dir / "evaluation_all_windows.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
