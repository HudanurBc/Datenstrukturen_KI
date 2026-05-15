from pathlib import Path
import pandas as pd
import numpy as np
import torch

from rem_detection.utils.config import load_yaml
from rem_detection.model.architecture import SimpleEOGCNN


def main(config_path: str = "configs/training.yaml"):
    cfg = load_yaml(config_path)
    processed_dir = Path(cfg["processed_dir"])
    model_dir = Path(cfg["model_dir"])
    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(processed_dir / "windows.npy").astype(np.float32)
    metadata = pd.read_csv(processed_dir / "metadata.csv")

    model = SimpleEOGCNN(window_samples=X.shape[1])
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X).unsqueeze(1))).squeeze().numpy()

    out = metadata.copy()
    out["predicted_probability"] = probs
    out["predicted_label"] = (probs >= cfg["threshold"]).astype(int)

    # This file can be reviewed manually / imported into a browser later.
    output_path = reports_dir / "predictions_for_review.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
