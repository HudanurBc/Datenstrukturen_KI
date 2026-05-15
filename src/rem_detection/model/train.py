from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from rem_detection.utils.config import load_yaml
from rem_detection.model.architecture import SimpleEOGCNN


def main(config_path: str = "configs/training.yaml"):
    cfg = load_yaml(config_path)
    processed_dir = Path(cfg["processed_dir"])
    model_dir = Path(cfg["model_dir"])
    reports_dir = Path(cfg["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(processed_dir / "windows.npy").astype(np.float32)
    y = np.load(processed_dir / "labels.npy").astype(np.float32)

    if X.ndim != 2:
        raise ValueError(f"Expected windows shape (n_windows, n_samples), got {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    X_train_t = torch.tensor(X_train).unsqueeze(1)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test).unsqueeze(1)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model = SimpleEOGCNN(window_samples=X.shape[1])

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    if positives > 0:
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs >= cfg["threshold"]).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "threshold": cfg["threshold"],
        "train_positive": int(y_train.sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
    }

    torch.save(model.state_dict(), model_dir / "model.pth")
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", model_dir / "model.pth")
    print(metrics)


if __name__ == "__main__":
    main()
