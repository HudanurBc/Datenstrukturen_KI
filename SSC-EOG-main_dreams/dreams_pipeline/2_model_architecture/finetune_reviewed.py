"""Fine-tune the DREAMS model with human-reviewed window labels."""

import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from preprocess_dreams import load_active_learning_dataset
from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel
from src.train_function import train
from settings import (
    ATTENTION_HEADS,
    CLASS_COUNT,
    CNN_LAYERS,
    DROPOUT,
    EMBEDDING_SIZE,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_OUTPUT_MODEL,
    HIDDEN_SIZE,
    LEARNING_RATE,
    REM_BOOST,
    REQUIRE_REVIEW_ORIGINAL_MATCH,
    REVIEWED_DATA_FILE,
    REVIEWED_PATIENT,
    TRANSFORMER_LAYERS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_label(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "rem"}:
        return 1
    if normalized in {"0", "non-rem", "nonrem"}:
        return 0
    raise ValueError(f"Unbekanntes Label: {value!r}. Erwartet wird REM oder Non-REM.")


def load_reviewed_labels(csv_path, expected_count, original_labels):
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {"epoch", "final_binary"}
    missing = required_columns.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"CSV-Spalten fehlen: {', '.join(sorted(missing))}")
    if len(rows) != expected_count:
        raise ValueError(
            f"Fensteranzahl passt nicht: CSV={len(rows)}, NPZ={expected_count}. "
            "Bitte die Review-Datei mit derselben Fensterung neu erzeugen."
        )

    epochs = np.array([int(row["epoch"]) for row in rows], dtype=np.int64)
    if not np.array_equal(epochs, np.arange(expected_count)):
        raise ValueError("Die CSV-Epochen muessen lueckenlos von 0 bis N-1 nummeriert sein.")

    labels = np.array([parse_label(row["final_binary"]) for row in rows], dtype=np.int64)
    if REQUIRE_REVIEW_ORIGINAL_MATCH and "original_binary" in rows[0]:
        reviewed_original = np.array(
            [parse_label(row["original_binary"]) for row in rows], dtype=np.int64
        )
        mismatch_count = int(np.sum(reviewed_original != original_labels))
        if mismatch_count:
            raise ValueError(
                f"Original-Labels der CSV passen an {mismatch_count} Fenstern nicht zur NPZ-Datei. "
                "Die Review-Datei stammt wahrscheinlich aus einer anderen Fensterung."
            )
    return labels


def load_latest_model():
    model_cnn = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(
        CLASS_COUNT,
        EMBEDDING_SIZE,
        ATTENTION_HEADS,
        HIDDEN_SIZE,
        TRANSFORMER_LAYERS,
        model_cnn,
        DROPOUT,
    ).to(DEVICE)
    pattern = os.path.join(project_root, "output", "scratch", "*", "dreams_model.pth")
    models = glob.glob(pattern)
    if not models:
        raise FileNotFoundError("Kein bestehendes DREAMS-Modell gefunden. Erst train_dreams.py ausfuehren.")
    model_path = max(models, key=os.path.getmtime)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Basis-Modell geladen: {model_path}")
    return model


def main():
    csv_path = os.path.join(project_root, REVIEWED_DATA_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Review-Datei nicht gefunden: {csv_path}")

    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    X, original_y, _ = load_active_learning_dataset(preprocessed_dir, [REVIEWED_PATIENT])
    reviewed_y = load_reviewed_labels(csv_path, len(X), original_y.numpy())

    model = load_latest_model()
    class_counts = torch.bincount(torch.from_numpy(reviewed_y), minlength=CLASS_COUNT).float()
    class_weights = len(reviewed_y) / (CLASS_COUNT * class_counts.clamp_min(1))
    class_weights[1] *= REM_BOOST
    criterion = nn.NLLLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LEARNING_RATE)
    dataset = TensorDataset(X.float(), torch.from_numpy(reviewed_y).long())
    loader = DataLoader(dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)

    print(f"Nachtraining mit Patient {REVIEWED_PATIENT}: {len(dataset)} Review-Fenster")
    model.train()
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        average_loss = train(model, loader, criterion, optimizer, epoch, DEVICE)
        print(f"Fine-tune-Epoch {epoch}/{FINETUNE_EPOCHS} | Loss: {average_loss:.4f}")

    output_dir = os.path.join(project_root, "output", "feedback")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, FINETUNE_OUTPUT_MODEL)
    torch.save(model.state_dict(), output_path)
    print(f"Nachtrainiertes Modell gespeichert: {output_path}")


if __name__ == "__main__":
    main()