"""Create a full-size review CSV for the current window configuration.

The generated file is a technical test/template, not human ground truth.
"""

import csv
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from preprocess_dreams import load_active_learning_dataset
from settings import REVIEWED_PATIENT, WINDOW_STEP_SECONDS


def label_name(value):
    return "REM" if int(value) == 1 else "Non-REM"


def main():
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    output_dir = os.path.abspath(os.path.join(project_root, "..", "reviewed_data"))
    output_path = os.path.join(output_dir, f"pat{REVIEWED_PATIENT}-review-template.csv")
    os.makedirs(output_dir, exist_ok=True)

    _, labels, _ = load_active_learning_dataset(preprocessed_dir, [REVIEWED_PATIENT])
    labels = labels.numpy().astype(int)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "start_seconds",
                "original_binary",
                "auto_suggestion",
                "final_binary",
                "corrected",
            ]
        )
        for epoch, label in enumerate(labels):
            name = label_name(label)
            writer.writerow([epoch, epoch * WINDOW_STEP_SECONDS, name, name, name, False])

    print(f"Review-Template erstellt: {output_path}")
    print(f"Fenster: {len(labels)}")
    print("Hinweis: Diese Datei enthaelt keine neuen menschlichen Korrekturen.")


if __name__ == "__main__":
    main()
