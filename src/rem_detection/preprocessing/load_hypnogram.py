from pathlib import Path
import numpy as np


def load_hypnogram(path: str | Path) -> np.ndarray:
    values = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("["):
                continue

            try:
                values.append(float(line.replace(",", ".")))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"No numeric hypnogram values found in {path}")

    return np.asarray(values, dtype=float)


def stage_at_time(hypnogram: np.ndarray, time_sec: float, epoch_sec: float = 5.0):
    idx = int(time_sec // epoch_sec)

    if idx < 0 or idx >= len(hypnogram):
        return None

    return hypnogram[idx]