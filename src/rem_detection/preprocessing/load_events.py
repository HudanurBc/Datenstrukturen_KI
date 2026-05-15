from pathlib import Path
import pandas as pd


def load_visual_events(path: str | Path) -> pd.DataFrame:
    events = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("["):
                continue

            parts = line.replace(",", ".").split()

            if len(parts) < 2:
                continue

            try:
                start_sec = float(parts[0])
                duration_sec = float(parts[1])
            except ValueError:
                continue

            if duration_sec <= 0:
                continue

            end_sec = start_sec + duration_sec

            events.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
            })

    if not events:
        raise ValueError(f"No visual events found in {path}")

    return pd.DataFrame(events)


def overlaps_event(start_sec: float, end_sec: float, events: pd.DataFrame) -> int:
    overlap = (events["start_sec"] < end_sec) & (events["end_sec"] > start_sec)
    return int(overlap.any())