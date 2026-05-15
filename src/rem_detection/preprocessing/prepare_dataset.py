from pathlib import Path
import re
import numpy as np
import pandas as pd

from rem_detection.utils.config import load_yaml
from rem_detection.preprocessing.load_edf import load_eog_from_edf
from rem_detection.preprocessing.load_hypnogram import load_hypnogram, stage_at_time
from rem_detection.preprocessing.load_events import load_visual_events, overlaps_event
from rem_detection.preprocessing.filter_signal import bandpass_filter
from rem_detection.preprocessing.create_windows import create_windows


def excerpt_id_from_name(name: str) -> str:
    match = re.search(r"excerpt(\d+)", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract excerpt id from filename: {name}")
    return match.group(1)


def main(config_path: str = "configs/preprocessing.yaml"):
    cfg = load_yaml(config_path)
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []
    all_meta = []

    edf_files = sorted(raw_dir.glob("excerpt*.edf"))
    if not edf_files:
        raise FileNotFoundError(
            f"No excerpt*.edf files found in {raw_dir}. Put DREAMS REMs files there first."
        )

    for edf_path in edf_files:
        excerpt_id = excerpt_id_from_name(edf_path.name)
        hyp_path = raw_dir / f"Hypnogram_excerpt{excerpt_id}.txt"
        visual_path = raw_dir / f"Visual_scoring1_excerpt{excerpt_id}.txt"

        if not hyp_path.exists():
            raise FileNotFoundError(f"Missing hypnogram: {hyp_path}")
        if not visual_path.exists():
            raise FileNotFoundError(f"Missing visual scoring file: {visual_path}")

        print(f"Processing excerpt {excerpt_id}...")
        eog_data, fs, selected_channels = load_eog_from_edf(edf_path, cfg.get("eog_channels") or None)
        hypnogram = load_hypnogram(hyp_path)
        events = load_visual_events(visual_path)

        # Simple first version: average all EOG channels into one signal.
        filtered_channels = []
        for ch_signal in eog_data:
            filtered_channels.append(
                bandpass_filter(
                    ch_signal,
                    fs=fs,
                    low_hz=cfg["filter_low_hz"],
                    high_hz=cfg["filter_high_hz"],
                )
            )
        signal = np.mean(np.vstack(filtered_channels), axis=0)

        windows, starts, ends = create_windows(
            signal,
            fs=fs,
            window_size_sec=cfg["window_size_sec"],
            step_size_sec=cfg["step_size_sec"],
        )

        for window, start_sec, end_sec in zip(windows, starts, ends):
            mid_sec = (start_sec + end_sec) / 2
            sleep_stage = stage_at_time(hypnogram, mid_sec, cfg["hypnogram_epoch_sec"])

            label = overlaps_event(start_sec, end_sec, events)

            # Optional later: keep only REM sleep-stage windows.
            # This is disabled by default because stage coding can differ.
            if cfg.get("only_rem_sleep_stage", False):
                # Adjust this condition after confirming DREAMS stage codes.
                # For now, it keeps all windows and warns in README.
                pass

            all_windows.append(window.astype(np.float32))
            all_labels.append(label)
            all_meta.append({
                "excerpt_id": excerpt_id,
                "edf_file": edf_path.name,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "label": int(label),
                "sleep_stage": sleep_stage,
                "channels": ";".join(selected_channels),
                "fs": fs,
            })

    X = np.asarray(all_windows, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.float32)
    meta = pd.DataFrame(all_meta)

    np.save(processed_dir / "windows.npy", X)
    np.save(processed_dir / "labels.npy", y)
    meta.to_csv(processed_dir / "metadata.csv", index=False)

    print("Done.")
    print(f"windows: {X.shape}")
    print(f"labels:  {y.shape}")
    print(f"positive REM-event windows: {int(y.sum())}")
    print(f"negative windows: {int((y == 0).sum())}")
    print(f"saved to: {processed_dir}")


if __name__ == "__main__":
    main()
