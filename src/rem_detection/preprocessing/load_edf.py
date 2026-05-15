from pathlib import Path
import mne
import numpy as np


def load_eog_from_edf(path: str | Path, eog_channels: list[str] | None = None) -> tuple[np.ndarray, float, list[str]]:
    """Load EOG channels from EDF.

    Returns:
        data: shape (n_channels, n_samples)
        fs: sampling frequency
        channel_names: selected channel names
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    fs = float(raw.info["sfreq"])
    all_channels = raw.info["ch_names"]

    if eog_channels:
        selected = eog_channels
    else:
        selected = [ch for ch in all_channels if "EOG" in ch.upper()]

    if not selected:
        raise ValueError(f"No EOG channels found in {path}. Available channels: {all_channels}")

    data = raw.get_data(picks=selected)
    return data, fs, selected
