import numpy as np


def create_windows(signal: np.ndarray, fs: float, window_size_sec: float, step_size_sec: float):
    """Cut 1D signal into overlapping windows.

    Returns windows and start/end times in seconds.
    """
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)

    windows = []
    starts = []
    ends = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal[start:end])
        starts.append(start / fs)
        ends.append(end / fs)

    return np.asarray(windows), np.asarray(starts), np.asarray(ends)
