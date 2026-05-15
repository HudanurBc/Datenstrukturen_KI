import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 10.0) -> np.ndarray:
    """Bandpass filter for EOG signal."""
    b, a = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs)
    return filtfilt(b, a, signal)
