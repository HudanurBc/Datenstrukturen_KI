# preprocess_sleepedf_excerpts.py
# Preprocesses 30-minute excerpts from 2 different Sleep-EDF patients (SC4001 and SC4011)
# Extracts the 'EOG horizontal' channel, resamples to 100 Hz, slices into 2s epochs,
# and saves preprocessed .npz files for model inference & export.

import os
import sys
import numpy as np
import mne
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from settings import SAMPLING_RATE_HZ, SLEEPEDF_EXCERPT_SECONDS, WINDOW_DURATION_SECONDS, WINDOW_STEP_SECONDS

def find_rem_30min_window(annotations, total_duration_sec, window_duration_sec=1800.0):
    """
    Finds a 30-minute (1800s) window centered around a REM sleep period.
    Returns (start_sec, end_sec).
    """
    rem_starts = []
    rem_durations = []
    
    for ann in annotations:
        desc = ann['description'].strip()
        # Sleep stage R / REM
        if 'Sleep stage R' in desc or 'REM' in desc:
            rem_starts.append(ann['onset'])
            rem_durations.append(ann['duration'])
            
    if len(rem_starts) > 0:
        # Find the middle of the first substantial REM period or average REM onset
        first_rem_onset = rem_starts[0]
        # Center the 30-min window around the REM period onset
        start_sec = max(0.0, first_rem_onset - 300.0) # Start 5 minutes before REM onset
        end_sec = start_sec + window_duration_sec
        if end_sec > total_duration_sec:
            end_sec = total_duration_sec
            start_sec = max(0.0, end_sec - window_duration_sec)
        print(f"Found REM sleep! Selected 30-min window: {start_sec:.1f}s to {end_sec:.1f}s (REM starts at {first_rem_onset:.1f}s)")
    else:
        # Fallback to middle 30 minutes of the recording if no REM stage found
        start_sec = max(0.0, (total_duration_sec / 2.0) - (window_duration_sec / 2.0))
        end_sec = start_sec + window_duration_sec
        print(f"No explicit REM stage found in annotations. Fallback to middle window: {start_sec:.1f}s to {end_sec:.1f}s")
        
    return start_sec, end_sec

def preprocess_sleepedf_patient(psg_path, hypno_path, patient_id, output_dir):
    """
    Processes one Sleep-EDF patient:
    - Loads EOG horizontal channel
    - Resamples to 100 Hz
    - Extracts 30-min window containing REM sleep
    - Slices into 2-second epochs (200 sample points)
    - Maps hypnogram sleep stages (REM = 4, Wake = 5, Stage 1-3 = 1-3)
    - Saves patient_sleepedf_{patient_id}.npz
    """
    print(f"\n================ Processing Sleep-EDF Patient {patient_id} ================")
    print(f"PSG File: {psg_path}")
    print(f"Hypnogram: {hypno_path}")
    
    # 1. Read Raw EDF
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    # Check EOG channel name
    available_ch = raw.ch_names
    eog_ch = None
    for ch in ['EOG horizontal', 'EOG', 'EOG1', 'EOG2']:
        if ch in available_ch:
            eog_ch = ch
            break
            
    if eog_ch is None:
        raise ValueError(f"No valid EOG channel found in {available_ch}")
        
    print(f"Selected EOG Channel: {eog_ch}")
    raw.pick_channels([eog_ch])
    
    # Resample to 100 Hz if needed
    if raw.info['sfreq'] != SAMPLING_RATE_HZ:
        print(f"Resampling from {raw.info['sfreq']} Hz to {SAMPLING_RATE_HZ} Hz...")
        raw.resample(SAMPLING_RATE_HZ, verbose=False)
        
    sampling_rate = raw.info['sfreq']
    full_eog_signal = raw.get_data()[0] * 1e6 # Convert Volts to Microvolts (uV) to match DREAMS scaling!
    total_duration_sec = len(full_eog_signal) / sampling_rate
    
    # 2. Read Annotations / Hypnogram
    annotations = mne.read_annotations(hypno_path)
    start_sec, end_sec = find_rem_30min_window(
        annotations, total_duration_sec, window_duration_sec=SLEEPEDF_EXCERPT_SECONDS
    )
    
    # 3. Crop signal to 30-min window
    start_idx = int(start_sec * sampling_rate)
    end_idx = int(end_sec * sampling_rate)
    cropped_signal = full_eog_signal[start_idx:end_idx]
    
    window_duration = WINDOW_DURATION_SECONDS
    window_step = WINDOW_STEP_SECONDS
    window_size = int(window_duration * sampling_rate)
    step_size = int(window_step * sampling_rate)
    num_epochs = max(0, 1 + (len(cropped_signal) - window_size) // step_size)
    window_starts = np.arange(num_epochs) * step_size

    # Slice into overlapping 2-second windows with a 1-second step.
    x_epochs = np.asarray([
        cropped_signal[start:start + window_size]
        for start in window_starts
    ])
    x_epochs = np.asarray(x_epochs).astype(np.float32)
    x_epochs = np.expand_dims(x_epochs, axis=1) # Shape: [num_epochs, 1, 200]
    
    # Dummy binary labels Y (for unlabeled active learning data)
    y_epochs = np.zeros(num_epochs, dtype=np.int32)
    
    # Map sleep stages per epoch for post-processing mask
    # DREAMS stage convention: REM = 4, Wake = 5, S1 = 3, S2 = 2, S3/S4 = 1/0
    stages_epochs = np.ones(num_epochs, dtype=np.int32) * 5 # Default Wake
    
    for ep in range(num_epochs):
        ep_time_global = start_sec + (ep * window_step) + (window_duration / 2.0)
        
        # Check annotations for stage at this timestamp
        for ann in annotations:
            ann_onset = ann['onset']
            ann_duration = ann['duration']
            desc = ann['description'].strip()
            
            if ann_onset <= ep_time_global < (ann_onset + ann_duration):
                if 'Sleep stage R' in desc or 'REM' in desc:
                    stages_epochs[ep] = 4 # REM
                elif 'Sleep stage W' in desc or 'Wake' in desc:
                    stages_epochs[ep] = 5 # Wake
                elif 'Sleep stage 1' in desc:
                    stages_epochs[ep] = 3
                elif 'Sleep stage 2' in desc:
                    stages_epochs[ep] = 2
                elif 'Sleep stage 3' in desc or 'Sleep stage 4' in desc:
                    stages_epochs[ep] = 1
                break
                
    rem_epoch_count = np.sum(stages_epochs == 4)
    print(f"Extracted {num_epochs} overlapping 2-second windows ({num_epochs * window_step / 60:.1f} min span).")
    print(f"Sleep Stage Mapping: REM stage (4) present in {rem_epoch_count} / {num_epochs} windows ({rem_epoch_count * window_step / 60:.1f} min by step).")
    
    # Save preprocessed .npz
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"patient_sleepedf_{patient_id}.npz")
    np.savez(out_path, x=x_epochs, y=y_epochs, stages=stages_epochs, fs=sampling_rate, start_sec=start_sec)
    print(f"Saved preprocessed Sleep-EDF dataset to: {out_path}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Sleep-EDF source directory
    sleepedf_dir = os.path.abspath(os.path.join(project_root, "..", "SSC-EOG-main-sleepdedf", "SLEEP_data", "physionet-sleep-data"))
    output_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    
    # Patient 1: SC4001E0 (Proband 0, Nacht 1)
    p1_psg = os.path.join(sleepedf_dir, "SC4001E0-PSG.edf")
    p1_hyp = os.path.join(sleepedf_dir, "SC4001EC-Hypnogram.edf")
    
    # Patient 2: SC4011E0 (Proband 1, Nacht 1)
    p2_psg = os.path.join(sleepedf_dir, "SC4011E0-PSG.edf")
    p2_hyp = os.path.join(sleepedf_dir, "SC4011EH-Hypnogram.edf")
    
    preprocess_sleepedf_patient(p1_psg, p1_hyp, "1", output_dir)
    preprocess_sleepedf_patient(p2_psg, p2_hyp, "2", output_dir)
    print("\n================ SLEEP-EDF PREPROCESSING COMPLETED ================")

if __name__ == "__main__":
    main()
