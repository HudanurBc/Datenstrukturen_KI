# preprocess_dreams.py
# Person 1: Data Engineer
# Task: Load DREAMS .edf and .txt files, resample to 100Hz, and split into 30s epochs (REM=1 vs Non-REM=0)

import os
import glob
import numpy as np
import mne
import torch

def preprocess_all_patients(data_dir, output_dir):
    """
    Process all 9 patients from the DREAMS REM database:
    - Extracts the EOG1 channel.
    - Resamples the signal from 200 Hz to 100 Hz.
    - Reads the hypnogram labels (txt) and converts them to binary (REM = 4 -> 1, rest -> 0).
    - Splits the EOG signal into 30-second epochs (3000 data points).
    - Saves each patient individually as a .npz file for flexible Active Learning.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all EDF files
    edf_files = sorted(glob.glob(os.path.join(data_dir, "excerpt*.edf")))
    print(f"Found patient files: {len(edf_files)}")
    
    for edf_path in edf_files:
        filename = os.path.basename(edf_path)
        patient_num = filename.replace("excerpt", "").replace(".edf", "")
        
        print(f"\n--- Processing Patient {patient_num} ---")
        
        # 1. Load EDF file (EOG1 channel)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        available_ch = raw.ch_names
        
        # Select EOG channel (usually EOG1)
        eog_ch = 'EOG1' if 'EOG1' in available_ch else ('EOG2' if 'EOG2' in available_ch else None)
        if eog_ch is None:
            print(f"Error: No EOG channel found in {available_ch}")
            continue
            
        print(f"Using EOG channel: {eog_ch}")
        raw.pick_channels([eog_ch])
        
        # 2. Resample to 100 Hz (to match the pretraining model!)
        original_sfreq = raw.info['sfreq']
        if original_sfreq != 100.0:
            print(f"Resampling from {original_sfreq} Hz to 100.0 Hz...")
            raw.resample(100.0, verbose=False)
            
        sampling_rate = raw.info['sfreq']
        eog_data = raw.get_data()[0] # Shape: [time_points]
        
        # 3. Read labels from Visual_scoring1 (REM events) as ground truth
        # DREAMS coding (Rechtschaffen & Kales):
        #   4 = REM sleep, 5 = Wake, 3 = S1, 2 = S2, 1 = S3, 0 = S4
        # We will split the signal into 2-second epochs (200 data points at 100 Hz).
        epoch_duration = 2.0  # seconds
        epoch_size = int(epoch_duration * sampling_rate)  # 200
        num_signal_epochs = len(eog_data) // epoch_size
        
        binary_labels = np.zeros(num_signal_epochs, dtype=np.int32)
        sleep_stages = np.ones(num_signal_epochs, dtype=np.int32) * 5  # Default to Wake (5)
        
        # Load hypnogram if exists (to identify Wake phases)
        hypno_path = os.path.join(data_dir, f"Hypnogram_excerpt{patient_num}.txt")
        if os.path.exists(hypno_path):
            hypno_labels = []
            with open(hypno_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("["):
                        continue
                    try:
                        hypno_labels.append(int(line))
                    except ValueError:
                        continue
            hypno_labels = np.array(hypno_labels)
            
            # Each hypnogram entry corresponds to 5 seconds
            # Map each 2-second epoch to the corresponding hypnogram stage
            for ep in range(num_signal_epochs):
                mid_time = ep * epoch_duration + (epoch_duration / 2.0)
                hypno_idx = int(mid_time // 5.0)
                if 0 <= hypno_idx < len(hypno_labels):
                    sleep_stages[ep] = hypno_labels[hypno_idx]
            print(f"Loaded hypnogram for Patient {patient_num} (mapped to {num_signal_epochs} epochs)")
            
        # Load REM events from Visual_scoring1
        visual_path = os.path.join(data_dir, f"Visual_scoring1_excerpt{patient_num}.txt")
        if os.path.exists(visual_path):
            event_count = 0
            with open(visual_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("["):
                        continue
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            start_sec = float(parts[0])
                            epoch_idx = int(start_sec // epoch_duration)
                            if 0 <= epoch_idx < num_signal_epochs:
                                binary_labels[epoch_idx] = 1
                                event_count += 1
                        except ValueError:
                            continue
            print(f"Loaded Visual_scoring1 REM events: mapped {event_count} events to {np.sum(binary_labels)} epochs")
        else:
            print(f"Warning: No Visual_scoring1 found for Patient {patient_num}!")

        # 4. Split into 2-second epochs (2s * 100Hz = 200 data points)
        print(f"Signal Epochs: {num_signal_epochs}")
        
        # Cut signal and expand dimensions for the channel axis
        trimmed_data = eog_data[:num_signal_epochs * epoch_size]
        x_epochs = np.split(trimmed_data, num_signal_epochs)
        x_epochs = np.asarray(x_epochs).astype(np.float32)
        x_epochs = np.expand_dims(x_epochs, axis=1) # Shape: [N, 1, 200]
        
        y_epochs = binary_labels.astype(np.int32)
        stages_epochs = sleep_stages.astype(np.int32)
        
        print(f"Final Shapes: X={x_epochs.shape}, Y={y_epochs.shape}, Stages={stages_epochs.shape}")
        print(f"REM Event Epochs: {np.sum(y_epochs)} / {len(y_epochs)} ({np.mean(y_epochs)*100:.2f}%)")
        
        # 5. Save preprocessed data
        out_filename = f"patient_{patient_num}.npz"
        out_path = os.path.join(output_dir, out_filename)
        np.savez(out_path, x=x_epochs, y=y_epochs, stages=stages_epochs, fs=sampling_rate)
        print(f"Successfully saved to: {out_path}")

def load_active_learning_dataset(preprocessed_dir, patient_numbers):
    """
    Helper function for Person 2 to load and concatenate a list of patients for training/fine-tuning.
    """
    X_list = []
    y_list = []
    stages_list = []
    
    for num in patient_numbers:
        path = os.path.join(preprocessed_dir, f"patient_{num}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data for Patient {num} not found! Run preprocessing first.")
            
        data = np.load(path)
        X_list.append(data["x"])
        y_list.append(data["y"])
        if "stages" in data:
            stages_list.append(data["stages"])
        else:
            stages_list.append(np.ones(len(data["y"]), dtype=np.int32) * 5)
            
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    stages = np.concatenate(stages_list, axis=0)
    
    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(stages)

if __name__ == "__main__":
    print("================ STARTING DATA ENGINEERING PIPELINE ================")
    # Define local paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    data_dir = os.path.join(project_root, "DatabaseREMs")
    output_dir = os.path.join(data_dir, "preprocessed")
    
    preprocess_all_patients(data_dir, output_dir)
    print("\n================ DATA PREPROCESSING COMPLETED ================")
