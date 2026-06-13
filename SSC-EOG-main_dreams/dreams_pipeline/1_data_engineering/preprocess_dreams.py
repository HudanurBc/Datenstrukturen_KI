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
        
        # 3. Read labels from hypnogram (primary source according to DREAMS documentation)
        # The hypnogram has 360 entries x 5 seconds = 1800 sec = 30 min (matches EDF!)
        # DREAMS coding (Rechtschaffen & Kales):
        #   4 = REM sleep (our label = 1)
        #   5 = Wake (NOT REM!)
        #   3 = S1, 2 = S2, 1 = S3, 0 = S4
        hypno_path = os.path.join(data_dir, f"Hypnogram_excerpt{patient_num}.txt")
        
        num_signal_epochs = len(eog_data) // int(30 * sampling_rate)
        binary_labels = np.zeros(num_signal_epochs, dtype=np.int32)
        
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
            
            # 6 hypnogram steps (each 5 sec) = 1 epoch (30 sec)
            # Majority vote: if more than 3 of 6 steps are REM (=4) -> epoch is REM
            steps_per_epoch = 6
            num_hypno_epochs = len(hypno_labels) // steps_per_epoch
            
            for ep in range(min(num_signal_epochs, num_hypno_epochs)):
                block = hypno_labels[ep * steps_per_epoch : (ep + 1) * steps_per_epoch]
                binary_labels[ep] = 1 if np.sum(block == 4) > 3 else 0
            
            rem_count = np.sum(binary_labels)
            print(f"Hypnogram: {len(hypno_labels)} entries (5-sec) -> {num_hypno_epochs} epochs (30-sec)")
            print(f"REM epochs (value 4=REM): {rem_count} / {num_signal_epochs}")
        else:
            print(f"Warning: No hypnogram found for Patient {patient_num}! Using Visual_scoring1.")
            visual_path = os.path.join(data_dir, f"Visual_scoring1_excerpt{patient_num}.txt")
            if os.path.exists(visual_path):
                with open(visual_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("["):
                            continue
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                start_sec = float(parts[0])
                                epoch_idx = int(start_sec // 30)
                                if 0 <= epoch_idx < num_signal_epochs:
                                    binary_labels[epoch_idx] = 1
                            except ValueError:
                                continue

        # 4. Split into 30-second epochs (30s * 100Hz = 3000 data points)
        epoch_size = int(30 * sampling_rate) # 3000
        num_signal_epochs = len(eog_data) // epoch_size
        num_epochs = min(num_signal_epochs, len(binary_labels))
        
        print(f"Signal Epochs: {num_signal_epochs}, Label Epochs: {len(binary_labels)}")
        print(f"Using minimum of: {num_epochs} epochs.")
        
        # Cut signal and expand dimensions for the channel axis
        trimmed_data = eog_data[:num_epochs * epoch_size]
        x_epochs = np.split(trimmed_data, num_epochs)
        x_epochs = np.asarray(x_epochs).astype(np.float32)
        x_epochs = np.expand_dims(x_epochs, axis=1) # Shape: [N, 1, 3000]
        
        y_epochs = binary_labels[:num_epochs].astype(np.int32)
        
        print(f"Final Shapes: X={x_epochs.shape}, Y={y_epochs.shape}")
        print(f"REM Epochs: {np.sum(y_epochs)} / {len(y_epochs)} ({np.mean(y_epochs)*100:.2f}%)")
        
        # 5. Save preprocessed data
        out_filename = f"patient_{patient_num}.npz"
        out_path = os.path.join(output_dir, out_filename)
        np.savez(out_path, x=x_epochs, y=y_epochs, fs=sampling_rate)
        print(f"Successfully saved to: {out_path}")

def load_active_learning_dataset(preprocessed_dir, patient_numbers):
    """
    Helper function for Person 2 to load and concatenate a list of patients for training/fine-tuning.
    """
    X_list = []
    y_list = []
    
    for num in patient_numbers:
        path = os.path.join(preprocessed_dir, f"patient_{num}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data for Patient {num} not found! Run preprocessing first.")
            
        data = np.load(path)
        X_list.append(data["x"])
        y_list.append(data["y"])
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return torch.from_numpy(X), torch.from_numpy(y)

if __name__ == "__main__":
    print("================ STARTING DATA ENGINEERING PIPELINE ================")
    # Define local paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    data_dir = os.path.join(project_root, "DatabaseREMs")
    output_dir = os.path.join(data_dir, "preprocessed")
    
    preprocess_all_patients(data_dir, output_dir)
    print("\n================ DATA PREPROCESSING COMPLETED ================")
