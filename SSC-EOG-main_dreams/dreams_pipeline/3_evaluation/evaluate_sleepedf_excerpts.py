# evaluate_sleepedf_excerpts.py
# Evaluates the trained DREAMS model on Sleep-EDF 30-minute excerpts WITHOUT stage masking.
# Since no ground-truth REM-event labels exist, this produces a QUALITATIVE demonstration:
#   - Timeline plot of predicted REM events over the 30-minute window
#   - Prediction probability distribution plot
#   - Summary statistics (how many events predicted, in which sleep stages)
# This is used for the presentation as a "generalization demonstration".

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import paths for src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "dreams_pipeline", "1_data_engineering"))

from src.model.se_resnet_18 import resnet18
from src.model.transformer_model import TransformerModel

# ===== SETTINGS =====
FROM_SCRATCH = True    # True -> use scratch model; False -> use transfer model
THRESHOLD = 0.50       # Probability threshold for predicting REM event
# NO MASK_STAGES: raw EOG classification, no hypnogram needed!

# Model Hyperparameters (must match train_dreams.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
CLASS = 2
EMB_SIZE = 512
nHEADS = 8
D_HID = 1024
nLAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1

# DREAMS stage names for annotation (from hypnogram stored in npz, only used for plot visualization)
STAGE_NAMES = {0: "Deep Sleep (S4)", 1: "Deep Sleep (S3)", 2: "N2 Sleep", 3: "N1 Sleep", 4: "REM Sleep", 5: "Wake"}
STAGE_COLORS = {0: "#1f4e79", 1: "#2e75b6", 2: "#9dc3e6", 3: "#bdd7ee", 4: "#c00000", 5: "#ffc000"}


def load_model(from_scratch):
    """Loads and returns the trained model."""
    modelCNN = resnet18(cnn_layers=CNN_LAYERS, in_lead=1).to(DEVICE)
    model = TransformerModel(CLASS, EMB_SIZE, nHEADS, D_HID, nLAYERS, modelCNN, DROPOUT).to(DEVICE)

    sub_dir = "scratch" if from_scratch else "transfer"
    import glob
    search_pattern = os.path.join(project_root, "output", sub_dir, "*", "dreams_model.pth")
    found_models = glob.glob(search_pattern)
    if found_models:
        model_file = max(found_models, key=os.path.getmtime)
    else:
        model_file = os.path.join(project_root, "output", sub_dir, "dreams_patients_1_2_3_4_5_6_7", "dreams_model.pth")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model not found at: {model_file}\nPlease run train_dreams.py first!")

    print(f"Loading trained model from: {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()
    return model, model_file


def run_inference(model, X_tensor):
    """Runs inference and returns REM probabilities for all epochs."""
    dataset = TensorDataset(X_tensor.float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (data,) in loader:
            data = data.to(DEVICE)
            output = model(data)
            probs = torch.exp(output[:, 0, :])  # shape: [batch, 2]
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    rem_probs = all_probs[:, 1]  # Probability of class 1 (REM event)
    return rem_probs


def evaluate_sleepedf_patient(patient_id, model, output_dir):
    """
    Runs model inference on one Sleep-EDF patient without any stage masking.
    Produces timeline and probability plots and saves summary statistics.
    """
    preprocessed_dir = os.path.join(project_root, "DatabaseREMs", "preprocessed")
    patient_file = os.path.join(preprocessed_dir, f"patient_{patient_id}.npz")

    if not os.path.exists(patient_file):
        raise FileNotFoundError(f"Preprocessed file not found: {patient_file}")

    data = np.load(patient_file)
    X = torch.from_numpy(data["x"])
    stages = data["stages"] if "stages" in data else np.ones(len(X), dtype=np.int32) * 5
    start_sec = float(data["start_sec"]) if "start_sec" in data else 0.0

    if len(X.shape) != 3:
        X = X.view(X.shape[0], 1, -1)

    num_epochs = X.shape[0]
    epoch_duration = 2.0  # seconds
    time_axis_min = np.arange(num_epochs) * epoch_duration / 60.0  # time in minutes

    print(f"\n{'='*55}")
    print(f"  Inference on Sleep-EDF Patient: {patient_id}")
    print(f"  Recording window: {start_sec/3600:.2f}h to {(start_sec + num_epochs*2)/3600:.2f}h (original recording)")
    print(f"  Epochs: {num_epochs} ({num_epochs * 2 / 60:.1f} min) | Threshold: {THRESHOLD}")
    print(f"  No stage masking applied (raw EOG classification only)")
    print(f"{'='*55}")

    # --- Run inference ---
    rem_probs = run_inference(model, X)
    preds = (rem_probs > THRESHOLD).astype(int)

    total_predicted = preds.sum()
    print(f"  Total predicted REM events: {total_predicted} / {num_epochs} epochs ({total_predicted*2:.0f}s = {total_predicted*2/60:.1f} min)")

    # Stage-by-stage breakdown
    print("\n  Predicted REM events per sleep stage:")
    unique_stages = np.unique(stages)
    for s in unique_stages:
        mask = (stages == s)
        in_stage = preds[mask].sum()
        total_stage = mask.sum()
        stage_name = STAGE_NAMES.get(int(s), f"Stage {s}")
        print(f"    {stage_name}: {in_stage} / {total_stage} epochs predicted as REM")

    os.makedirs(output_dir, exist_ok=True)

    # ===== TXT EXPORT for Group 3 (EDF Browser format, same as prediction_excerpt9.txt) =====
    predictions_dir = os.path.join(project_root, "DatabaseREMs", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    txt_path = os.path.join(predictions_dir, f"prediction_{patient_id}.txt")
    with open(txt_path, 'w') as f:
        f.write("[predicted_REMs/EOGs]\n")
        for i, pred in enumerate(preds):
            if pred == 1:
                t_start = i * epoch_duration
                f.write(f"   {t_start:.4f}\t    {epoch_duration:.4f}\n")
    print(f"  TXT predictions for Group 3 saved to: {txt_path}")

    # ===== PLOT 1: Timeline of predicted REM events + sleep stages =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"DREAMS Model Applied to Sleep-EDF Patient {patient_id}\n"
                 f"(30-Min. EOG Window | No Stage Masking | Threshold={THRESHOLD})",
                 fontsize=13, fontweight='bold')

    # Top panel: Probability timeline + threshold line
    ax1.plot(time_axis_min, rem_probs, color='#2e75b6', alpha=0.7, linewidth=0.8, label='REM Probability')
    ax1.axhline(y=THRESHOLD, color='#c00000', linewidth=1.5, linestyle='--', label=f'Threshold = {THRESHOLD}')
    ax1.fill_between(time_axis_min, rem_probs, THRESHOLD,
                     where=(rem_probs > THRESHOLD), color='#c00000', alpha=0.4, label='Predicted REM Events')
    ax1.set_ylabel('Predicted REM Probability', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom panel: Sleep stage hypnogram (from stored hypnogram, for visual context)
    for i in range(num_epochs - 1):
        s = int(stages[i])
        ax2.fill_between([time_axis_min[i], time_axis_min[i+1]], [0, 0], [1, 1],
                         color=STAGE_COLORS.get(s, 'grey'), alpha=0.8)

    # Legend for stages
    legend_patches = [mpatches.Patch(color=STAGE_COLORS[s], label=STAGE_NAMES[s])
                      for s in sorted(STAGE_COLORS.keys()) if s in unique_stages]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)
    ax2.set_ylabel('Sleep Stage\n(Hypnogram)', fontsize=10)
    ax2.set_xlabel('Time in 30-Min. Window (minutes)', fontsize=11)
    ax2.set_yticks([])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    timeline_path = os.path.join(output_dir, f"sleepedf_rem_timeline_patient_{patient_id}.png")
    plt.savefig(timeline_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  Timeline plot saved to: {timeline_path}")

    # ===== PLOT 2: Probability Distribution (Histogram) =====
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rem_probs, bins=50, color='#2e75b6', edgecolor='white', alpha=0.85)
    ax.axvline(x=THRESHOLD, color='#c00000', linewidth=2, linestyle='--', label=f'Threshold = {THRESHOLD}')
    ax.set_xlabel('Predicted REM Probability', fontsize=12)
    ax.set_ylabel('Number of 2s Epochs', fontsize=12)
    ax.set_title(f'Probability Distribution – Sleep-EDF Patient {patient_id}\n'
                 f'({total_predicted} epochs predicted as REM out of {num_epochs})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, f"sleepedf_prob_distribution_patient_{patient_id}.png")
    plt.savefig(dist_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Distribution plot saved to: {dist_path}")

    # ===== Save summary text =====
    summary_path = os.path.join(output_dir, f"sleepedf_summary_patient_{patient_id}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Sleep-EDF Generalization Demonstration – Patient {patient_id}\n")
        f.write(f"{'='*55}\n")
        f.write(f"Window selection: 5 min before first REM onset of the night\n")
        f.write(f"Recording window (original): {start_sec/3600:.2f}h – {(start_sec + num_epochs*2)/3600:.2f}h\n")
        f.write(f"  -> In seconds: {start_sec:.0f}s to {start_sec + num_epochs*2:.0f}s\n")
        f.write(f"Total epochs (2s each): {num_epochs} ({num_epochs * 2 / 60:.1f} min)\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"Stage masking: NONE (raw EOG classification, no hypnogram required)\n\n")
        f.write(f"Total predicted REM events: {total_predicted} / {num_epochs} ({total_predicted*2/60:.1f} min)\n\n")
        f.write("Breakdown by sleep stage (from hypnogram, not used for classification):\n")
        for s in unique_stages:
            mask = (stages == s)
            in_stage = preds[mask].sum()
            total_stage = mask.sum()
            stage_name = STAGE_NAMES.get(int(s), f"Stage {s}")
            f.write(f"  {stage_name}: {in_stage} / {total_stage} epochs predicted as REM\n")
        f.write(f"\nPrediction file for Group 3: DatabaseREMs/predictions/prediction_{patient_id}.txt\n")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*55}")


def main():
    sub_dir = "scratch" if FROM_SCRATCH else "transfer"
    output_dir = os.path.join(project_root, "output", sub_dir, "sleepedf_generalization")

    print("\n===== SLEEP-EDF GENERALIZATION DEMONSTRATION =====")
    print(f"Model type: {'From Scratch' if FROM_SCRATCH else 'Transfer Learning'}")
    print(f"Output directory: {output_dir}")
    print("NOTE: No stage masking applied – raw EOG signal only.\n")

    model, _ = load_model(FROM_SCRATCH)

    for patient_id in ["sleepedf_1", "sleepedf_2"]:
        evaluate_sleepedf_patient(patient_id, model, output_dir)

    print("\n===== SLEEP-EDF EVALUATION COMPLETED =====")


if __name__ == "__main__":
    main()
