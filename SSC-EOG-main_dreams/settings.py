"""Zentrale Konfiguration fuer die DREAMS-Pipeline.

Aendere Werte in dieser Datei. Die Pipeline-Skripte importieren die Einstellungen
von hier und verwenden sie automatisch fuer Preprocessing, Training, Evaluation
und Export.
"""

# Fensterung und Signal
SAMPLING_RATE_HZ = 100.0
WINDOW_DURATION_SECONDS = 2.0
WINDOW_STEP_SECONDS = 1.0

# Datenaufteilung
DREAMS_TRAIN_PATIENTS = [1, 2, 3, 4, 5, 6, 7]
DREAMS_TEST_PATIENT = 9
SLEEPEDF_EXCERPT_SECONDS = 1800.0

# Labels und Evaluation
DREAMS_REM_LABEL = 1
DREAMS_NON_REM_LABEL = 0
SLEEPEDF_REM_STAGE = 4
SLEEPEDF_WAKE_STAGE = 5
REM_THRESHOLD = 0.50
MASK_STAGES = []

# Training
FROM_SCRATCH = True
USE_HYPNOGRAM_FOR_TRAINING = False
TRAINING_EPOCHS = 15
TRAINING_BATCH_SIZE = 32
LEARNING_RATE = 0.0002
REM_BOOST = 2.0

# Evaluation und Export
EVALUATION_BATCH_SIZE = 16
EXPORT_STEP_SECONDS = WINDOW_STEP_SECONDS
EXPORT_ACTION = "export"
EXPORT_PATIENT = 8
FEEDBACK_FILE = "DatabaseREMs/Visual_scoring1_excerpt9.txt"

# Human-in-the-loop Nachtraining
REVIEWED_DATA_FILE = "../reviewed_data/pat8-korriegiert.csv"
REVIEWED_PATIENT = 8
FINETUNE_EPOCHS = 5
FINETUNE_BATCH_SIZE = 32
FINETUNE_LEARNING_RATE = 0.00005
FINETUNE_OUTPUT_MODEL = "dreams_model_feedback_patient8.pth"
REQUIRE_REVIEW_ORIGINAL_MATCH = True

# Modellparameter. Diese muessen bei Transfer Learning zum Checkpoint passen.
CLASS_COUNT = 2
EMBEDDING_SIZE = 512
ATTENTION_HEADS = 8
HIDDEN_SIZE = 1024
TRANSFORMER_LAYERS = 2
CNN_LAYERS = [2, 2, 2, 2]
DROPOUT = 0.1
INPUT_CHANNELS = 1


if WINDOW_DURATION_SECONDS <= 0:
    raise ValueError("WINDOW_DURATION_SECONDS muss groesser als 0 sein.")
if WINDOW_STEP_SECONDS <= 0 or WINDOW_STEP_SECONDS > WINDOW_DURATION_SECONDS:
    raise ValueError(
        "WINDOW_STEP_SECONDS muss groesser als 0 und kleiner/gleich der Fensterlaenge sein."
    )
if SAMPLING_RATE_HZ <= 0:
    raise ValueError("SAMPLING_RATE_HZ muss groesser als 0 sein.")
