# SSC-EOG DREAMS Pipeline

Automatische Erkennung von schnellen Augenbewegungen (REM-Ereignissen) in EOG-Signalen auf Basis von 2-Sekunden-Epochen. Das Modell kombiniert ein SE-ResNet-18 mit einem Transformer-Encoder und wird auf der DREAMS REM-Datenbank trainiert.


## Ziel

Das Modell bewertet alle 2 Sekunden, ob ein REM-Ereignis im EOG-Signal vorliegt. Die klinische Anforderung ist ein Recall groesser als 85 Prozent bei moeglichst hoher Praezision.


## Datenbasis

- DREAMS REM-Datenbank mit 9 Patienten (excerpt1.edf bis excerpt9.edf)
- Jeder Patient hat ein Hypnogramm (Schlafstadien) und manuell annotierte REM-Ereignisse (Visual_scoring1)
- Zusaetzlich wurden 2 Sleep-EDF Patienten fuer den Generalisierungstest vorbereitet


## Ordnerstruktur

```
SSC-EOG-main_dreams/
|
|-- DatabaseREMs/                        Rohdaten und vorverarbeitete Daten
|   |-- excerpt1.edf bis excerpt9.edf    EDF-Dateien der 9 Patienten
|   |-- Hypnogram_excerpt1.txt ...       Schlafstadien-Labels (5-Sekunden-Aufloesung)
|   |-- Visual_scoring1_excerpt1.txt ... Manuell annotierte REM-Ereignisse (Ground Truth)
|   |-- Automatic_detection_excerpt1.txt Automatische Detektion (Referenz)
|   |-- preprocessed/                    Vorverarbeitete .npz Dateien pro Patient
|   |   |-- patient_1.npz bis patient_9.npz
|   |   |-- patient_sleepedf_1.npz      Sleep-EDF Patient 1
|   |   |-- patient_sleepedf_2.npz      Sleep-EDF Patient 2
|   |-- predictions/                     Exportierte Vorhersagen im EDF-Browser-Format
|       |-- prediction_excerpt6.txt
|       |-- prediction_excerpt9.txt
|       |-- prediction_sleepedf_1.txt
|       |-- prediction_sleepedf_2.txt
|       |-- prediction_sleepedf_excerpt1.txt
|       |-- prediction_sleepedf_excerpt2.txt
|
|-- dreams_pipeline/                     Hauptpipeline in 4 Schritten
|   |-- 1_data_engineering/
|   |   |-- preprocess_dreams.py         Vorverarbeitung der DREAMS EDF-Dateien
|   |   |-- preprocess_sleepedf_excerpts.py  Vorverarbeitung der Sleep-EDF Daten
|   |-- 2_model_architecture/
|   |   |-- train_dreams.py              Training des Modells (Scratch oder Transfer)
|   |-- 3_evaluation/
|   |   |-- evaluate_dreams.py           Evaluation auf ungesehenen Testpatienten
|   |   |-- evaluate_sleepedf_excerpts.py  Generalisierungstest auf Sleep-EDF Daten
|   |-- 4_integration/
|       |-- export_annotations.py        Export und Import von Vorhersagen fuer Gruppe 3
|
|-- src/                                 Modell-Quellcode
|   |-- model/
|   |   |-- se_resnet_18.py              SE-ResNet-18 (1D CNN Backbone)
|   |   |-- transformer_model.py         Transformer-Encoder mit Positional Encoding
|   |   |-- se_layer.py                  Squeeze-and-Excitation Layer
|   |   |-- model_initialization.py      Referenz fuer Modell-Initialisierung
|   |-- train_function.py               Trainingsschleife (eine Epoche)
|
|-- output/                              Trainierte Modelle und Evaluationsergebnisse
|   |-- scratch/                         Ergebnisse des From-Scratch-Modells
|   |   |-- dreams_patients_1_2_3_4_5_6_7/
|   |   |   |-- dreams_model.pth         Trainiertes Modell (gitignored)
|   |   |   |-- parameters.txt           Trainings-Hyperparameter
|   |   |   |-- classification_report_patient_8.txt
|   |   |   |-- classification_report_patient_9.txt
|   |   |   |-- dreams_confusion_matrix_patient_8.png
|   |   |   |-- dreams_confusion_matrix_patient_9.png
|   |   |   |-- dreams_roc_curve_patient_8.png
|   |   |   |-- dreams_roc_curve_patient_9.png
|   |   |-- transfer/                     Ergebnisse des Transfer-Learning-Modells
|   |   |-- dreams_patients_1_2_3_4_5_6_7/
|   |       |-- (gleiche Struktur wie oben)
|
|-- scratch/                             Experimentelle Skripte
|   |-- experiment_scratch_vs_pretrained.py  Vergleich Scratch vs. Transfer Learning
|   |-- tune_evaluation.py               Threshold- und Masken-Tuning
|
|-- .gitignore                           Git-Konfiguration
```


## Pipeline-Ablauf

Die Pipeline besteht aus 4 Schritten. Jeder Schritt ist ein eigenes Skript im Ordner dreams_pipeline.

### Schritt 1 - Data Engineering

Skript: `dreams_pipeline/1_data_engineering/preprocess_dreams.py`

- Liest die EDF-Dateien und extrahiert den EOG1-Kanal
- Resampled das Signal von 200 Hz auf 100 Hz
- Liest das Hypnogramm und die manuellen REM-Annotationen
- Teilt das Signal in 2-Sekunden-Epochen (200 Datenpunkte)
- Speichert pro Patient eine .npz Datei mit Signal, Labels und Schlafstadien

Ausgabe: `DatabaseREMs/preprocessed/patient_1.npz` bis `patient_9.npz`

### Schritt 2 - Modell Training

Skript: `dreams_pipeline/2_model_architecture/train_dreams.py`

- Laedt die vorverarbeiteten Daten der gewaehlten Trainingspatienten
- Filtert Wachphasen (Stadium 5) aus den Trainingsdaten
- Initialisiert das SE-ResNet-18 + Transformer Modell
- Trainiert mit gewichteter NLLLoss (REM-Boost fuer hoeheren Recall)
- Speichert das Modell und die Hyperparameter

Einstellungen oben im Skript anpassen:
- FROM_SCRATCH: True fuer Training von Grund auf, False fuer Transfer Learning
- PATIENTS: Kommagetrennte Liste der Trainingspatienten
- EPOCHS, LR, REM_BOOST: Trainingsparameter

Ausgabe: `output/scratch/dreams_patients_X/dreams_model.pth` und `parameters.txt`

### Schritt 3 - Evaluation

Skript: `dreams_pipeline/3_evaluation/evaluate_dreams.py`

- Laedt das trainierte Modell und die Daten eines ungesehenen Testpatienten
- Generiert Vorhersagen mit konfigurierbarem Threshold
- Wendet Post-Processing an: Schlafphasen-Maskierung (alle Stadien ausser REM werden blockiert)
- Berechnet Accuracy, Precision, Recall, F1-Score und ROC AUC
- Erstellt Confusion Matrix und ROC-Kurve als PNG

Einstellungen oben im Skript anpassen:
- PATIENT: Nummer des Testpatienten
- THRESHOLD: Entscheidungsschwelle (Standard 0.50)
- MASK_STAGES: Welche Stadien maskiert werden

Ausgabe: Classification Reports, Confusion Matrices und ROC-Kurven im jeweiligen output-Ordner

### Schritt 4 - Integration

Skript: `dreams_pipeline/4_integration/export_annotations.py`

- Exportiert die Modell-Vorhersagen im EDF-Browser-Format fuer Gruppe 3
- Kann korrigierte Annotationen von Gruppe 3 zuruecklesen und die Labels aktualisieren

Zwei Modi:
- export: Erstellt Vorhersage-Dateien fuer den EDF-Browser
- import: Liest korrigierte Feedback-Dateien ein und aktualisiert die .npz Labels

Ausgabe: `DatabaseREMs/predictions/prediction_excerptX.txt`


## Wo finde ich die Ergebnisse

### Evaluationsergebnisse (Metriken und Plots)

Scratch-Modell (Patienten 1-7 trainiert, 8 und 9 getestet):
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/classification_report_patient_8.txt`
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/classification_report_patient_9.txt`
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/dreams_confusion_matrix_patient_8.png`
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/dreams_confusion_matrix_patient_9.png`
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/dreams_roc_curve_patient_8.png`
- `output/scratch/dreams_patients_1_2_3_4_5_6_7/dreams_roc_curve_patient_9.png`

Transfer-Modell (gleiche Struktur):
- `output/transfer/dreams_patients_1_2_3_4_5_6_7/`

### Vorhersagen fuer Gruppe 3 (EDF-Browser-Format)

- `DatabaseREMs/predictions/prediction_excerpt6.txt`
- `DatabaseREMs/predictions/prediction_excerpt9.txt`
- `DatabaseREMs/predictions/prediction_sleepedf_1.txt`
- `DatabaseREMs/predictions/prediction_sleepedf_2.txt`


### Trainingsparameter

- `output/scratch/dreams_patients_1_2_3_4_5_6_7/parameters.txt`
- `output/transfer/dreams_patients_1_2_3_4_5_6_7/parameters.txt`


## Modellarchitektur

Das Modell besteht aus zwei Komponenten:

1. SE-ResNet-18: Ein 1D Convolutional Neural Network mit Squeeze-and-Excitation Bloecken. Extrahiert Features aus den 200 Datenpunkten jeder 2-Sekunden-Epoche.

2. Transformer-Encoder: Verarbeitet die CNN-Features mit Positional Encoding und Multi-Head Self-Attention. Klassifiziert jede Epoche als REM-Ereignis oder Nicht-REM.

Hyperparameter:
- Embedding-Groesse: 512
- Attention Heads: 8
- Hidden Size: 1024
- Transformer Layers: 2
- CNN Layers: [2, 2, 2, 2]
- Dropout: 0.1


## Post-Processing

Zwei Massnahmen zur Reduktion von Fehlalarmen:

1. Training: Wachphasen (Stadium 5) werden aus den Trainingsdaten entfernt, damit das Modell keine willkuerlichen Augenbewegungen im Wachzustand als REM lernt.

2. Test: Schlafphasen-Maskierung. Alle Vorhersagen ausserhalb des echten REM-Schlafs (Stadium 4) werden auf 0 gesetzt. Das betrifft die Stadien 0 (S4), 1 (S3), 2 (S2), 3 (S1) und 5 (Wake).


## Aktuelle Ergebnisse

Trainiert auf Patienten 1-7, getestet auf Patienten 8 und 9 mit Threshold 0.50 und Maskierung aktiv.

Patient 9 (Scratch-Modell):
- Accuracy: 94.78 Prozent
- Recall: 91.89 Prozent (34 von 37 REM-Ereignisse erkannt)
- Precision: 43.59 Prozent
- F1-Score: 59.13 Prozent
- ROC AUC: 0.9724

Patient 8 (Scratch-Modell, Stadium-5-Maske deaktiviert):
- Accuracy: 82.00 Prozent
- Recall: 51.97 Prozent (79 von 152 REM-Ereignisse erkannt)
- Precision: 47.02 Prozent
- F1-Score: 49.38 Prozent
- ROC AUC: 0.7282


## Experimentelle Skripte

Im Ordner scratch liegen zwei Skripte fuer Experimente:

- `experiment_scratch_vs_pretrained.py`: Automatisierter Vergleich zwischen Scratch und Transfer Learning mit verschiedenen Thresholds und Maskierungskonfigurationen
- `tune_evaluation.py`: Systematisches Tuning von Threshold und Maskierung auf verschiedenen Patienten


## Wichtige Hinweise

- Die .pth Modelldateien und .npz Daten werden per .gitignore nicht ins Repository gepusht. Nach dem Klonen muss zuerst die Vorverarbeitung und dann das Training ausgefuehrt werden.
- Die EDF-Rohdaten (excerpt1.edf bis excerpt9.edf) muessen im Ordner DatabaseREMs vorhanden sein.
- Alle Skripte nutzen relative Pfade ausgehend vom Projektverzeichnis. Sie koennen direkt aus ihrem jeweiligen Ordner ausgefuehrt werden.
- Die Einstellungen (Patienten, Threshold, Maskierung, Hyperparameter) werden direkt in den jeweiligen Skripten als Konstanten oben im Code angepasst.


## Abhaengigkeiten

- Python 3.8 oder hoeher
- PyTorch
- NumPy
- MNE (fuer EDF-Dateien)
- scikit-learn (fuer Metriken)
- matplotlib und seaborn (fuer Plots)


