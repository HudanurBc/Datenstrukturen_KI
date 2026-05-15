# REM Detection — simples Projekt

Ziel: binäre Klassifikation von EOG-Fenstern.

- `0 = kein REM-Event`
- `1 = REM-Event`

Das Projekt ist bewusst einfach gehalten. Es nutzt die manuellen DREAMS-Annotationen `Visual_scoring1_*.txt` als Ground Truth. Die Hypnogramme werden zusätzlich als Metadata gespeichert, aber am Anfang nicht als direktes Label verwendet.

## 1. DREAMS-Daten einfügen

Lege die entpackten DREAMS-REMs-Dateien hier ab:

```text
data/raw/dreams_rems/
```

Dort sollten danach z. B. diese Dateien liegen:

```text
data/raw/dreams_rems/excerpt1.edf
data/raw/dreams_rems/Hypnogram_excerpt1.txt
data/raw/dreams_rems/Visual_scoring1_excerpt1.txt
data/raw/dreams_rems/Automatic_detection_excerpt1.txt
...
data/raw/dreams_rems/excerpt9.edf
```

Wichtig:

- `excerpt*.edf` = Signaldateien
- `Visual_scoring1_excerpt*.txt` = Ground Truth für REM-Events
- `Hypnogram_excerpt*.txt` = Schlafstadien-Kontext
- `Automatic_detection_excerpt*.txt` = nur Vergleich/Baseline, nicht fürs erste Training

## 2. Setup

Im Projektordner ausführen:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 3. Datensatz vorbereiten

```bash
python -m rem_detection.preprocessing.prepare_dataset
```

Danach entstehen:

```text
data/processed/windows.npy
data/processed/labels.npy
data/processed/metadata.csv
```

## 4. Modell trainieren

```bash
python -m rem_detection.model.train
```

Danach entstehen:

```text
models/round1/model.pth
models/round1/metrics.json
```

## 5. Modell auswerten

```bash
python -m rem_detection.model.evaluate
```

## 6. Vorhersagen exportieren

```bash
python -m rem_detection.model.predict
```

Danach entsteht:

```text
reports/predictions_for_review.csv
```

Diese Datei kann später für Human-in-the-loop oder EDF-Browser-Review genutzt werden.

## Projektstruktur

```text
rem_detection_simple/
├── configs/
│   ├── preprocessing.yaml
│   └── training.yaml
├── data/
│   ├── raw/
│   │   └── dreams_rems/        # Hier kommen deine DREAMS-Daten rein
│   └── processed/              # Wird automatisch erzeugt
├── models/
├── reports/
├── src/
│   └── rem_detection/
│       ├── preprocessing/
│       │   ├── load_edf.py
│       │   ├── load_hypnogram.py
│       │   ├── load_events.py
│       │   ├── filter_signal.py
│       │   ├── create_windows.py
│       │   └── prepare_dataset.py
│       ├── model/
│       │   ├── architecture.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── predict.py
│       └── utils/
│           └── config.py
├── requirements.txt
└── pyproject.toml
```

## Einfacher Workflow

```text
DREAMS-Dateien einfügen
        ↓
EDF + Visual Scoring + Hypnogramm laden
        ↓
EOG filtern
        ↓
2-Sekunden-Fenster schneiden
        ↓
Label setzen:
  1 = Fenster überlappt manuelles REM-Event
  0 = keine Überlappung
        ↓
CNN trainieren
        ↓
Precision / Recall / F1 auswerten
        ↓
Predictions für Review exportieren
```
