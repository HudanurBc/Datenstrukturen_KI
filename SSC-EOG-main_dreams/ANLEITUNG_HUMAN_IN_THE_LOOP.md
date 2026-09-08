# Anleitung: Human-in-the-Loop und Nachtraining

Diese Anleitung beschreibt den kompletten Ablauf fuer die DREAMS-Pipeline:

1. Einstellungen in `settings.py` pruefen
2. menschlich korrigierte Daten in `reviewed_data` ablegen
3. Review-Datei in `settings.py` auswaehlen
4. Nachtraining starten
5. neues Modell evaluieren und Plots ansehen

## 1. Richtigen Projektordner verwenden

Empfohlen wird die GitHub-Arbeitskopie:

```text
C:\Users\sfrt15\Desktop\Persoenlich\Gross&Fischer\Datenstrukturen_KI-github
```

Der zentrale DREAMS-Ordner ist:

```text
Datenstrukturen_KI-github\SSC-EOG-main_dreams
```

Die Python-Umgebung liegt hier:

```text
Datenstrukturen_KI-github\.venv
```

PowerShell oeffnen und Umgebung aktivieren:

```powershell
cd "C:\Users\sfrt15\Desktop\Persönlich\Groß&Fischer\Datenstrukturen_KI-github"
.\.venv\Scripts\Activate.ps1
```

## 2. Zentrale Einstellungen pruefen

Alle wichtigen Einstellungen stehen in:

```text
SSC-EOG-main_dreams\settings.py
```

Wichtige Werte:

```python
WINDOW_DURATION_SECONDS = 2.0
WINDOW_STEP_SECONDS = 1.0
USE_HYPNOGRAM_FOR_TRAINING = False
MASK_STAGES = []
```

Das bedeutet:

- 2-Sekunden-EOG-Fenster
- 1-Sekunde Schrittweite
- 50 Prozent Fensterueberlappung
- Training ohne Hypnogramm-Filter
- Evaluation ohne Hypnogramm-Maske

Einstellungen fuer das Nachtraining:

```python
REVIEWED_DATA_FILE = "../reviewed_data/pat8-korriegiert.csv"
REVIEWED_PATIENT = 8
FINETUNE_EPOCHS = 5
FINETUNE_BATCH_SIZE = 32
FINETUNE_LEARNING_RATE = 0.00005
FINETUNE_OUTPUT_MODEL = "dreams_model_feedback_patient8.pth"
REQUIRE_REVIEW_ORIGINAL_MATCH = True
```

### Welche Review-Datei wird verwendet?

Ausschliesslich der Wert `REVIEWED_DATA_FILE` entscheidet, welche Datei beim
Nachtraining gelesen wird. Beispiel:

```python
REVIEWED_DATA_FILE = "../reviewed_data/pat8-korriegiert.csv"
```

Die Datei liegt dann hier:

```text
Datenstrukturen_KI-github\reviewed_data\pat8-korriegiert.csv
```

Fuer Patient 9 waere es zum Beispiel:

```python
REVIEWED_PATIENT = 9
REVIEWED_DATA_FILE = "../reviewed_data/pat9-korrigiert.csv"
```

## 3. Menschlich korrigierte Dateien ablegen

Die korrigierte CSV-Datei in diesen Ordner kopieren:

```text
Datenstrukturen_KI-github\reviewed_data\
```

Beispiel:

```text
reviewed_data\pat8-korriegiert.csv
```

Die CSV muss mindestens diese Spalten enthalten:

```text
epoch,final_binary
```

Empfohlenes vollstaendiges Format:

```text
epoch,start_seconds,original_binary,auto_suggestion,final_binary,corrected
0,0.0,Non-REM,REM,Non-REM,True
1,1.0,Non-REM,Non-REM,Non-REM,False
2,2.0,REM,Non-REM,REM,True
```

Erlaubte Labelwerte sind:

```text
REM
Non-REM
```

Alternativ sind `1` und `0` erlaubt.

## 4. Wichtige Sicherheitspruefung

Die Review-CSV muss exakt zur aktuell vorverarbeiteten NPZ-Datei passen:

- gleiche Anzahl Fenster
- Epochennummern lueckenlos von `0` bis `N-1`
- gleiche Fensterung
- gleiche urspruengliche Labels, falls `original_binary` vorhanden ist

Bei der aktuellen Konfiguration muss Patient 8 insgesamt `1799` Fenster haben.

Die Datei `pat8-korriegiert.csv` mit `1537` Fenstern stammt aus einer anderen
Fensterung und wird deshalb vom Skript absichtlich abgelehnt. Diese Datei nicht
manuell auffuellen oder verschieben. Eine neue Review-Datei mit allen `1799`
aktuellen Fenstern erzeugen lassen.

## 5. Vorverarbeitete Daten und Basismodell

Vor dem Nachtraining muessen vorhanden sein:

```text
SSC-EOG-main_dreams\DatabaseREMs\preprocessed\patient_8.npz
SSC-EOG-main_dreams\output\scratch\dreams_patients_1_2_3_4_5_6_7\dreams_model.pth
```

Falls die NPZ-Dateien fehlen:

```powershell
.\.venv\Scripts\python.exe `
"SSC-EOG-main_dreams\dreams_pipeline\1_data_engineering\preprocess_dreams.py"
```

Falls das Basismodell fehlt, zuerst trainieren:

```powershell
.\.venv\Scripts\python.exe `
"SSC-EOG-main_dreams\dreams_pipeline\2_model_architecture\train_dreams.py"
```

## 6. Nachtraining starten

Nach dem Einlegen der passenden Review-CSV und der Anpassung von
`REVIEWED_DATA_FILE` in `settings.py`:

```powershell
.\.venv\Scripts\python.exe `
"SSC-EOG-main_dreams\dreams_pipeline\2_model_architecture\finetune_reviewed.py"
```

Das Skript:

- liest die ausgewaehlte CSV
- prueft Fensteranzahl und Epochennummern
- prueft die urspruenglichen Labels
- laedt das letzte DREAMS-Basismodell
- trainiert mit den menschlich korrigierten Labels weiter
- veraendert das Basismodell nicht

Das neue Modell wird gespeichert unter:

```text
SSC-EOG-main_dreams\output\feedback\dreams_model_feedback_patient8.pth
```

## 7. Feedback-Modell evaluieren

Standardmaessig wird das Feedback-Modell auf Patient 9 getestet:

```powershell
.\.venv\Scripts\python.exe `
"SSC-EOG-main_dreams\dreams_pipeline\3_evaluation\evaluate_feedback_model.py"
```

Das erzeugt:

```text
SSC-EOG-main_dreams\output\feedback\classification_report_patient_9.txt
SSC-EOG-main_dreams\output\feedback\feedback_confusion_matrix_patient_9.png
SSC-EOG-main_dreams\output\feedback\feedback_roc_curve_patient_9.png
```

Wenn ein anderer Testpatient verwendet werden soll, `PATIENT` oben in
`evaluate_feedback_model.py` anpassen. Der Patient darf nicht derselbe sein,
der fuer das Nachtraining verwendet wurde.

## 8. Technische Testvorlage

Zum Testen des Ablaufs ohne neue menschliche Labels gibt es:

```powershell
.\.venv\Scripts\python.exe `
"SSC-EOG-main_dreams\dreams_pipeline\2_model_architecture\create_review_template.py"
```

Die Vorlage wird erzeugt als:

```text
reviewed_data\pat8-review-template.csv
```

Diese Datei enthaelt keine echten menschlichen Korrekturen. Sie darf nur fuer
einen technischen Funktionstest verwendet werden und nicht als Ergebnis einer
fachlichen Evaluation interpretiert werden.

## 9. Typische Fehler

### CSV=1537, NPZ=1799

Die Review-Datei stammt aus einer anderen Fensterung. Eine neue vollstaendige
Review-Datei mit derselben Konfiguration erzeugen lassen.

### Review-Datei nicht gefunden

`REVIEWED_DATA_FILE` in `settings.py` pruefen. Der Pfad ist relativ zum Ordner
`SSC-EOG-main_dreams`.

### Kein Basismodell gefunden

Zuerst `train_dreams.py` ausfuehren oder pruefen, ob `dreams_model.pth` im
Scratch-Output liegt.

### Neue Plots nicht sichtbar

Sicherstellen, dass der richtige Projektordner geoeffnet ist. Die Ausgabe liegt
unter:

```text
SSC-EOG-main_dreams\output\feedback\
```

## 10. Kurzablauf

```text
Review-CSV erhalten
    |
    v
reviewed_data ablegen
    |
    v
REVIEWED_DATA_FILE in settings.py waehlen
    |
    v
Fensteranzahl pruefen lassen
    |
    v
finetune_reviewed.py starten
    |
    v
Feedback-Modell in output/feedback/
    |
    v
evaluate_feedback_model.py starten
    |
    v
Report, Confusion Matrix und ROC-Kurve pruefen
```
