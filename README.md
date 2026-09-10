# Team Anveshan - CPRI Data Challenge

This is the reproducible workflow we used for the CPRI test-record challenge.
The main idea is simple: **check whether a record is trustworthy before
deciding how much to trust its predicted reference parameter**. The pipeline
therefore makes a validity decision, predicts the reference parameter, and
adds uncertainty so a reviewer can see which rows deserve attention.

## Judge in 60 seconds

- We analysed **350 test records**; the final classifier marked **31 as
  Invalid**.
- The validity model is a class-balanced
  `HistGradientBoostingClassifier`: F1 **0.9753 +/- 0.0064**, balanced
  accuracy **0.8698**, ROC AUC **0.9500**.
- For records labelled valid by engineers, GPR was the best tested regressor:
  repeated-CV R2 **0.9952 +/- 0.0009**, MAE **0.5150**, RMSE **0.7366**.
- GPR is useful here because it gives both a prediction and an uncertainty
  estimate. We use that uncertainty for an attention ranking, not as a claim
  that the model knows the true physical state.
- An XGBoost regressor is kept as a tree-based comparison and extrapolation
  guardrail. The final submission is `Anveshan.csv`; the evidence and
  algorithmic explanation and attention ranking are in `summary.json`.

## Run the workflow

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py
```

The script reads `training_data.csv` and `test_data.csv` beside
`pipeline.py`, then writes `Anveshan.csv` and `summary.json` in the same
directory. It can also be run from another location:

```powershell
python C:\path\to\mit_hack\pipeline.py --data-dir C:\path\to\mit_hack
```

## What the pipeline does

1. Checks the schema, labels, duplicate rows, and duplicate IDs.
2. Selects the seven useful predictors and learns training-only medians for
   missing values. Missingness indicators remain available to the validity
   classifier.
3. Trains the validity classifier on both valid and invalid labelled rows.
4. Trains GPR and the XGBoost guardrail on engineer-labelled valid rows only.
5. Writes one row per test record with the prediction and validity label.
6. Stores uncertainty, attention IDs, and the algorithmic explanation in
   `summary.json`.

The physical interpretation is deliberately modest: current is the strongest
feature in the guardrail explanation, which is consistent with resistive
heating increasing with current squared; temperature sensors provide the
thermal context. This is evidence from the data, not a replacement for
engineering review.

## Files

| File | Purpose |
| --- | --- |
| `pipeline.py` | Reproducible training, prediction, explanation, and export |
| `training_data.csv` | Engineer-labelled historical records |
| `test_data.csv` | Unlabelled records used for inference |
| `Anveshan.csv` | Submission predictions |
| `summary.json` | Counts, attention ranking, and algorithmic explanation |
| `methodology_note.md` | Short explanation of approach and assumptions |
| `ARCHITECTURE.md` | Data flow, trust story, limitations, and future design |

The official submission ZIP intentionally contains exactly four files:
`Anveshan.csv`, `summary.json`, `pipeline.py`, and `methodology_note.md`.
