# CPRI Data Challenge - Team Anveshan

## 1. Approach

The pipeline uses two supervised stages. First, it classifies each record as
`Valid` or `Invalid`. Second, it predicts the Reference Parameter using only
engineer-labelled valid training records. The train/test files are schema
checked, duplicate full rows and duplicate `Test_ID` values are rejected, and
missing numeric values are median-imputed from training data. Missingness
indicators are retained for validity classification.

Candidate regressors were compared using repeated, shuffled 5-fold
cross-validation (three repeats, `random_state=42`) on the valid subset:
GaussianProcessRegressor (GPR), XGBRegressor, ExtraTreesRegressor,
RandomForestRegressor, and HistGradientBoostingRegressor. GPR using seven
features (excluding Sensor_S4) was selected:

| Model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| GPR | 0.9952 +/- 0.0009 | 0.5150 | 0.7366 |
| XGBRegressor | 0.9935 +/- 0.0027 | 0.5313 | 0.8416 |

ExtraTrees, RandomForest, and HistGradientBoosting were weaker. Adding
Sensor_S4 or tested physical features (`I^2 * duration`, thermal gradient,
sensor mean) did not improve held-out error. GPR also supplies predictive
uncertainty, which drives attention ranking. XGBoost remains a tree-based
extrapolation guardrail and supplies feature explanations.

For validity classification, repeated stratified 5-fold CV selected a balanced
HistGradientBoostingClassifier over XGBoost, ExtraTrees, and RandomForest:
F1 = 0.9753 +/- 0.0064, balanced accuracy = 0.8698, and ROC AUC = 0.9500.
The final model is refit on all labelled rows.

## 2. Important Parameters

The selected predictors are Applied_Voltage_kV, Load_Current_A,
Ambient_Temperature_C, Test_Duration_min, Sensor_S1, Sensor_S2, and Sensor_S3.
Load current is physically important because resistive heating scales with
current squared. Sensor_S2 and Sensor_S1 capture terminal temperatures and
thermal gradients; Sensor_S3 adds spatial context. Voltage and duration affect
electrical and thermal loading, while ambient temperature sets the baseline.
Sensor_S4 was excluded after valid-record correlation and cross-validation
checks showed no useful predictive contribution.

## 3. Abnormal-Data Method

The supervised classifier learns nonlinear conditional relationships rather
than treating every extreme reading as anomalous. Thus, high temperatures can
remain valid when supported by high current or long duration, while sensor
spikes, missing critical measurements, and physically inconsistent combinations
can be flagged as invalid. The model is class-balanced because invalid labels
are the minority class. Regression is never trained on invalid labelled rows.

## 4. Assumptions

1. Engineer-provided validity labels and reference parameters are trusted.
2. The physical relationship is sufficiently stationary between historical and
   test campaigns.
3. Records are independent trials; no temporal grouping is available.
4. Missing values are measurement gaps, and training medians are a safe
   baseline for imputation.
5. The selected electrical and thermal variables contain the relevant signal;
   Sensor_S4 is treated as auxiliary noise.

## 5. Digital-Twin Steps

1. **Ingest:** stream sensor records through MQTT or OPC-UA into a durable
   broker such as Google Pub/Sub or AWS IoT.
2. **Preprocess:** align timestamps, validate ranges, apply training-derived
   imputation, and compute approved derived variables if later validated.
3. **Validity service:** classify records as Valid or Invalid.
4. **Regression service:** predict the Reference Parameter for each record and
   return the GPR uncertainty.
5. **Monitor:** alert on invalid status, high prediction uncertainty, or a
   predicted parameter above the equipment limit; persist results for trends.

## 6. Reproducibility and Setup

Dependencies are pinned in `requirements.txt` (NumPy 2.5.2, pandas 3.0.5,
scikit-learn 1.9.0, XGBoost 3.4.1, and SHAP 0.52.0). From PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py
```

The pipeline reads `training_data.csv` and `test_data.csv` and writes the
official `Anveshan.csv` and `summary.json`. The submission ZIP contains
`Anveshan.csv`, `summary.json`, `pipeline.py`, and this methodology note.
