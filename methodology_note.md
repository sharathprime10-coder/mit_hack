# Team Anveshan - Methodology Note

## 1. Approach

We treated the task as a trust-validity-prediction pipeline. First, a
class-balanced `HistGradientBoostingClassifier` estimates whether each record
is `Valid` or `Invalid`. This uses engineer labels from both classes. Then a
`GaussianProcessRegressor` (GPR) predicts `Reference_Parameter` using
engineer-labelled valid rows only. The output includes GPR standard deviation,
so a reviewer can separate an ordinary prediction from one needing attention.
The classifier is a trust signal, not proof that a record is physically safe.

We compared GPR, `XGBRegressor`, `ExtraTreesRegressor`,
`RandomForestRegressor`, and `HistGradientBoostingRegressor` with repeated
shuffled 5-fold cross-validation (three repeats, `random_state=42`). GPR on
seven features, excluding `Sensor_S4`, was selected. Its CV results were R2
**0.9952 +/- 0.0009**, MAE **0.5150**, and RMSE **0.7366**. The classifier
results were F1 **0.9753 +/- 0.0064**, balanced accuracy **0.8698**, and ROC
AUC **0.9500**. The final models were refit on all available labelled data.

## 2. Important parameters and physical interpretation

The predictors are `Applied_Voltage_kV`, `Load_Current_A`,
`Ambient_Temperature_C`, `Test_Duration_min`, and `Sensor_S1` through
`Sensor_S3`. Current is physically relevant because resistive heating scales
with current squared. The sensor readings provide thermal context; voltage and
duration describe loading; ambient temperature provides a baseline. `Sensor_S4`
was excluded after the valid-record relationship and cross-validation checks
showed no useful contribution. This is a modelling decision, not a claim that
the sensor is intrinsically unimportant.

GPR uses standardized predictors, `normalize_y=True`, three optimizer
restarts, and a constant-kernel/RBF/white-noise kernel. The XGBoost model uses
300 trees, depth 4, learning rate 0.04, row and feature subsampling of 0.9,
and serves as a tree-based comparison and extrapolation guardrail. The
classifier uses 250 boosting iterations, 15 maximum leaf nodes, learning
rate 0.05, L2 regularization 1.0, balanced class weights, and seed 42.

## 3. Abnormal-data method

The classifier learns nonlinear combinations rather than declaring every
large value abnormal. For example, a high temperature can be consistent with
high current and long duration, while a sensor spike, missing critical
measurement, or inconsistent combination can lower validity. Numeric gaps are
filled with medians learned from training data; missingness indicators are
retained for classification because the gap itself may be informative.
Invalid labelled rows are not used to fit the physical regression. The
pipeline rejects missing required columns, unknown labels, duplicate rows, and
duplicate `Test_ID` values before training.

## 4. Assumptions and limitations

We assume engineer labels and reference parameters are reliable, trials are
independent, the relationship is reasonably stationary between campaigns,
and the selected electrical and thermal variables contain the useful signal.
No temporal or equipment grouping is available, so the reported CV is not a
time-forward or machine-held-out test. Median imputation is a baseline, not a
replacement for recovering a failed measurement. The model has no equipment
limit and should not be used alone for a safety decision. On the current run,
350 test records were processed and 31 were classified as invalid.

## 5. Digital-twin steps

The following is a future deployment plan, not part of the submitted batch
run:

1. **Ingest:** receive timestamped sensor records through an approved broker
   or plant interface.
2. **Prepare:** align timestamps, check ranges and schema, apply the
   training-derived preprocessing, and compute only physically validated
   derived variables.
3. **Assess validity:** return the classifier label and retain the reason
   signals available for review.
4. **Predict:** return the GPR estimate, uncertainty, and the XGBoost
   comparison.
5. **Review and learn:** route invalid or high-uncertainty records to an
   engineer, persist outcomes, and recalibrate only after new labelled data
   passes a documented validation process.

## 6. Running the supplied code

The program uses Python 3 and the following pinned packages:

```text
numpy==2.5.2
pandas==3.0.5
scikit-learn==1.9.0
xgboost==3.4.1
shap==0.52.0
```

Create a virtual environment and install them with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy==2.5.2 pandas==3.0.5 scikit-learn==1.9.0 xgboost==3.4.1 shap==0.52.0
python pipeline.py
```

The supplied `training_data.csv` and `test_data.csv` must be placed beside
`pipeline.py` before running it. The script then creates `Anveshan.csv` and
`summary.json`. The repository also contains the same dependency list in
`requirements.txt`.
