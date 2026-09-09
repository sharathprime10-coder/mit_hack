"""
CPRI Data Challenge - Team Anveshan
Reproducible two-stage modelling pipeline.

Run from any directory:
    python pipeline.py

The script reads the CSV files beside this file and writes the submission
artifacts beside it, so the result is independent of the current directory.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]
REQUIRED_TRAIN_COLUMNS = {"Test_ID", "Reference_Parameter", "Validity_Label", *FEATURE_COLS}
REQUIRED_TEST_COLUMNS = {"Test_ID", *FEATURE_COLS}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the challenge files."""
    train_path = data_dir / "training_data.csv"
    test_path = data_dir / "test_data.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected training_data.csv and test_data.csv in {data_dir}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    missing_train = REQUIRED_TRAIN_COLUMNS - set(train.columns)
    missing_test = REQUIRED_TEST_COLUMNS - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Invalid input schema. Missing train columns: {sorted(missing_train)}; "
            f"missing test columns: {sorted(missing_test)}"
        )
    if not train["Validity_Label"].isin(["Valid", "Invalid"]).all():
        raise ValueError("Validity_Label must contain only 'Valid' or 'Invalid'.")
    return train, test


def choose_features(train: pd.DataFrame) -> list[str]:
    """Drop S4 only when it has no measurable relationship to the target."""
    valid = train.loc[train["Validity_Label"].eq("Valid")]
    correlation = valid["Sensor_S4"].corr(valid["Reference_Parameter"])
    if pd.isna(correlation) or abs(correlation) < 0.10:
        return [col for col in FEATURE_COLS if col != "Sensor_S4"]
    return FEATURE_COLS.copy()


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    """
    Impute from train medians while retaining missingness indicators.

    Missingness is useful for validity classification but is not passed to the
    regression model as a proxy for the physical target.
    """
    train = train.copy()
    test = test.copy()
    medians: dict[str, float] = {}
    missing_cols: list[str] = []
    for col in features:
        medians[col] = float(train[col].median())
        indicator = f"{col}__was_missing"
        train[indicator] = train[col].isna().astype(int)
        test[indicator] = test[col].isna().astype(int)
        if train[indicator].any() or test[indicator].any():
            missing_cols.append(indicator)
        train[col] = train[col].fillna(medians[col])
        test[col] = test[col].fillna(medians[col])
    return train, test, medians, missing_cols


def train_validity_classifier(
    train: pd.DataFrame, classifier_features: list[str]
) -> xgb.XGBClassifier:
    """Train a balanced classifier, with missingness retained as a signal."""
    x = train[classifier_features]
    y = train["Validity_Label"].eq("Valid").astype(int)
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )
    weights = np.where(y.to_numpy() == 0, 1.5, 1.0)
    model.fit(x, y, sample_weight=weights)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring="f1", params={"sample_weight": weights})
    print(f"[validity] 5-fold F1: {scores.mean():.4f} +/- {scores.std():.4f}")
    return model


def train_regression(
    train: pd.DataFrame, features: list[str]
) -> tuple[GaussianProcessRegressor, xgb.XGBRegressor, StandardScaler]:
    """Train GPR and a tree-based guardrail using engineer-labelled valid rows."""
    valid = train.loc[train["Validity_Label"].eq("Valid")]
    x = valid[features]
    y = valid["Reference_Parameter"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(1.0, (1e-2, 1e2))
        + WhiteKernel(1.0, (1e-5, 1e2))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        random_state=42,
        normalize_y=True,
    )
    gpr.fit(x_scaled, y)

    fallback = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=1,
        objective="reg:squarederror",
    )
    fallback.fit(x, y)

    fitted = gpr.predict(x_scaled)
    tree_fitted = fallback.predict(x)
    print(
        f"[regression] GPR R2={r2_score(y, fitted):.4f}, "
        f"MAE={mean_absolute_error(y, fitted):.4f}; "
        f"XGB MAE={mean_absolute_error(y, tree_fitted):.4f}"
    )
    return gpr, fallback, scaler


def predict(
    test: pd.DataFrame,
    classifier: xgb.XGBClassifier,
    gpr: GaussianProcessRegressor,
    fallback: xgb.XGBRegressor,
    scaler: StandardScaler,
    classifier_features: list[str],
    features: list[str],
) -> pd.DataFrame:
    """Generate validity, prediction, uncertainty, and guardrail columns."""
    result = test.copy()
    result["Validity_Label"] = np.where(
        classifier.predict(result[classifier_features]) == 1, "Valid", "Invalid"
    )
    prediction, uncertainty = gpr.predict(
        scaler.transform(result[features]), return_std=True
    )
    result["Predicted_Reference_Parameter"] = prediction
    result["GPR_Uncertainty"] = uncertainty
    result["XGB_Fallback_Prediction"] = fallback.predict(result[features])
    result["Attention_Score"] = result["GPR_Uncertainty"]
    return result


def generate_shap(
    fallback: xgb.XGBRegressor, test: pd.DataFrame, features: list[str]
) -> list[tuple[str, float]]:
    """Compute model explanations for the physical regression guardrail."""
    values = shap.TreeExplainer(fallback).shap_values(test[features])
    if isinstance(values, list):
        values = values[0]
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    return sorted(zip(features, mean_abs), key=lambda item: -item[1])


def export_artifacts(
    result: pd.DataFrame, importance: list[tuple[str, float]], output_dir: Path
) -> None:
    """Write the challenge submission and a judge-friendly audit summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = result[["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]].copy()
    submission.columns = ["Test_ID", "Predicted_Reference_Parameter", "Valid / Invalid"]
    submission["Predicted_Reference_Parameter"] = submission[
        "Predicted_Reference_Parameter"
    ].round(4)
    submission.to_csv(output_dir / "Anveshan.csv", index=False)

    attention = result.nlargest(3, "Attention_Score")["Test_ID"].tolist()
    highest_risk = result.nlargest(3, "Predicted_Reference_Parameter")["Test_ID"].tolist()
    summary = {
        "total_records_analyzed": int(len(result)),
        "invalid_records_count": int(result["Validity_Label"].eq("Invalid").sum()),
        "min_predicted_reference_parameter": round(float(result["Predicted_Reference_Parameter"].min()), 4),
        "max_predicted_reference_parameter": round(float(result["Predicted_Reference_Parameter"].max()), 4),
        "avg_predicted_reference_parameter": round(float(result["Predicted_Reference_Parameter"].mean()), 4),
        "top_3_attention_test_ids": attention,
        "top_3_highest_predicted_parameter_ids": highest_risk,
        "feature_importance": {name: round(float(value), 6) for name, value in importance},
        "algorithmic_explanation": (
            "A balanced supervised XGBoost classifier identifies invalid records while "
            "retaining missingness indicators. Gaussian Process Regression is trained "
            "only on engineer-labelled valid records and supplies uncertainty for "
            "attention ranking. An XGBoost regressor is retained as an extrapolation "
            "guardrail. Sensor_S4 is removed when its valid-record correlation is negligible."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Team Anveshan CPRI pipeline.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    args = parser.parse_args()

    train, test = load_data(args.data_dir)
    features = choose_features(train)
    train, test, _, missing_cols = preprocess(train, test, features)
    classifier_features = features + missing_cols
    print(f"[data] train={len(train)} rows, test={len(test)} rows")
    print(f"[data] regression features={features}")
    print(f"[data] validity features include {len(missing_cols)} missingness indicators")

    classifier = train_validity_classifier(train, classifier_features)
    gpr, fallback, scaler = train_regression(train, features)
    result = predict(
        test, classifier, gpr, fallback, scaler, classifier_features, features
    )
    importance = generate_shap(fallback, result, features)
    export_artifacts(result, importance, args.output_dir)
    print(f"[done] wrote Anveshan.csv and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
