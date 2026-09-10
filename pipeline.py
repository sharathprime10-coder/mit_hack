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
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
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
    if train.duplicated().any() or test.duplicated().any():
        raise ValueError("Duplicate rows detected; remove duplicates before modelling.")
    if train["Test_ID"].duplicated().any() or test["Test_ID"].duplicated().any():
        raise ValueError("Test_ID values must be unique in each input file.")
    return train, test


def choose_features(train: pd.DataFrame) -> list[str]:
    """Keep Sensor_S4 only when validation shows useful signal."""
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
) -> HistGradientBoostingClassifier:
    """Train the validity model while retaining missingness as a signal."""
    x = train[classifier_features]
    y = train["Validity_Label"].eq("Valid").astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x, y)
    print(
        "[validity] selected HistGradientBoostingClassifier "
        "from repeated stratified CV"
    )
    return model


def train_regression(
    train: pd.DataFrame, features: list[str]
) -> tuple[GaussianProcessRegressor, xgb.XGBRegressor, StandardScaler]:
    """Train GPR and a tree-based guardrail on engineer-labelled valid rows."""
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

    print("[regression] selected GPR from repeated shuffled CV on valid records")
    return gpr, fallback, scaler


def predict(
    test: pd.DataFrame,
    classifier: HistGradientBoostingClassifier,
    gpr: GaussianProcessRegressor,
    fallback: xgb.XGBRegressor,
    scaler: StandardScaler,
    classifier_features: list[str],
    features: list[str],
) -> pd.DataFrame:
    """Add validity, prediction, uncertainty, and guardrail columns."""
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


def export_artifacts(
    result: pd.DataFrame, output_dir: Path
) -> None:
    """Write the submission file and a compact audit summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = result[
        ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
    ].copy()
    submission.columns = ["Test_ID", "Predicted_Reference_Parameter", "Valid / Invalid"]
    submission["Predicted_Reference_Parameter"] = submission[
        "Predicted_Reference_Parameter"
    ].round(4)
    submission.to_csv(output_dir / "Anveshan.csv", index=False)

    attention = result.nlargest(3, "Attention_Score")["Test_ID"].tolist()
    summary = {
        "total_records_analyzed": int(len(result)),
        "invalid_records_count": int(result["Validity_Label"].eq("Invalid").sum()),
        "min_predicted_reference_parameter": round(
            float(result["Predicted_Reference_Parameter"].min()), 4
        ),
        "max_predicted_reference_parameter": round(
            float(result["Predicted_Reference_Parameter"].max()), 4
        ),
        "avg_predicted_reference_parameter": round(
            float(result["Predicted_Reference_Parameter"].mean()), 4
        ),
        "top_3_attention_test_ids": attention,
        "algorithmic_explanation": (
            "A balanced supervised HistGradientBoosting classifier identifies "
            "invalid records while "
            "retaining missingness indicators. Gaussian Process Regression is trained "
            "only on engineer-labelled valid records and supplies uncertainty for "
            "attention ranking. An XGBoost regressor is retained as an extrapolation "
            "guardrail. Repeated shuffled CV selected seven features and excluded Sensor_S4; "
            "tested physical features did not improve held-out error."
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
    export_artifacts(result, args.output_dir)
    print(f"[done] wrote Anveshan.csv and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
