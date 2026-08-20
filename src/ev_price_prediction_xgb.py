"""Train a portable XGBoost baseline and create an EV-price submission CSV.

The original competition notebook remains under ``notebooks/``. This CLI
replaces its machine-specific export; the competition data is not versioned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:
    from .config import RANDOM_SEED, TEST_CSV, TEST_SIZE, TRAIN_CSV, result_path
except ImportError:  # Supports ``python src/ev_price_prediction_xgb.py``.
    from config import RANDOM_SEED, TEST_CSV, TEST_SIZE, TRAIN_CSV, result_path

TARGET = "가격(백만원)"
ID_COLUMN = "ID"
BATTERY = "배터리용량"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create documented features only when their source columns are present."""
    data = frame.copy()
    if {"제조사", "모델", "차량상태"}.issubset(data):
        data["제조사_모델_상태"] = data[["제조사", "모델", "차량상태"]].astype(str).agg("_".join, axis=1)
    if {"주행거리(km)", "연식(년)"}.issubset(data):
        age = pd.to_numeric(data["연식(년)"], errors="coerce").clip(lower=0)
        mileage = pd.to_numeric(data["주행거리(km)"], errors="coerce")
        data["주행거리비율"] = mileage.div(age.mask(age.eq(0)))
        data["사용연한주행거리비율"] = mileage.div(age.add(1))
    if {"주행거리(km)", BATTERY}.issubset(data):
        battery = pd.to_numeric(data[BATTERY], errors="coerce").mask(lambda value: value.eq(0))
        data["전비(km/kWh)"] = pd.to_numeric(data["주행거리(km)"], errors="coerce").div(battery)
    if "차량상태" in data:
        data["중고여부"] = data["차량상태"].ne("Brand New").astype(int)
    if "구동방식" in data:
        data["구동방식가중치"] = data["구동방식"].map({"AWD": 1.2, "RWD": 1.1, "FWD": 1.0})
    if "사고이력" in data:
        data["사고가중치"] = data["사고이력"].eq("Yes").astype(int)
    return data


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = features.columns.difference(categorical).tolist()
    return ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
         ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)],
        remainder="drop",
    )


def impute_battery(reference: pd.DataFrame, *frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Fit battery imputation on observed reference rows and apply it to all frames."""
    all_frames = (reference, *frames)
    if any(BATTERY not in frame for frame in all_frames):
        return all_frames
    predictors = [column for column in ["제조사", "모델", "차량상태", "구동방식", "주행거리(km)", "보증기간(년)"] if column in reference]
    observed = reference.loc[reference[BATTERY].notna(), predictors]
    if not predictors or observed.empty:
        return all_frames
    model = Pipeline([("preprocessor", make_preprocessor(observed)), ("regressor", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_SEED, objective="reg:squarederror"))])
    model.fit(observed, reference.loc[reference[BATTERY].notna(), BATTERY])
    for frame in all_frames:
        missing = frame[BATTERY].isna()
        if missing.any():
            frame.loc[missing, BATTERY] = model.predict(frame.loc[missing, predictors])
    return all_frames


def run(train_path: Path, test_path: Path, output_path: Path, metrics_path: Path) -> float:
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Provide DACON train.csv and test.csv via --train-csv/--test-csv or EV_PRICE_DATA_DIR.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if TARGET not in train or ID_COLUMN not in test:
        raise KeyError(f"Expected training target '{TARGET}' and test ID column '{ID_COLUMN}'.")
    development, validation = train_test_split(train, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    development, validation, _ = impute_battery(development.copy(), validation.copy(), test.copy())
    development, validation = add_features(development), add_features(validation)
    features = [column for column in validation.columns if column != ID_COLUMN and column in development.columns and column != TARGET]
    X_train, y_train = development[features], development[TARGET]
    X_valid, y_valid = validation[features], validation[TARGET]
    params = dict(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, objective="reg:squarederror")
    model = Pipeline([("preprocessor", make_preprocessor(X_train)), ("regressor", XGBRegressor(**params))])
    model.fit(X_train, y_train)
    rmse = mean_squared_error(y_valid, model.predict(X_valid)) ** 0.5
    full_train, full_test = impute_battery(train.copy(), test.copy())
    full_train, full_test = add_features(full_train), add_features(full_test)
    final_features = [column for column in full_test.columns if column != ID_COLUMN and column in full_train.columns and column != TARGET]
    final_model = Pipeline([("preprocessor", make_preprocessor(full_train[final_features])), ("regressor", XGBRegressor(**params))])
    final_model.fit(full_train[final_features], full_train[TARGET])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: full_test[ID_COLUMN], TARGET: final_model.predict(full_test[final_features])}).to_csv(output_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"validation_rmse": rmse, "split": "single 80/20 random holdout", "random_seed": RANDOM_SEED}, indent=2), encoding="utf-8")
    return rmse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an XGBoost EV-price model and create a submission CSV.")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--output-csv", type=Path, default=result_path("submission.csv"))
    parser.add_argument("--metrics-json", type=Path, default=result_path("metrics.json"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(f"Validation RMSE: {run(args.train_csv, args.test_csv, args.output_csv, args.metrics_json):.4f}")
