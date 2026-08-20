# Used-EV Price Forecasting: Separating Battery Imputation from Price Regression

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

Used-EV valuation is not a simple mileage-and-age problem. Depreciation differs
by manufacturer, model, condition, and drivetrain, while battery capacity—a
potentially important value driver—is missing in some rows. This DACON project
treats that missing field as a prediction problem before training the final
price regressor.

## Key decision

Battery capacity is imputed with a separate XGBoost model trained only on
observed training records. Manufacturer, model, vehicle condition, drivetrain,
mileage, and warranty fields act as predictors. Fitting this step on the
development partition avoids learning from validation values.

The price workflow then creates manufacturer/model/state combinations, mileage
and age ratios, efficiency, used-vehicle, drivetrain, and accident-history
features when source columns exist. Numeric values use median imputation;
categorical values use one-hot encoding before XGBoost regression.

## Validation and result

The maintained CLI uses a seeded 80/20 holdout, writes local RMSE to
`results/metrics.json`, and trains on full training data for submission. The
user-confirmed DACON normalized leaderboard RMSE is **0.919**.

The original data, submission, and leaderboard capture are not retained, so a
new local holdout metric is not directly comparable with the leaderboard value.

## What this demonstrates

The project separates imputation from the target model instead of treating all
missing values as a mean-fill problem, then packages the notebook workflow as
a configurable command-line path.

## Limitations

One random holdout is not time, group, or external validation. A next iteration
should retain the original submission and compare mean, model-based, and
missing-indicator treatments under the same split.

## Evidence

- [`src/ev_price_prediction_xgb.py`](src/ev_price_prediction_xgb.py)
- [`notebooks/EV_price_prediction_xgb.ipynb`](notebooks/EV_price_prediction_xgb.ipynb)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)

## Why battery imputation is a separate model

Battery capacity is not a minor auxiliary field in used-EV pricing. Its relationship to residual value can differ by make, model, drivetrain, condition, mileage, and warranty. The project trains an imputation model only on rows with observed capacity, applies it to missing rows, and then passes the completed feature to the price model. The order avoids training the imputer on validation information or allowing the target price to fill the missing value.

The final regressor uses make/model/condition combinations, mileage per year, efficiency in km/kWh, used status, drivetrain, and accident-history weighting. These features are intended to represent differences in depreciation patterns beyond a single mileage rule. Categorical values use one-hot encoding and ordinary numeric missingness uses median imputation.

The user-confirmed DACON normalized leaderboard RMSE of **0.919** remains the actual competition result. The current checkout does not retain the original data, submission CSV, or leaderboard capture, so a new 80/20 local holdout cannot be treated as the same metric. Future provenance should retain the data version, submission, feature list, and the separate imputation/price-model split.
