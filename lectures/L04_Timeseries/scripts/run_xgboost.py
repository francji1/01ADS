"""Train XGBoost regressors for 1/2/3 hour horizons on the bike sharing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import optuna
import numpy as np
from xgboost import XGBRegressor

from timeseries_utils import (
    SplitData,
    create_time_features,
    evaluate_metrics,
    load_bike_data,
    split_time_series,
)

RANDOM_SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgb(split: SplitData, horizon: int, n_trials: int) -> Dict[str, float]:
    """Optuna search for XGBoost parameters."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            tree_method="hist",
            **params,
        )
        model.fit(split.X_train, split.y_train, eval_set=[(split.X_val, split.y_val)], verbose=False)
        preds = model.predict(split.X_val)
        rmse = evaluate_metrics(split.y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize", study_name=f"xgb_h{horizon}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_xgb(split: SplitData, params: Dict[str, float]) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        tree_method="hist",
        **params,
    )
    X_combined = np.vstack([split.X_train, split.X_val])
    y_combined = np.concatenate([split.y_train, split.y_val])
    model.fit(X_combined, y_combined, verbose=False)
    return model


def run(horizons: List[int], trials: int, output: Path) -> None:
    data = load_bike_data()
    results = []

    for horizon in horizons:
        frame, feature_cols, target_col = create_time_features(data, horizon)
        split = split_time_series(frame, feature_cols, target_col)

        print(f"=== Horizon +{horizon}h | XGBoost tuning ({trials} trials) ===")
        best_params = tune_xgb(split, horizon, trials)
        print("Best parameters:", best_params)

        model = train_xgb(split, best_params)
        preds = model.predict(split.X_test)
        metrics = evaluate_metrics(split.y_test, preds)
        metrics.update({"model": "XGBoost", "horizon": horizon})
        results.append(metrics)
        print("Test metrics:", metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost forecasts for the bike sharing dataset.")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3], help="Forecast horizons to evaluate.")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials per horizon.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/xgboost_metrics.json"),
        help="Path to save the JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.horizons, args.trials, args.output)
