"""Train CatBoost regressors for the bike sharing dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from timeseries_utils import (
    create_time_features,
    evaluate_metrics,
    get_categorical_columns,
    load_bike_data,
    prepare_catboost_frame,
    split_time_series,
)

RANDOM_SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_catboost(split, categorical_columns: List[str], horizon: int, n_trials: int) -> Dict[str, float]:
    """Hyperparameter search for CatBoost."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "iterations": trial.suggest_int("iterations", 300, 800),
        }
        model = CatBoostRegressor(
            loss_function="RMSE",
            random_state=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        X_train_df = prepare_catboost_frame(split.X_train, split.feature_names, categorical_columns)
        X_val_df = prepare_catboost_frame(split.X_val, split.feature_names, categorical_columns)
        model.fit(X_train_df, split.y_train, cat_features=categorical_columns if categorical_columns else None)
        preds = model.predict(X_val_df)
        rmse = evaluate_metrics(split.y_val, preds)["rmse"]
        return rmse

    study = optuna.create_study(direction="minimize", study_name=f"catboost_h{horizon}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_catboost(split, params: Dict[str, float], categorical_columns: List[str]) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    X_train_df = prepare_catboost_frame(split.X_train, split.feature_names, categorical_columns)
    X_val_df = prepare_catboost_frame(split.X_val, split.feature_names, categorical_columns)
    X_combined = pd.concat([X_train_df, X_val_df], axis=0)
    y_combined = np.concatenate([split.y_train, split.y_val])
    model.fit(X_combined, y_combined, cat_features=categorical_columns if categorical_columns else None)
    return model


def run(horizons: List[int], trials: int, output: Path) -> None:
    data = load_bike_data()
    results = []

    for horizon in horizons:
        frame, feature_cols, target_col = create_time_features(data, horizon)
        split = split_time_series(frame, feature_cols, target_col)
        categorical_columns = get_categorical_columns(split.feature_names)

        print(f"=== Horizon +{horizon}h | CatBoost tuning ({trials} trials) ===")
        best_params = tune_catboost(split, categorical_columns, horizon, trials)
        print("Best parameters:", best_params)

        model = train_catboost(split, best_params, categorical_columns)
        test_df = prepare_catboost_frame(split.X_test, split.feature_names, categorical_columns)
        preds = model.predict(test_df)
        metrics = evaluate_metrics(split.y_test, preds)
        metrics.update({"model": "CatBoost", "horizon": horizon})
        results.append(metrics)
        print("Test metrics:", metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatBoost forecasts for the bike sharing dataset.")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3], help="Forecast horizons to evaluate.")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna trials per horizon.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/catboost_metrics.json"),
        help="Path to save the JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.horizons, args.trials, args.output)
