"""Train an LSTM model for forecasting 1/2/3 hour horizons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from timeseries_utils import (
    create_sequence_data,
    evaluate_metrics,
    load_bike_data,
)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, horizons: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def prepare_sequence_datasets(series: np.ndarray, window: int, horizons: List[int], train_ratio=0.7, val_ratio=0.15):
    scaler = MinMaxScaler()
    train_cut = int(len(series) * train_ratio)
    val_cut = int(len(series) * (train_ratio + val_ratio))

    scaler.fit(series[:train_cut].reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).flatten()

    X, y = create_sequence_data(scaled, window, horizons)

    max_h = max(horizons)
    train_end = train_cut - window - max_h + 1
    val_end = val_cut - window - max_h + 1

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_ds = TensorDataset(torch.tensor(X_train[:, :, None], dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val[:, :, None], dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test[:, :, None], dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return train_ds, val_ds, test_ds, scaler


def train_lstm(train_ds: TensorDataset, val_ds: TensorDataset, horizons: int, epochs: int, lr: float) -> LSTMRegressor:
    model = LSTMRegressor(horizons=horizons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(xb), yb).item() for xb, yb in val_loader]
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(horizons: List[int], epochs: int, lr: float, output: Path) -> None:
    data = load_bike_data()
    series = data["y"].values
    window = 24
    train_ds, val_ds, test_ds, scaler = prepare_sequence_datasets(series, window, horizons)

    print(f"Training LSTM for horizons {horizons}")
    model = train_lstm(train_ds, val_ds, horizons=len(horizons), epochs=epochs, lr=lr)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(test_ds.tensors[0]).numpy()

    results = []
    for idx, horizon in enumerate(horizons):
        pred_column = preds_scaled[:, idx : idx + 1]
        true_column = test_ds.tensors[1].numpy()[:, idx : idx + 1]
        preds = scaler.inverse_transform(pred_column).ravel()
        targets = scaler.inverse_transform(true_column).ravel()
        metrics = evaluate_metrics(targets, preds)
        metrics.update({"model": "LSTM", "horizon": horizon})
        results.append(metrics)
        print(f"Horizon +{horizon}h metrics:", metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM forecasts for the bike sharing dataset.")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3, 4], help="Forecast horizons to evaluate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lstm_metrics.json"),
        help="Path to save the JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.horizons, args.epochs, args.lr, args.output)
