"""Train a feed-forward neural network for each forecast horizon."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from timeseries_utils import create_time_features, evaluate_metrics, load_bike_data, split_time_series

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare_tensors(split) -> Tuple[TensorDataset, TensorDataset, TensorDataset, StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(split.X_train)
    X_val = scaler.transform(split.X_val)
    X_test = scaler.transform(split.X_test)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(split.y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(split.y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(split.y_test, dtype=torch.float32))
    return train_ds, val_ds, test_ds, scaler


def train_mlp(train_ds: TensorDataset, val_ds: TensorDataset, input_dim: int, epochs: int, lr: float) -> FeedForwardNet:
    model = FeedForwardNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                preds = model(xb).squeeze()
                val_losses.append(criterion(preds, yb).item())
            val_loss = float(np.mean(val_losses))
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(horizons: List[int], epochs: int, lr: float, output: Path) -> None:
    data = load_bike_data()
    results = []

    for horizon in horizons:
        frame, feature_cols, target_col = create_time_features(data, horizon)
        split = split_time_series(frame, feature_cols, target_col)
        train_ds, val_ds, test_ds, scaler = prepare_tensors(split)

        print(f"=== Horizon +{horizon}h | Feed-forward NN ===")
        model = train_mlp(train_ds, val_ds, input_dim=train_ds.tensors[0].shape[1], epochs=epochs, lr=lr)

        model.eval()
        with torch.no_grad():
            X_test_scaled = torch.tensor(scaler.transform(split.X_test), dtype=torch.float32)
            preds = model(X_test_scaled).squeeze().numpy()

        metrics = evaluate_metrics(split.y_test, preds)
        metrics.update({"model": "FeedForwardNN", "horizon": horizon})
        results.append(metrics)
        print("Test metrics:", metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feed-forward NN forecasts for the bike sharing dataset.")
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 2, 3, 4], help="Forecast horizons to evaluate.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mlp_metrics.json"),
        help="Path to save the JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.horizons, args.epochs, args.lr, args.output)
