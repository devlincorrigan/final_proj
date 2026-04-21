#!/usr/bin/env python3

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CONFIG = {
    "mapping_csv": str(PROJECT_ROOT / "data/game_id_event_id_bijective.csv"),
    "players_csv": str(PROJECT_ROOT / "data/box_scores/players.csv"),
    "rolling_window": 20,
    "min_history": 20,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.10,
    "batch_size": 256,
    "epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple MLP for player points using rolling-average features and q50 pinball loss."
    )
    parser.add_argument("--mapping-csv", default=DEFAULT_CONFIG["mapping_csv"])
    parser.add_argument("--players-csv", default=DEFAULT_CONFIG["players_csv"])
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_CONFIG["rolling_window"])
    parser.add_argument("--min-history", type=int, default=DEFAULT_CONFIG["min_history"])
    parser.add_argument("--train-frac", type=float, default=DEFAULT_CONFIG["train_frac"])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_CONFIG["val_frac"])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_CONFIG["test_frac"])
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_game_id(game_id):
    return str(int(game_id))


def infer_season(game_date):
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def load_mapped_game_ids(mapping_csv):
    game_ids = set()
    with open(mapping_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            game_ids.add(canonical_game_id(row["game_id"]))
    return game_ids


def build_dataset(players_csv, mapping_csv, rolling_window, min_history):
    mapped_game_ids = load_mapped_game_ids(mapping_csv)

    players = pd.read_csv(
        players_csv,
        dtype={"gameId": str, "personId": str},
    )
    players["gameId"] = players["gameId"].map(canonical_game_id)
    players["gameDate"] = pd.to_datetime(players["gameDate"]).dt.date
    players["points"] = players["points"].astype(float)
    players["player_name"] = players["firstName"] + " " + players["familyName"]

    players = players.sort_values(["personId", "gameDate", "gameId"]).copy()
    players["rolling_points_mean"] = (
        players.groupby("personId")["points"]
        .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=min_history).mean())
    )
    players["history_games_used"] = (
        players.groupby("personId")["points"]
        .transform(lambda s: s.shift(1).rolling(rolling_window, min_periods=1).count())
    )
    players["season_inferred"] = players["gameDate"].map(infer_season)

    dataset = players[
        players["gameId"].isin(mapped_game_ids) & players["rolling_points_mean"].notna()
    ].copy()

    return dataset[
        [
            "season_inferred",
            "gameId",
            "gameDate",
            "personId",
            "player_name",
            "rolling_points_mean",
            "history_games_used",
            "points",
        ]
    ].rename(
        columns={
            "season_inferred": "season",
            "gameId": "game_id",
            "gameDate": "game_date",
            "points": "actual_points",
        }
    )


class QuantileMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def pinball_loss(predictions, targets, quantile=0.5):
    errors = targets - predictions
    return torch.maximum(quantile * errors, (quantile - 1.0) * errors).mean()


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values):
        return (values - self.mean) / self.std


def fit_standardizer(train_values):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0.0] = 1.0
    return Standardizer(mean=mean, std=std)


def make_loaders(train_x, train_y, val_x, val_y, batch_size):
    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate_predictions(targets, predictions, baseline_predictions):
    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def pinball(y_true, y_pred, q=0.5):
        errors = y_true - y_pred
        return float(np.mean(np.maximum(q * errors, (q - 1.0) * errors)))

    return {
        "mlp_mae": mae(targets, predictions),
        "baseline_mae": mae(targets, baseline_predictions),
        "mlp_pinball": pinball(targets, predictions),
        "baseline_pinball": pinball(targets, baseline_predictions),
    }


def chronological_split(df, train_frac, val_frac, test_frac):
    total_frac = train_frac + val_frac + test_frac
    if total_frac <= 0:
        raise ValueError("train/val/test fractions must sum to a positive value.")

    train_frac /= total_frac
    val_frac /= total_frac
    test_frac /= total_frac

    ordered = df.sort_values(["game_date", "game_id", "personId"]).reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Chronological split produced an empty train/val/test segment. "
            "Adjust the fractions."
        )

    return train_df, val_df, test_df


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = build_dataset(
        players_csv=args.players_csv,
        mapping_csv=args.mapping_csv,
        rolling_window=args.rolling_window,
        min_history=args.min_history,
    )

    train_df, val_df, test_df = chronological_split(
        dataset,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    fit_df = train_df.copy()

    feature_cols = ["rolling_points_mean"]
    fit_x = fit_df[feature_cols].to_numpy(dtype=np.float32)
    fit_y = fit_df["actual_points"].to_numpy(dtype=np.float32)
    val_x = val_df[feature_cols].to_numpy(dtype=np.float32)
    val_y = val_df["actual_points"].to_numpy(dtype=np.float32)
    test_x = test_df[feature_cols].to_numpy(dtype=np.float32)
    test_y = test_df["actual_points"].to_numpy(dtype=np.float32)

    standardizer = fit_standardizer(fit_x)
    fit_x = standardizer.transform(fit_x)
    val_x = standardizer.transform(val_x)
    test_x = standardizer.transform(test_x)

    train_loader, val_loader = make_loaders(
        fit_x, fit_y, val_x, val_y, args.batch_size
    )

    model = QuantileMLP(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = pinball_loss(preds, batch_y, quantile=0.5)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                val_losses.append(pinball_loss(preds, batch_y, quantile=0.5).item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train_pinball={mean_train_loss:.4f} val_pinball={mean_val_loss:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_predictions = (
            model(torch.tensor(test_x, dtype=torch.float32)).cpu().numpy()
        )

    baseline_predictions = test_df["rolling_points_mean"].to_numpy(dtype=np.float32)
    metrics = evaluate_predictions(test_y, test_predictions, baseline_predictions)

    print()
    print("Dataset")
    print(f"  total_rows={len(dataset)}")
    print(
        f"  split=train:{args.train_frac:.2f} "
        f"val:{args.val_frac:.2f} test:{args.test_frac:.2f}"
    )
    print(f"  train_rows={len(train_df)}")
    print(f"  fit_rows={len(fit_df)}")
    print(f"  val_rows={len(val_df)}")
    print(f"  test_rows={len(test_df)}")
    print()
    print("Evaluation")
    print(f"  mlp_mae={metrics['mlp_mae']:.4f}")
    print(f"  baseline_mae={metrics['baseline_mae']:.4f}")
    print(f"  mlp_q50_pinball={metrics['mlp_pinball']:.4f}")
    print(f"  baseline_q50_pinball={metrics['baseline_pinball']:.4f}")
    print()
    print("Sample Predictions")
    preview = test_df.copy()
    preview["predicted_points_mlp"] = test_predictions
    preview = preview[
        [
            "season",
            "game_date",
            "player_name",
            "rolling_points_mean",
            "predicted_points_mlp",
            "actual_points",
        ]
    ].head(10)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
