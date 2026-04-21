#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "baseline_csv": str(SCRIPT_DIR / "rolling baseline/rolling_baseline.csv"),
    "players_csv": str(SCRIPT_DIR / "data/box_scores/players.csv"),
    "teams_csv": str(SCRIPT_DIR / "data/box_scores/teams.csv"),
    "train_seasons": ["2023-24", "2024-25"],
    "test_seasons": ["2025-26"],
    "best_n_per_event": 1,
    "team_window": 20,
    "player_points_window": 40,
    "player_role_window": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a simple context-adjusted player-points model against the rolling baseline."
    )
    parser.add_argument(
        "--baseline-csv",
        default=DEFAULT_CONFIG["baseline_csv"],
        help="Rolling baseline CSV path.",
    )
    parser.add_argument(
        "--players-csv",
        default=DEFAULT_CONFIG["players_csv"],
        help="Box score players CSV path.",
    )
    parser.add_argument(
        "--teams-csv",
        default=DEFAULT_CONFIG["teams_csv"],
        help="Box score teams CSV path.",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        default=DEFAULT_CONFIG["train_seasons"],
        help="Seasons used to fit the context model.",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        default=DEFAULT_CONFIG["test_seasons"],
        help="Seasons used as holdout evaluation.",
    )
    parser.add_argument(
        "--best-n-per-event",
        type=int,
        default=DEFAULT_CONFIG["best_n_per_event"],
        help="Evaluate only the top N absolute-edge bets per event.",
    )
    parser.add_argument(
        "--team-window",
        type=int,
        default=DEFAULT_CONFIG["team_window"],
        help="Opponent team rolling window size.",
    )
    parser.add_argument(
        "--player-points-window",
        type=int,
        default=DEFAULT_CONFIG["player_points_window"],
        help="Player rolling points window size.",
    )
    parser.add_argument(
        "--player-role-window",
        type=int,
        default=DEFAULT_CONFIG["player_role_window"],
        help="Player rolling minutes/usage window size.",
    )
    return parser.parse_args()


def infer_season(game_date):
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def parse_minutes(value):
    if pd.isna(value) or value == "":
        return np.nan
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_baseline_rows(path):
    baseline = pd.read_csv(path, dtype={"game_id": str, "person_id": str})
    baseline["game_id"] = baseline["game_id"].str.lstrip("0")
    baseline["game_date"] = pd.to_datetime(baseline["game_date"])
    baseline["season"] = baseline["game_date"].map(infer_season)
    for column in [
        "rolling_mean_points",
        "rolling_median_points",
        "mean_minus_line",
        "median_minus_line",
        "line_points",
        "actual_points",
    ]:
        baseline[column] = baseline[column].astype(float)
    return baseline


def build_context_rows(players_csv, teams_csv, team_window, player_points_window, player_role_window):
    players = pd.read_csv(
        players_csv,
        dtype={"gameId": str, "teamId": str, "personId": str},
    )
    teams = pd.read_csv(
        teams_csv,
        dtype={"gameId": str, "teamId": str},
    )

    for df in (players, teams):
        df["gameId"] = df["gameId"].str.lstrip("0")
        df["gameDate"] = pd.to_datetime(df["gameDate"])

    players["minutes_float"] = players["minutes"].map(parse_minutes)
    players["isHomeInt"] = players["isHome"].astype(int)

    teams = teams.sort_values(["teamId", "gameDate", "gameId"]).copy()
    teams["opp_defensive_rating_roll"] = (
        teams.groupby("teamId")["defensiveRating"]
        .transform(lambda s: s.shift(1).rolling(team_window, min_periods=team_window).mean())
    )
    teams["opp_pace_roll"] = (
        teams.groupby("teamId")["pace"]
        .transform(lambda s: s.shift(1).rolling(team_window, min_periods=team_window).mean())
    )

    opponent_lookup = teams[
        ["gameId", "teamId", "opp_defensive_rating_roll", "opp_pace_roll"]
    ].rename(columns={"teamId": "opponentTeamId"})

    context = players.merge(opponent_lookup, on="gameId", how="left")
    context = context[context["teamId"] != context["opponentTeamId"]].copy()
    context = context.sort_values(["personId", "gameDate", "gameId"]).copy()

    context["player_minutes_roll"] = (
        context.groupby("personId")["minutes_float"]
        .transform(
            lambda s: s.shift(1).rolling(player_role_window, min_periods=player_role_window).mean()
        )
    )
    context["player_usage_roll"] = (
        context.groupby("personId")["usagePercentage"]
        .transform(
            lambda s: s.shift(1).rolling(player_role_window, min_periods=player_role_window).mean()
        )
    )
    context["player_points_roll_context"] = (
        context.groupby("personId")["points"]
        .transform(
            lambda s: s.shift(1).rolling(player_points_window, min_periods=player_points_window).mean()
        )
    )

    return context[
        [
            "gameId",
            "personId",
            "opp_defensive_rating_roll",
            "opp_pace_roll",
            "player_minutes_roll",
            "player_usage_roll",
            "player_points_roll_context",
            "isHomeInt",
        ]
    ].rename(columns={"gameId": "game_id", "personId": "person_id"})


def selected_price(row, pick_col):
    if row[pick_col] == "over":
        try:
            return float(row["over_price"])
        except (TypeError, ValueError):
            return np.nan
    if row[pick_col] == "under":
        try:
            return float(row["under_price"])
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def evaluate_top_n(df, score_col, pick_col, best_n):
    picked = df[df[pick_col] != "push"].copy()
    picked["edge_abs"] = picked[score_col].abs()
    picked = (
        picked.sort_values(["event_id", "edge_abs"], ascending=[True, False])
        .groupby("event_id", as_index=False)
        .head(best_n)
        .copy()
    )
    picked["selected_price"] = picked.apply(lambda row: selected_price(row, pick_col), axis=1)
    picked = picked[np.isfinite(picked["selected_price"])].copy()

    profits = np.where(
        picked[pick_col] == picked["actual_result"],
        picked["selected_price"] - 1.0,
        -1.0,
    )

    return {
        "selected_rows": len(picked),
        "events": picked["event_id"].nunique(),
        "correct": int((picked[pick_col] == picked["actual_result"]).sum()),
        "accuracy": float((picked[pick_col] == picked["actual_result"]).mean()),
        "profit": float(profits.sum()),
        "roi": float(profits.mean()),
    }


def fit_context_model(train_df, feature_cols):
    X = train_df[feature_cols].astype(float).to_numpy()
    y = (train_df["actual_points"] - train_df["rolling_mean_points"]).astype(float).to_numpy()

    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    feature_stds[feature_stds == 0.0] = 1.0

    Xz = (X - feature_means) / feature_stds
    design = np.column_stack([np.ones(len(Xz)), Xz])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    return coefficients, feature_means, feature_stds


def apply_context_model(df, feature_cols, coefficients, feature_means, feature_stds):
    scored = df.copy()
    X = scored[feature_cols].astype(float).to_numpy()
    Xz = (X - feature_means) / feature_stds
    predicted_residual = np.column_stack([np.ones(len(Xz)), Xz]) @ coefficients

    scored["predicted_points_context"] = scored["rolling_mean_points"] + predicted_residual
    scored["context_minus_line"] = scored["predicted_points_context"] - scored["line_points"]
    scored["context_pick"] = np.where(
        scored["context_minus_line"] > 0,
        "over",
        np.where(scored["context_minus_line"] < 0, "under", "push"),
    )

    return scored


def print_metrics(title, metrics):
    print(title)
    print(
        f"  selected_rows={metrics['selected_rows']} "
        f"events={metrics['events']} "
        f"correct={metrics['correct']} "
        f"accuracy={100.0 * metrics['accuracy']:.2f}% "
        f"profit={metrics['profit']:.3f} "
        f"roi={100.0 * metrics['roi']:.2f}%"
    )


def main():
    args = parse_args()

    baseline = load_baseline_rows(args.baseline_csv)
    context = build_context_rows(
        args.players_csv,
        args.teams_csv,
        args.team_window,
        args.player_points_window,
        args.player_role_window,
    )

    joined = baseline.merge(context, on=["game_id", "person_id"], how="left")

    feature_cols = [
        "opp_defensive_rating_roll",
        "opp_pace_roll",
        "player_minutes_roll",
        "player_usage_roll",
        "isHomeInt",
    ]
    model_df = joined.dropna(subset=feature_cols).copy()

    train_df = model_df[model_df["season"].isin(args.train_seasons)].copy()
    test_df = model_df[model_df["season"].isin(args.test_seasons)].copy()

    coefficients, feature_means, feature_stds = fit_context_model(train_df, feature_cols)

    train_scored = apply_context_model(train_df, feature_cols, coefficients, feature_means, feature_stds)
    test_scored = apply_context_model(test_df, feature_cols, coefficients, feature_means, feature_stds)

    print("Rows")
    print(f"  baseline_rows={len(baseline)}")
    print(f"  joined_usable_rows={len(model_df)}")
    print(f"  train_rows={len(train_df)}")
    print(f"  test_rows={len(test_df)}")
    print()

    print("Coefficients")
    print(f"  intercept={coefficients[0]:.6f}")
    for feature_name, coefficient in zip(feature_cols, coefficients[1:]):
        print(f"  {feature_name}={coefficient:.6f}")
    print()

    for split_name, split_df in [("Train", train_scored), ("Test", test_scored)]:
        baseline_metrics = evaluate_top_n(
            split_df,
            score_col="mean_minus_line",
            pick_col="mean_pick",
            best_n=args.best_n_per_event,
        )
        context_metrics = evaluate_top_n(
            split_df,
            score_col="context_minus_line",
            pick_col="context_pick",
            best_n=args.best_n_per_event,
        )

        print(split_name)
        print_metrics("  Rolling baseline", baseline_metrics)
        print_metrics("  Context-adjusted", context_metrics)
        print()


if __name__ == "__main__":
    main()
