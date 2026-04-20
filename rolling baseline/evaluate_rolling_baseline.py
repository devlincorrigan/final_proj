#!/usr/bin/env python3

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "input": str(SCRIPT_DIR / "rolling_baseline.csv"),
    "edge_buckets": [0.0, 1.0, 2.0, 3.0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline player-points predictions."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_CONFIG["input"],
        help="Baseline CSV produced by baseline.py.",
    )
    parser.add_argument(
        "--best-n-per-event",
        type=int,
        default=0,
        help="If > 0, only evaluate the top N absolute-edge picks per event.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[],
        help="Optional season filters like 2023-24 2024-25.",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_season(game_date_text):
    game_date = date.fromisoformat(game_date_text)
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def filter_rows_by_seasons(rows, seasons):
    if not seasons:
        return rows
    season_set = set(seasons)
    return [row for row in rows if infer_season(row["game_date"]) in season_set]


def to_float(row, key):
    return float(row[key])


def to_int(row, key):
    return int(row[key])


def maybe_float(value):
    if value in ("", None):
        return float("nan")
    return float(value)


def accuracy(rows, pick_key):
    eligible = [row for row in rows if row["actual_result"] != "push"]
    decided = [row for row in eligible if row[pick_key] != "push"]
    correct = sum(1 for row in decided if row[pick_key] == row["actual_result"])
    return {
        "eligible": len(eligible),
        "decided": len(decided),
        "correct": correct,
        "accuracy": (correct / len(decided)) if decided else float("nan"),
    }


def select_best_n_per_event(rows, score_key, pick_key, best_n_per_event):
    if best_n_per_event <= 0:
        return rows

    grouped = defaultdict(list)
    for row in rows:
        if row[pick_key] == "push":
            continue
        grouped[row["event_id"]].append(row)

    selected = []
    for event_rows in grouped.values():
        ranked = sorted(
            event_rows,
            key=lambda row: abs(to_float(row, score_key)),
            reverse=True,
        )
        selected.extend(ranked[:best_n_per_event])

    return selected


def regression_metrics(rows, projection_key):
    errors = [to_float(row, projection_key) - to_int(row, "actual_points") for row in rows]
    mae = sum(abs(error) for error in errors) / len(errors) if errors else float("nan")
    rmse = (
        math.sqrt(sum(error * error for error in errors) / len(errors))
        if errors
        else float("nan")
    )
    return {"count": len(errors), "mae": mae, "rmse": rmse}


def selected_price(row, pick_key):
    pick = row[pick_key]
    if pick == "over":
        return maybe_float(row["over_price"])
    if pick == "under":
        return maybe_float(row["under_price"])
    return float("nan")


def roi_metrics(rows, pick_key):
    eligible = [
        row for row in rows if row["actual_result"] != "push" and row[pick_key] != "push"
    ]
    if not eligible:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": float("nan"),
            "avg_price": float("nan"),
            "total_profit": float("nan"),
            "roi": float("nan"),
        }

    wins = 0
    total_profit = 0.0
    total_price = 0.0

    for row in eligible:
        price = selected_price(row, pick_key)
        if math.isnan(price):
            continue
        total_price += price
        if row[pick_key] == row["actual_result"]:
            wins += 1
            total_profit += price - 1.0
        else:
            total_profit -= 1.0

    priced_bets = wins + sum(
        1
        for row in eligible
        if not math.isnan(selected_price(row, pick_key)) and row[pick_key] != row["actual_result"]
    )
    if priced_bets == 0:
        return {
            "bets": 0,
            "wins": 0,
            "win_rate": float("nan"),
            "avg_price": float("nan"),
            "total_profit": float("nan"),
            "roi": float("nan"),
        }

    return {
        "bets": priced_bets,
        "wins": wins,
        "win_rate": wins / priced_bets,
        "avg_price": total_price / priced_bets,
        "total_profit": total_profit,
        "roi": total_profit / priced_bets,
    }


def bucket_label(value, bucket_edges):
    for start, end in zip(bucket_edges, bucket_edges[1:]):
        if start <= value < end:
            return f"[{start:.0f}, {end:.0f})"
    return f"[{bucket_edges[-1]:.0f}, +inf)"


def accuracy_by_edge(rows, margin_key, pick_key, bucket_edges):
    grouped = defaultdict(list)

    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        edge = abs(to_float(row, margin_key))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    results = []
    all_labels = [f"[{start:.0f}, {end:.0f})" for start, end in zip(bucket_edges, bucket_edges[1:])]
    all_labels.append(f"[{bucket_edges[-1]:.0f}, +inf)")

    for label in all_labels:
        bucket_rows = grouped.get(label, [])
        correct = sum(1 for row in bucket_rows if row[pick_key] == row["actual_result"])
        results.append(
            {
                "bucket": label,
                "count": len(bucket_rows),
                "accuracy": (correct / len(bucket_rows)) if bucket_rows else float("nan"),
            }
        )

    return results


def roi_by_edge(rows, margin_key, pick_key, bucket_edges):
    grouped = defaultdict(list)

    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        edge = abs(to_float(row, margin_key))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    results = []
    all_labels = [f"[{start:.0f}, {end:.0f})" for start, end in zip(bucket_edges, bucket_edges[1:])]
    all_labels.append(f"[{bucket_edges[-1]:.0f}, +inf)")

    for label in all_labels:
        bucket_rows = grouped.get(label, [])
        roi = roi_metrics(bucket_rows, pick_key)
        results.append(
            {
                "bucket": label,
                "count": roi["bets"],
                "profit": roi["total_profit"],
                "roi": roi["roi"],
            }
        )

    return results


def counts_by_field(rows, field):
    counter = Counter(row[field] for row in rows)
    return sorted(counter.items())


def format_pct(value):
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def format_float(value):
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def print_summary(rows):
    print(f"Rows: {len(rows)}")
    pushes = sum(1 for row in rows if row["actual_result"] == "push")
    print(f"Pushes: {pushes}")
    print()


def print_accuracy(rows, best_n_per_event):
    mean_rows = select_best_n_per_event(
        rows, "mean_minus_line", "mean_pick", best_n_per_event
    )
    median_rows = select_best_n_per_event(
        rows, "median_minus_line", "median_pick", best_n_per_event
    )
    mean_stats = accuracy(mean_rows, "mean_pick")
    median_stats = accuracy(median_rows, "median_pick")

    print("Directional Accuracy")
    print(
        f"mean_pick: correct={mean_stats['correct']} "
        f"decided={mean_stats['decided']} "
        f"eligible={mean_stats['eligible']} "
        f"accuracy={format_pct(mean_stats['accuracy'])}"
    )
    print(
        f"median_pick: correct={median_stats['correct']} "
        f"decided={median_stats['decided']} "
        f"eligible={median_stats['eligible']} "
        f"accuracy={format_pct(median_stats['accuracy'])}"
    )
    print()


def print_regression(rows):
    mean_metrics = regression_metrics(rows, "rolling_mean_points")
    median_metrics = regression_metrics(rows, "rolling_median_points")

    print("Projection Error")
    print(
        f"rolling_mean_points: count={mean_metrics['count']} "
        f"mae={format_float(mean_metrics['mae'])} "
        f"rmse={format_float(mean_metrics['rmse'])}"
    )
    print(
        f"rolling_median_points: count={median_metrics['count']} "
        f"mae={format_float(median_metrics['mae'])} "
        f"rmse={format_float(median_metrics['rmse'])}"
    )
    print()


def print_roi(rows, best_n_per_event):
    mean_rows = select_best_n_per_event(
        rows, "mean_minus_line", "mean_pick", best_n_per_event
    )
    median_rows = select_best_n_per_event(
        rows, "median_minus_line", "median_pick", best_n_per_event
    )
    mean_metrics = roi_metrics(mean_rows, "mean_pick")
    median_metrics = roi_metrics(median_rows, "median_pick")

    print("Betting ROI")
    print(
        f"mean_pick: bets={mean_metrics['bets']} "
        f"wins={mean_metrics['wins']} "
        f"win_rate={format_pct(mean_metrics['win_rate'])} "
        f"avg_price={format_float(mean_metrics['avg_price'])} "
        f"total_profit={format_float(mean_metrics['total_profit'])} "
        f"roi={format_pct(mean_metrics['roi'])}"
    )
    print(
        f"median_pick: bets={median_metrics['bets']} "
        f"wins={median_metrics['wins']} "
        f"win_rate={format_pct(median_metrics['win_rate'])} "
        f"avg_price={format_float(median_metrics['avg_price'])} "
        f"total_profit={format_float(median_metrics['total_profit'])} "
        f"roi={format_pct(median_metrics['roi'])}"
    )
    print()


def print_edge_buckets(rows, best_n_per_event):
    print("Accuracy By Edge Size")
    for label, margin_key, pick_key in [
        ("mean", "mean_minus_line", "mean_pick"),
        ("median", "median_minus_line", "median_pick"),
    ]:
        filtered_rows = select_best_n_per_event(
            rows, margin_key, pick_key, best_n_per_event
        )
        print(f"{label}:")
        for bucket in accuracy_by_edge(
            filtered_rows, margin_key, pick_key, DEFAULT_CONFIG["edge_buckets"]
        ):
            print(
                f"  {bucket['bucket']}: count={bucket['count']} "
                f"accuracy={format_pct(bucket['accuracy'])}"
            )
    print()

    print("ROI By Edge Size")
    for label, margin_key, pick_key in [
        ("mean", "mean_minus_line", "mean_pick"),
        ("median", "median_minus_line", "median_pick"),
    ]:
        filtered_rows = select_best_n_per_event(
            rows, margin_key, pick_key, best_n_per_event
        )
        print(f"{label}:")
        for bucket in roi_by_edge(
            filtered_rows, margin_key, pick_key, DEFAULT_CONFIG["edge_buckets"]
        ):
            print(
                f"  {bucket['bucket']}: count={bucket['count']} "
                f"profit={format_float(bucket['profit'])} "
                f"roi={format_pct(bucket['roi'])}"
            )
    print()


def print_slices(rows):
    print("Rows By Season")
    for value, count in counts_by_field(
        [{"season": infer_season(row["game_date"])} for row in rows], "season"
    ):
        print(f"{value}: {count}")
    print()

    print("Rows By Window")
    for value, count in counts_by_field(rows, "window"):
        print(f"{value}: {count}")
    print()

    print("Rows By Bookmaker")
    for value, count in counts_by_field(rows, "bookmaker_title"):
        print(f"{value}: {count}")
    print()


def main():
    args = parse_args()
    rows = filter_rows_by_seasons(load_rows(args.input), args.seasons)
    print_summary(rows)
    if args.seasons:
        print(f"Season Filter: {' '.join(args.seasons)}")
        print()
    if args.best_n_per_event > 0:
        print(f"Selection Mode: top {args.best_n_per_event} bets per event")
        print()
    print_accuracy(rows, args.best_n_per_event)
    print_regression(rows)
    print_roi(rows, args.best_n_per_event)
    print_edge_buckets(rows, args.best_n_per_event)
    print_slices(rows)


if __name__ == "__main__":
    main()
