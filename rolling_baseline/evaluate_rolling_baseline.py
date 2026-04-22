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
    "split_ratios": None,
}

MODEL_SPECS = [
    {
        "name": "mean",
        "pick_key": "mean_pick",
        "score_key": "mean_minus_line",
        "projection_key": "rolling_mean_points",
        "summary_label": "Rolling mean baseline",
    },
    {
        "name": "median",
        "pick_key": "median_pick",
        "score_key": "median_minus_line",
        "projection_key": "rolling_median_points",
        "summary_label": "Rolling median baseline",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate rolling baseline player-points predictions."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_CONFIG["input"],
        help="Baseline CSV produced by rolling_baseline.py.",
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
    parser.add_argument(
        "--split-ratios",
        nargs=2,
        type=float,
        default=DEFAULT_CONFIG["split_ratios"],
        metavar=("TRAIN_RATIO", "TEST_RATIO"),
        help="Optional chronological train/test split ratios, e.g. --split-ratios 0.7 0.3.",
    )
    return parser.parse_args()


def infer_season(game_date):
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["_game_date"] = date.fromisoformat(row["game_date"])
        row["_season"] = infer_season(row["_game_date"])

    return rows


def filter_rows_by_seasons(rows, seasons):
    if not seasons:
        return rows
    season_set = set(seasons)
    return [row for row in rows if row["_season"] in season_set]


def chronological_split(rows, split_ratios):
    train_ratio, test_ratio = split_ratios
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    ordered = sorted(
        rows,
        key=lambda row: (
            row["_game_date"],
            row["game_id"],
            row["event_id"],
            row["person_id"],
        ),
    )
    total_ratio = train_ratio + test_ratio
    train_fraction = train_ratio / total_ratio
    split_index = int(len(ordered) * train_fraction)

    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError(
            "Chronological split produced an empty train or test set; adjust --split-ratios."
        )

    return ordered[:split_index], ordered[split_index:]


def to_float(row, key):
    return float(row[key])


def to_int(row, key):
    return int(row[key])


def maybe_float(value):
    if value in ("", None):
        return float("nan")
    return float(value)


def select_rows(rows, score_key, pick_key, best_n_per_event):
    if best_n_per_event <= 0:
        return list(rows)

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


def selected_price(row, pick_key):
    pick = row[pick_key]
    if pick == "over":
        return maybe_float(row["over_price"])
    if pick == "under":
        return maybe_float(row["under_price"])
    return float("nan")


def priced_bets(rows, pick_key):
    priced = []
    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        price = selected_price(row, pick_key)
        if math.isnan(price):
            continue
        priced.append((row, price))
    return priced


def accuracy_metrics(rows, pick_key):
    eligible = [row for row in rows if row["actual_result"] != "push"]
    decided = [row for row in eligible if row[pick_key] != "push"]
    correct = sum(1 for row in decided if row[pick_key] == row["actual_result"])
    return {
        "eligible": len(eligible),
        "decided": len(decided),
        "correct": correct,
        "accuracy": (correct / len(decided)) if decided else float("nan"),
    }


def regression_metrics(rows, projection_key):
    errors = [to_float(row, projection_key) - to_int(row, "actual_points") for row in rows]
    mae = sum(abs(error) for error in errors) / len(errors) if errors else float("nan")
    rmse = (
        math.sqrt(sum(error * error for error in errors) / len(errors))
        if errors
        else float("nan")
    )
    return {"count": len(errors), "mae": mae, "rmse": rmse}


def roi_metrics(rows, pick_key):
    bets = priced_bets(rows, pick_key)
    if not bets:
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

    for row, price in bets:
        total_price += price
        if row[pick_key] == row["actual_result"]:
            wins += 1
            total_profit += price - 1.0
        else:
            total_profit -= 1.0

    return {
        "bets": len(bets),
        "wins": wins,
        "win_rate": wins / len(bets),
        "avg_price": total_price / len(bets),
        "total_profit": total_profit,
        "roi": total_profit / len(bets),
    }


def top_n_summary(rows, score_key, pick_key, best_n_per_event):
    selected = select_rows(rows, score_key, pick_key, best_n_per_event)
    bets = priced_bets(selected, pick_key)
    if not bets:
        return {
            "selected_rows": 0,
            "events": 0,
            "correct": 0,
            "accuracy": float("nan"),
            "profit": float("nan"),
            "roi": float("nan"),
        }

    correct = sum(1 for row, _ in bets if row[pick_key] == row["actual_result"])
    profit = sum(
        (price - 1.0) if row[pick_key] == row["actual_result"] else -1.0
        for row, price in bets
    )

    return {
        "selected_rows": len(bets),
        "events": len({row["event_id"] for row, _ in bets}),
        "correct": correct,
        "accuracy": correct / len(bets),
        "profit": profit,
        "roi": profit / len(bets),
    }


def bucket_label(value, bucket_edges):
    for start, end in zip(bucket_edges, bucket_edges[1:]):
        if start <= value < end:
            return f"[{start:.0f}, {end:.0f})"
    return f"[{bucket_edges[-1]:.0f}, +inf)"


def accuracy_by_edge(rows, score_key, pick_key, bucket_edges):
    grouped = defaultdict(list)
    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        edge = abs(to_float(row, score_key))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    labels = [f"[{start:.0f}, {end:.0f})" for start, end in zip(bucket_edges, bucket_edges[1:])]
    labels.append(f"[{bucket_edges[-1]:.0f}, +inf)")

    results = []
    for label in labels:
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


def roi_by_edge(rows, score_key, pick_key, bucket_edges):
    grouped = defaultdict(list)
    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        edge = abs(to_float(row, score_key))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    labels = [f"[{start:.0f}, {end:.0f})" for start, end in zip(bucket_edges, bucket_edges[1:])]
    labels.append(f"[{bucket_edges[-1]:.0f}, +inf)")

    results = []
    for label in labels:
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
    return sorted(Counter(row[field] for row in rows).items())


def evaluate_models(rows, best_n_per_event, bucket_edges):
    results = {}
    for spec in MODEL_SPECS:
        selected = select_rows(rows, spec["score_key"], spec["pick_key"], best_n_per_event)
        results[spec["name"]] = {
            "selected_rows": selected,
            "accuracy": accuracy_metrics(selected, spec["pick_key"]),
            "regression": regression_metrics(rows, spec["projection_key"]),
            "roi": roi_metrics(selected, spec["pick_key"]),
            "accuracy_by_edge": accuracy_by_edge(
                selected,
                spec["score_key"],
                spec["pick_key"],
                bucket_edges,
            ),
            "roi_by_edge": roi_by_edge(
                selected,
                spec["score_key"],
                spec["pick_key"],
                bucket_edges,
            ),
        }
    return results


def format_pct(value):
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def format_float(value):
    if math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def selection_mode_label(best_n_per_event):
    if best_n_per_event > 0:
        return f"top {best_n_per_event} per event"
    return "all priced non-push bets"


def print_summary(rows):
    print(f"Rows: {len(rows)}")
    print(f"Actual Pushes: {sum(1 for row in rows if row['actual_result'] == 'push')}")
    for spec in MODEL_SPECS:
        push_count = sum(1 for row in rows if row[spec["pick_key"]] == "push")
        print(f"{spec['summary_label']} Pushes: {push_count}")
    print()


def print_accuracy(results):
    print("Directional Accuracy")
    for spec in MODEL_SPECS:
        metrics = results[spec["name"]]["accuracy"]
        print(
            f"{spec['pick_key']}: correct={metrics['correct']} "
            f"decided={metrics['decided']} "
            f"eligible={metrics['eligible']} "
            f"accuracy={format_pct(metrics['accuracy'])}"
        )
    print()


def print_regression(results):
    print("Projection Error")
    for spec in MODEL_SPECS:
        metrics = results[spec["name"]]["regression"]
        print(
            f"{spec['projection_key']}: count={metrics['count']} "
            f"mae={format_float(metrics['mae'])} "
            f"rmse={format_float(metrics['rmse'])}"
        )
    print()


def print_roi(results):
    print("Betting ROI")
    for spec in MODEL_SPECS:
        metrics = results[spec["name"]]["roi"]
        print(
            f"{spec['pick_key']}: bets={metrics['bets']} "
            f"wins={metrics['wins']} "
            f"win_rate={format_pct(metrics['win_rate'])} "
            f"avg_price={format_float(metrics['avg_price'])} "
            f"total_profit={format_float(metrics['total_profit'])} "
            f"roi={format_pct(metrics['roi'])}"
        )
    print()


def print_edge_buckets(results):
    print("Accuracy By Edge Size")
    for spec in MODEL_SPECS:
        print(f"{spec['name']}:")
        for bucket in results[spec["name"]]["accuracy_by_edge"]:
            print(
                f"  {bucket['bucket']}: count={bucket['count']} "
                f"accuracy={format_pct(bucket['accuracy'])}"
            )
    print()

    print("ROI By Edge Size")
    for spec in MODEL_SPECS:
        print(f"{spec['name']}:")
        for bucket in results[spec["name"]]["roi_by_edge"]:
            print(
                f"  {bucket['bucket']}: count={bucket['count']} "
                f"profit={format_float(bucket['profit'])} "
                f"roi={format_pct(bucket['roi'])}"
            )
    print()


def print_slices(rows):
    print("Rows By Season")
    for value, count in counts_by_field(rows, "_season"):
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


def print_split_overview(train_rows, test_rows, split_ratios):
    print("Chronological Split")
    print(f"  split_ratios={split_ratios[0]:.3f}/{split_ratios[1]:.3f}")
    print(f"  train_rows={len(train_rows)}")
    print(f"  test_rows={len(test_rows)}")
    print(
        f"  train_date_range={train_rows[0]['game_date']} to {train_rows[-1]['game_date']}"
    )
    print(f"  test_date_range={test_rows[0]['game_date']} to {test_rows[-1]['game_date']}")
    print()


def print_split_top_n_summary(split_name, rows, best_n_per_event):
    print(split_name)
    for spec in MODEL_SPECS:
        summary = top_n_summary(
            rows,
            spec["score_key"],
            spec["pick_key"],
            best_n_per_event,
        )
        print(f"  {spec['summary_label']} ({selection_mode_label(best_n_per_event)})")
        print(
            f"  selected_rows={summary['selected_rows']} "
            f"events={summary['events']} "
            f"correct={summary['correct']} "
            f"accuracy={format_pct(summary['accuracy'])} "
            f"profit={format_float(summary['profit'])} "
            f"roi={format_pct(summary['roi'])}"
        )
    print()


def main():
    args = parse_args()
    rows = filter_rows_by_seasons(load_rows(args.input), args.seasons)
    results = evaluate_models(rows, args.best_n_per_event, DEFAULT_CONFIG["edge_buckets"])

    print_summary(rows)
    if args.seasons:
        print(f"Season Filter: {' '.join(args.seasons)}")
        print()
    if args.best_n_per_event > 0:
        print(f"Selection Mode: top {args.best_n_per_event} bets per event")
        print()
    if args.split_ratios is not None:
        train_rows, test_rows = chronological_split(rows, args.split_ratios)
        print_split_overview(train_rows, test_rows, args.split_ratios)
        print_split_top_n_summary("Train", train_rows, args.best_n_per_event)
        print_split_top_n_summary("Test", test_rows, args.best_n_per_event)

    print_accuracy(results)
    print_regression(results)
    print_roi(results)
    print_edge_buckets(results)
    print_slices(rows)


if __name__ == "__main__":
    main()
