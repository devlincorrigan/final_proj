#!/usr/bin/env python3

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import date

DEFAULT_CONFIG = {
    "input": "empirical baseline/empirical_baseline.csv",
    "edge_buckets": [0.0, 0.10, 0.20, 0.30],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate empirical player-points baseline predictions."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_CONFIG["input"],
        help="Empirical baseline CSV produced by empirical_baseline.py.",
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


def maybe_float(value):
    if value in ("", None):
        return float("nan")
    return float(value)


def accuracy(rows):
    eligible = [row for row in rows if row["actual_result"] != "push"]
    decided = [row for row in eligible if row["empirical_pick"] != "push"]
    correct = sum(1 for row in decided if row["empirical_pick"] == row["actual_result"])
    return {
        "eligible": len(eligible),
        "decided": len(decided),
        "correct": correct,
        "accuracy": (correct / len(decided)) if decided else float("nan"),
    }


def select_best_n_per_event(rows, best_n_per_event):
    if best_n_per_event <= 0:
        return rows

    grouped = defaultdict(list)
    for row in rows:
        if row["empirical_pick"] == "push":
            continue
        grouped[row["event_id"]].append(row)

    selected = []
    for event_rows in grouped.values():
        ranked = sorted(
            event_rows,
            key=lambda row: abs(to_float(row, "empirical_edge")),
            reverse=True,
        )
        selected.extend(ranked[:best_n_per_event])

    return selected


def selected_price(row):
    pick = row["empirical_pick"]
    if pick == "over":
        return maybe_float(row["over_price"])
    if pick == "under":
        return maybe_float(row["under_price"])
    return float("nan")


def roi_metrics(rows):
    eligible = [
        row for row in rows if row["actual_result"] != "push" and row["empirical_pick"] != "push"
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
    priced_bets = 0

    for row in eligible:
        price = selected_price(row)
        if math.isnan(price):
            continue
        priced_bets += 1
        total_price += price
        if row["empirical_pick"] == row["actual_result"]:
            wins += 1
            total_profit += price - 1.0
        else:
            total_profit -= 1.0

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
            return f"[{start:.2f}, {end:.2f})"
    return f"[{bucket_edges[-1]:.2f}, +inf)"


def accuracy_by_edge(rows, bucket_edges):
    grouped = defaultdict(list)

    for row in rows:
        if row["actual_result"] == "push" or row["empirical_pick"] == "push":
            continue
        edge = abs(to_float(row, "empirical_edge"))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    results = []
    all_labels = [
        f"[{start:.2f}, {end:.2f})"
        for start, end in zip(bucket_edges, bucket_edges[1:])
    ]
    all_labels.append(f"[{bucket_edges[-1]:.2f}, +inf)")

    for label in all_labels:
        bucket_rows = grouped.get(label, [])
        correct = sum(
            1 for row in bucket_rows if row["empirical_pick"] == row["actual_result"]
        )
        results.append(
            {
                "bucket": label,
                "count": len(bucket_rows),
                "accuracy": (correct / len(bucket_rows)) if bucket_rows else float("nan"),
            }
        )

    return results


def roi_by_edge(rows, bucket_edges):
    grouped = defaultdict(list)

    for row in rows:
        if row["actual_result"] == "push" or row["empirical_pick"] == "push":
            continue
        edge = abs(to_float(row, "empirical_edge"))
        grouped[bucket_label(edge, bucket_edges)].append(row)

    results = []
    all_labels = [
        f"[{start:.2f}, {end:.2f})"
        for start, end in zip(bucket_edges, bucket_edges[1:])
    ]
    all_labels.append(f"[{bucket_edges[-1]:.2f}, +inf)")

    for label in all_labels:
        bucket_rows = grouped.get(label, [])
        roi = roi_metrics(bucket_rows)
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
    filtered_rows = select_best_n_per_event(rows, best_n_per_event)
    stats = accuracy(filtered_rows)

    print("Directional Accuracy")
    print(
        f"empirical_pick: correct={stats['correct']} "
        f"decided={stats['decided']} "
        f"eligible={stats['eligible']} "
        f"accuracy={format_pct(stats['accuracy'])}"
    )
    print()


def print_roi(rows, best_n_per_event):
    filtered_rows = select_best_n_per_event(rows, best_n_per_event)
    metrics = roi_metrics(filtered_rows)

    print("Betting ROI")
    print(
        f"empirical_pick: bets={metrics['bets']} "
        f"wins={metrics['wins']} "
        f"win_rate={format_pct(metrics['win_rate'])} "
        f"avg_price={format_float(metrics['avg_price'])} "
        f"total_profit={format_float(metrics['total_profit'])} "
        f"roi={format_pct(metrics['roi'])}"
    )
    print()


def print_edge_buckets(rows, best_n_per_event):
    filtered_rows = select_best_n_per_event(rows, best_n_per_event)
    print("Accuracy By Edge Size")
    for bucket in accuracy_by_edge(filtered_rows, DEFAULT_CONFIG["edge_buckets"]):
        print(
            f"  {bucket['bucket']}: count={bucket['count']} "
            f"accuracy={format_pct(bucket['accuracy'])}"
        )
    print()

    print("ROI By Edge Size")
    for bucket in roi_by_edge(filtered_rows, DEFAULT_CONFIG["edge_buckets"]):
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
    print_roi(rows, args.best_n_per_event)
    print_edge_buckets(rows, args.best_n_per_event)
    print_slices(rows)


if __name__ == "__main__":
    main()
