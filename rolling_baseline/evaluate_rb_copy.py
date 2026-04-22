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
        help="If > 0, only evaluate the top N bets per event.",
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
    parser.add_argument(
        "--positive-ev-only",
        action="store_true",
        help=(
            "Only place bets when the estimated expected ROI is positive. "
            "Estimated win probability comes from historical accuracy by edge bucket."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["edge", "estimated-roi"],
        default="edge",
        help=(
            "How to rank bets within each event when --best-n-per-event is used. "
            "'edge' ranks by absolute model edge; 'estimated-roi' ranks by "
            "price-aware estimated ROI."
        ),
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
    split_index = int(len(ordered) * (train_ratio / (train_ratio + test_ratio)))
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


def bucket_label(value, bucket_edges):
    for start, end in zip(bucket_edges, bucket_edges[1:]):
        if start <= value < end:
            return f"[{start:.0f}, {end:.0f})"
    return f"[{bucket_edges[-1]:.0f}, +inf)"


def selected_price(row, pick_key):
    pick = row[pick_key]
    if pick == "over":
        return maybe_float(row["over_price"])
    if pick == "under":
        return maybe_float(row["under_price"])
    return float("nan")


def priced_bets(rows, pick_key):
    bets = []
    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        price = selected_price(row, pick_key)
        if math.isnan(price):
            continue
        bets.append((row, price))
    return bets


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


def build_accuracy_calibration(rows, score_key, pick_key, bucket_edges):
    labels = [f"[{start:.0f}, {end:.0f})" for start, end in zip(bucket_edges, bucket_edges[1:])]
    labels.append(f"[{bucket_edges[-1]:.0f}, +inf)")

    bucket_counts = Counter()
    bucket_wins = Counter()
    total_decided = 0
    total_wins = 0

    for row in rows:
        if row["actual_result"] == "push" or row[pick_key] == "push":
            continue
        bucket = bucket_label(abs(to_float(row, score_key)), bucket_edges)
        bucket_counts[bucket] += 1
        total_decided += 1
        if row[pick_key] == row["actual_result"]:
            bucket_wins[bucket] += 1
            total_wins += 1

    global_accuracy = total_wins / total_decided if total_decided else float("nan")
    bucket_accuracy = {}
    for label in labels:
        if bucket_counts[label]:
            bucket_accuracy[label] = bucket_wins[label] / bucket_counts[label]

    return {
        "global_accuracy": global_accuracy,
        "bucket_accuracy": bucket_accuracy,
    }


def estimated_pick_accuracy(row, score_key, calibration, bucket_edges):
    bucket = bucket_label(abs(to_float(row, score_key)), bucket_edges)
    return calibration["bucket_accuracy"].get(bucket, calibration["global_accuracy"])


def annotate_price_aware_rows(rows, spec, calibration, bucket_edges):
    annotated = []
    raw_pick_key = spec["pick_key"]
    score_key = spec["score_key"]

    for row in rows:
        annotated_row = dict(row)
        raw_pick = row[raw_pick_key]
        if raw_pick == "push":
            annotated_row["_eval_pick"] = "push"
            annotated_row["_eval_estimated_roi"] = float("nan")
            annotated_row["_eval_roi_over"] = float("nan")
            annotated_row["_eval_roi_under"] = float("nan")
            annotated.append(annotated_row)
            continue

        p_pick = estimated_pick_accuracy(row, score_key, calibration, bucket_edges)
        if math.isnan(p_pick):
            annotated_row["_eval_pick"] = "push"
            annotated_row["_eval_estimated_roi"] = float("nan")
            annotated_row["_eval_roi_over"] = float("nan")
            annotated_row["_eval_roi_under"] = float("nan")
            annotated.append(annotated_row)
            continue

        if raw_pick == "over":
            p_over = p_pick
            p_under = 1.0 - p_pick
        else:
            p_under = p_pick
            p_over = 1.0 - p_pick

        over_price = maybe_float(row["over_price"])
        under_price = maybe_float(row["under_price"])
        ev_over = p_over * over_price - 1.0 if not math.isnan(over_price) else float("nan")
        ev_under = p_under * under_price - 1.0 if not math.isnan(under_price) else float("nan")

        candidates = []
        if not math.isnan(ev_over):
            candidates.append(("over", ev_over))
        if not math.isnan(ev_under):
            candidates.append(("under", ev_under))

        if not candidates:
            annotated_row["_eval_pick"] = "push"
            annotated_row["_eval_estimated_roi"] = float("nan")
        else:
            best_side, best_ev = max(candidates, key=lambda item: item[1])
            annotated_row["_eval_pick"] = best_side
            annotated_row["_eval_estimated_roi"] = best_ev

        annotated_row["_eval_roi_over"] = ev_over
        annotated_row["_eval_roi_under"] = ev_under
        annotated.append(annotated_row)

    return annotated


def select_rows(
    rows,
    pick_key,
    score_key,
    best_n_per_event,
    selection_metric,
    positive_ev_only=False,
    estimated_roi_key=None,
):
    if best_n_per_event <= 0:
        if not positive_ev_only:
            return list(rows)
        return [
            row
            for row in rows
            if row[pick_key] != "push"
            and not math.isnan(row[estimated_roi_key])
            and row[estimated_roi_key] > 0.0
        ]

    grouped = defaultdict(list)
    for row in rows:
        if row[pick_key] == "push":
            continue
        if positive_ev_only:
            est_roi = row[estimated_roi_key]
            if math.isnan(est_roi) or est_roi <= 0.0:
                continue
        grouped[row["event_id"]].append(row)

    if selection_metric == "estimated-roi":

        def rank_value(row):
            est_roi = row[estimated_roi_key]
            if math.isnan(est_roi):
                return float("-inf")
            return est_roi

    else:

        def rank_value(row):
            return abs(to_float(row, score_key))

    selected = []
    for event_rows in grouped.values():
        ranked = sorted(event_rows, key=rank_value, reverse=True)
        selected.extend(ranked[:best_n_per_event])

    return selected


def top_n_summary(
    rows,
    pick_key,
    score_key,
    best_n_per_event,
    selection_metric,
    positive_ev_only=False,
    estimated_roi_key=None,
):
    selected = select_rows(
        rows,
        pick_key,
        score_key,
        best_n_per_event,
        selection_metric,
        positive_ev_only=positive_ev_only,
        estimated_roi_key=estimated_roi_key,
    )
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


def evaluate_models(rows, best_n_per_event, bucket_edges, calibrations, args):
    results = {}
    for spec in MODEL_SPECS:
        use_price_aware = args.positive_ev_only or args.selection_metric == "estimated-roi"
        if use_price_aware:
            model_rows = annotate_price_aware_rows(
                rows,
                spec,
                calibrations[spec["name"]],
                bucket_edges,
            )
            eval_pick_key = "_eval_pick"
            eval_roi_key = "_eval_estimated_roi"
        else:
            model_rows = rows
            eval_pick_key = spec["pick_key"]
            eval_roi_key = None

        selected = select_rows(
            model_rows,
            eval_pick_key,
            spec["score_key"],
            best_n_per_event,
            args.selection_metric,
            positive_ev_only=args.positive_ev_only,
            estimated_roi_key=eval_roi_key,
        )

        results[spec["name"]] = {
            "rows": model_rows,
            "pick_key": eval_pick_key,
            "selected_rows": selected,
            "accuracy": accuracy_metrics(selected, eval_pick_key),
            "regression": regression_metrics(rows, spec["projection_key"]),
            "roi": roi_metrics(selected, eval_pick_key),
            "accuracy_by_edge": accuracy_by_edge(
                selected,
                spec["score_key"],
                eval_pick_key,
                bucket_edges,
            ),
            "roi_by_edge": roi_by_edge(
                selected,
                spec["score_key"],
                eval_pick_key,
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


def selection_mode_label(best_n_per_event, selection_metric, positive_ev_only):
    if best_n_per_event > 0:
        if selection_metric == "estimated-roi":
            label = f"top {best_n_per_event} per event by estimated ROI"
        else:
            label = f"top {best_n_per_event} per event by absolute edge"
    else:
        label = "all priced non-push bets"
    if positive_ev_only:
        label += ", positive estimated ROI only"
    return label


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


def print_split_top_n_summary(split_name, rows, calibrations, args):
    print(split_name)
    for spec in MODEL_SPECS:
        use_price_aware = args.positive_ev_only or args.selection_metric == "estimated-roi"
        if use_price_aware:
            summary_rows = annotate_price_aware_rows(
                rows,
                spec,
                calibrations[spec["name"]],
                DEFAULT_CONFIG["edge_buckets"],
            )
            pick_key = "_eval_pick"
            estimated_roi_key = "_eval_estimated_roi"
        else:
            summary_rows = rows
            pick_key = spec["pick_key"]
            estimated_roi_key = None

        summary = top_n_summary(
            summary_rows,
            pick_key,
            spec["score_key"],
            args.best_n_per_event,
            args.selection_metric,
            positive_ev_only=args.positive_ev_only,
            estimated_roi_key=estimated_roi_key,
        )
        print(
            f"  {spec['summary_label']} "
            f"({selection_mode_label(args.best_n_per_event, args.selection_metric, args.positive_ev_only)})"
        )
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

    overall_calibrations = {
        spec["name"]: build_accuracy_calibration(
            rows,
            spec["score_key"],
            spec["pick_key"],
            DEFAULT_CONFIG["edge_buckets"],
        )
        for spec in MODEL_SPECS
    }
    results = evaluate_models(
        rows,
        args.best_n_per_event,
        DEFAULT_CONFIG["edge_buckets"],
        overall_calibrations,
        args,
    )

    print_summary(rows)
    if args.seasons:
        print(f"Season Filter: {' '.join(args.seasons)}")
        print()
    if args.best_n_per_event > 0:
        print(f"Selection Mode: top {args.best_n_per_event} bets per event")
        print()
    print(f"Selection Metric: {args.selection_metric}")
    print()
    if args.positive_ev_only:
        print(
            "Bet Filter: only bets with positive estimated ROI, where estimated win "
            "probability comes from historical accuracy by edge bucket"
        )
        print()
    if args.selection_metric == "estimated-roi" or args.positive_ev_only:
        print(
            "Decision Rule: compute ev_over = p_over * over_price - 1 and "
            "ev_under = p_under * under_price - 1, then choose the better side"
        )
        print()

    if args.split_ratios is not None:
        train_rows, test_rows = chronological_split(rows, args.split_ratios)
        split_calibrations = {
            spec["name"]: build_accuracy_calibration(
                train_rows,
                spec["score_key"],
                spec["pick_key"],
                DEFAULT_CONFIG["edge_buckets"],
            )
            for spec in MODEL_SPECS
        }
        print_split_overview(train_rows, test_rows, args.split_ratios)
        print_split_top_n_summary("Train", train_rows, split_calibrations, args)
        print_split_top_n_summary("Test", test_rows, split_calibrations, args)

    print_accuracy(results)
    print_regression(results)
    print_roi(results)
    print_edge_buckets(results)
    print_slices(rows)


if __name__ == "__main__":
    main()
