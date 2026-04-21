#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = SCRIPT_DIR / "data/box_scores/players.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "data/box_scores/players_filtered.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter player box scores down to rows with positive minutes played."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the source players CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to the filtered output CSV.",
    )
    return parser.parse_args()


def parse_minutes_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_minutes_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return int(minutes) + int(seconds) / 60.0
        except ValueError:
            return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def played_positive_minutes(row):
    minutes_float = parse_minutes_float(row.get("minutesFloat"))
    if minutes_float is not None:
        return minutes_float > 0

    minutes_text = parse_minutes_text(row.get("minutes"))
    if minutes_text is not None:
        return minutes_text > 0

    return False


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0

    with input_csv.open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in {input_csv}")

        with output_csv.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                if played_positive_minutes(row):
                    writer.writerow(row)
                    kept_rows += 1

    print(f"Wrote {kept_rows} of {total_rows} rows to {output_csv}")


if __name__ == "__main__":
    main()
