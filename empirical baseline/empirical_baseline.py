#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

BOOKMAKER_PRIORITY = [
    "draftkings",
    "fanduel",
    "williamhill_us",  # Caesars
    "betmgm",
    "betrivers",
    "fanatics",
    "pointsbetus",
    "unibet_us",
    "bovada",
    "betonlineag",
    "barstool",
    "mybookieag",
]

SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
MANUAL_NAME_ALIASES = {
    "carlton carrington": ["bub carrington"],
}
DEFAULT_CONFIG = {
    "mapping": "data/game_id_event_id_bijective.csv",
    "players": "data/box_scores/players.csv",
    "odds_dir": "data/historical_event_odds_v4",
    "output": "empirical baseline/empirical_baseline.csv",
    "window": 80,
    "min_history": 40,
    "limit": 0,
}


@dataclass(frozen=True)
class PlayerGame:
    game_id: str
    game_date: date
    person_id: str
    player_name: str
    points: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an empirical NBA player-points baseline from odds and box scores."
    )
    parser.add_argument(
        "--mapping",
        default=DEFAULT_CONFIG["mapping"],
        help="CSV with game_id,event_id pairs.",
    )
    parser.add_argument(
        "--players",
        default=DEFAULT_CONFIG["players"],
        help="NBA player box score CSV.",
    )
    parser.add_argument(
        "--odds-dir",
        default=DEFAULT_CONFIG["odds_dir"],
        help="Directory containing event_id.json odds files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CONFIG["output"],
        help="Output CSV path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_CONFIG["window"],
        help="Lookback window size for the empirical above/equal/below baseline.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_CONFIG["min_history"],
        help="Minimum prior games required to emit a row.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CONFIG["limit"],
        help="Optional cap on mapping rows processed.",
    )
    return parser.parse_args()


def canonical_game_id(game_id):
    return str(int(game_id))


def normalize_text(text):
    ascii_text = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(
        char for char in ascii_text if not unicodedata.combining(char)
    )
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.split())


def name_tokens(name):
    tokens = normalize_text(name).split()
    if tokens and tokens[-1] in SUFFIX_TOKENS:
        core_tokens = tokens[:-1]
    else:
        core_tokens = tokens
    return tokens, core_tokens


def alias_keys(name):
    tokens, core_tokens = name_tokens(name)
    aliases = []

    if tokens:
        aliases.append(" ".join(tokens))
    if core_tokens and core_tokens != tokens:
        aliases.append(" ".join(core_tokens))
    if len(core_tokens) >= 2:
        family_name = " ".join(core_tokens[1:])
        aliases.append(f"{core_tokens[0][0]} {family_name}")

    initials = []
    index = 0
    while index < len(core_tokens) - 1 and len(core_tokens[index]) == 1:
        initials.append(core_tokens[index])
        index += 1
    if len(initials) >= 2:
        aliases.append(f"{''.join(initials)} {' '.join(core_tokens[index:])}")

    seen = set()
    ordered = []
    for alias in aliases:
        if alias and alias not in seen:
            ordered.append(alias)
            seen.add(alias)

    for manual_alias in MANUAL_NAME_ALIASES.get(normalize_text(name), []):
        if manual_alias not in seen:
            ordered.append(manual_alias)
            seen.add(manual_alias)
    return ordered


def safe_int(value):
    return int(float(value))


def safe_float(value, default=0.0):
    if value in (None, ""):
        return default
    return float(value)


def bookmaker_title(bookmaker):
    if bookmaker.get("key") == "williamhill_us":
        return "Caesars"
    return bookmaker.get("title", bookmaker.get("key", "unknown"))


def load_players(players_path):
    roster_index = defaultdict(lambda: defaultdict(list))
    roster_rows = defaultdict(list)
    histories = defaultdict(list)
    history_dates = {}
    game_dates = {}

    with open(players_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = canonical_game_id(row["gameId"])
            game_date = date.fromisoformat(row["gameDate"])
            minutes_float = safe_float(row.get("minutesFloat"))
            player_name = f'{row["firstName"]} {row["familyName"]}'.strip()

            game_dates[game_id] = game_date

            if minutes_float < 1.0:
                continue

            record = PlayerGame(
                game_id=game_id,
                game_date=game_date,
                person_id=row["personId"],
                player_name=player_name,
                points=safe_int(row["points"]),
            )

            roster_rows[game_id].append(record)
            histories[record.person_id].append(record)

            aliases = set(alias_keys(player_name))
            if row.get("playerSlug"):
                aliases.update(alias_keys(row["playerSlug"].replace("-", " ")))

            for alias in aliases:
                roster_index[game_id][alias].append(record)

    for person_id, games in histories.items():
        games.sort(key=lambda game: (game.game_date, game.game_id))
        history_dates[person_id] = [game.game_date for game in games]

    return game_dates, roster_rows, roster_index, histories, history_dates


def unique_record(records):
    if not records:
        return None
    person_ids = {record.person_id for record in records}
    if len(person_ids) == 1:
        return records[0]
    return None


def family_name_from_core(core_tokens):
    if len(core_tokens) < 2:
        return ""
    return " ".join(core_tokens[1:])


def match_current_player(game_id, odds_player_name, roster_rows, roster_index):
    canonical_id = canonical_game_id(game_id)
    for alias in alias_keys(odds_player_name):
        match = unique_record(roster_index[canonical_id].get(alias, []))
        if match is not None:
            return match

    _, core_tokens = name_tokens(odds_player_name)
    family_name = family_name_from_core(core_tokens)
    if not family_name:
        return None

    first_initial = core_tokens[0][0]
    candidates = []
    for record in roster_rows[canonical_id]:
        _, record_core = name_tokens(record.player_name)
        if not record_core:
            continue
        if family_name_from_core(record_core) != family_name:
            continue
        if record_core[0][0] != first_initial:
            continue
        candidates.append(record)

    return unique_record(candidates)


def load_event(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["data"]


def select_bookmaker(event):
    for book_key in BOOKMAKER_PRIORITY:
        for bookmaker in event.get("bookmakers", []):
            if bookmaker.get("key") != book_key:
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") == "player_points" and market.get("outcomes"):
                    return bookmaker, market
    return None, None


def pick_primary_point(points_by_line):
    ranked_points = []

    for point, sides in points_by_line.items():
        over_price = sides.get("over_price")
        under_price = sides.get("under_price")

        if over_price is not None and under_price is not None:
            ranked_points.append(
                (
                    0,
                    abs(over_price - under_price),
                    abs(((over_price + under_price) / 2.0) - 1.91),
                    point,
                )
            )
        else:
            ranked_points.append((1, float("inf"), float("inf"), point))

    if not ranked_points:
        return None

    ranked_points.sort()
    return ranked_points[0][3]


def primary_lines_for_market(market):
    by_player = defaultdict(lambda: defaultdict(dict))

    for outcome in market.get("outcomes", []):
        player_name = outcome.get("description")
        point = outcome.get("point")
        side = (outcome.get("name") or "").strip().lower()

        if player_name is None or point is None or side not in {"over", "under"}:
            continue

        entry = by_player[player_name][float(point)]
        entry["player_name"] = player_name
        entry["point"] = float(point)
        entry[f"{side}_price"] = outcome.get("price")

    lines = []
    for player_name, points_by_line in by_player.items():
        chosen_point = pick_primary_point(points_by_line)
        if chosen_point is None:
            continue
        chosen_line = points_by_line[chosen_point]
        lines.append(
            {
                "player_name": player_name,
                "line": chosen_line["point"],
                "over_price": chosen_line.get("over_price"),
                "under_price": chosen_line.get("under_price"),
            }
        )

    lines.sort(key=lambda item: normalize_text(item["player_name"]))
    return lines


def lookback_points(histories, history_dates, person_id, current_game_date, window):
    games = histories[person_id]
    dates = history_dates[person_id]
    cutoff = bisect.bisect_left(dates, current_game_date)
    recent_games = games[max(0, cutoff - window) : cutoff]
    return [game.points for game in recent_games]


def empirical_breakdown(prior_points, line):
    above = sum(1 for points in prior_points if points > line)
    equal = sum(1 for points in prior_points if points == line)
    below = sum(1 for points in prior_points if points < line)
    total = len(prior_points)
    return {
        "above_count": above,
        "equal_count": equal,
        "below_count": below,
        "above_pct": above / total,
        "equal_pct": equal / total,
        "below_pct": below / total,
        "edge": (above - below) / total,
    }


def empirical_pick(breakdown):
    if breakdown["above_count"] > breakdown["below_count"]:
        return "over"
    if breakdown["below_count"] > breakdown["above_count"]:
        return "under"
    return "push"


def actual_result(actual_points, line):
    if actual_points > line:
        return "over"
    if actual_points < line:
        return "under"
    return "push"


def make_output_row(
    game_id,
    event_id,
    game_date,
    bookmaker,
    player_record,
    odds_player_name,
    line_info,
    prior_points,
    window,
):
    line = line_info["line"]
    breakdown = empirical_breakdown(prior_points, line)
    pick = empirical_pick(breakdown)
    result = actual_result(player_record.points, line)

    return {
        "game_id": game_id,
        "event_id": event_id,
        "game_date": game_date.isoformat(),
        "bookmaker_key": bookmaker["key"],
        "bookmaker_title": bookmaker_title(bookmaker),
        "player_name_odds": odds_player_name,
        "player_name_box_score": player_record.player_name,
        "person_id": player_record.person_id,
        "line_points": f"{line:.1f}",
        "over_price": line_info["over_price"],
        "under_price": line_info["under_price"],
        "window": window,
        "history_games_used": len(prior_points),
        "above_count": breakdown["above_count"],
        "equal_count": breakdown["equal_count"],
        "below_count": breakdown["below_count"],
        "above_pct": f"{breakdown['above_pct']:.6f}",
        "equal_pct": f"{breakdown['equal_pct']:.6f}",
        "below_pct": f"{breakdown['below_pct']:.6f}",
        "empirical_edge": f"{breakdown['edge']:.6f}",
        "actual_points": player_record.points,
        "actual_result": result,
        "empirical_pick": pick,
    }


def write_rows(path, rows):
    fieldnames = [
        "game_id",
        "event_id",
        "game_date",
        "bookmaker_key",
        "bookmaker_title",
        "player_name_odds",
        "player_name_box_score",
        "person_id",
        "line_points",
        "over_price",
        "under_price",
        "window",
        "history_games_used",
        "above_count",
        "equal_count",
        "below_count",
        "above_pct",
        "equal_pct",
        "below_pct",
        "empirical_edge",
        "actual_points",
        "actual_result",
        "empirical_pick",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_baseline(args):
    game_dates, roster_rows, roster_index, histories, history_dates = load_players(
        args.players
    )
    rows = []
    stats = Counter()

    with open(args.mapping, newline="", encoding="utf-8") as handle:
        mapping_reader = csv.DictReader(handle)

        for index, mapping_row in enumerate(mapping_reader, start=1):
            if args.limit and index > args.limit:
                break

            stats["mapping_rows"] += 1
            game_id = canonical_game_id(mapping_row["game_id"])
            event_id = mapping_row["event_id"]
            game_date = game_dates.get(game_id)

            if game_date is None:
                stats["missing_game_date"] += 1
                continue

            event_path = Path(args.odds_dir) / f"{event_id}.json"
            if not event_path.exists():
                stats["missing_event_file"] += 1
                continue

            event = load_event(event_path)
            bookmaker, market = select_bookmaker(event)
            if bookmaker is None:
                stats["missing_player_points_bookmaker"] += 1
                continue

            stats["events_with_bookmaker"] += 1
            line_infos = primary_lines_for_market(market)
            stats["candidate_player_lines"] += len(line_infos)

            for line_info in line_infos:
                player_record = match_current_player(
                    game_id=game_id,
                    odds_player_name=line_info["player_name"],
                    roster_rows=roster_rows,
                    roster_index=roster_index,
                )
                if player_record is None:
                    stats["unmatched_current_player"] += 1
                    continue

                prior_points = lookback_points(
                    histories=histories,
                    history_dates=history_dates,
                    person_id=player_record.person_id,
                    current_game_date=game_date,
                    window=args.window,
                )
                if len(prior_points) < args.min_history:
                    stats["insufficient_history"] += 1
                    continue

                rows.append(
                    make_output_row(
                        game_id=game_id,
                        event_id=event_id,
                        game_date=game_date,
                        bookmaker=bookmaker,
                        player_record=player_record,
                        odds_player_name=line_info["player_name"],
                        line_info=line_info,
                        prior_points=prior_points,
                        window=args.window,
                    )
                )
                stats["rows_written"] += 1

    return rows, stats


def print_summary(stats, output_path):
    print(f"Wrote {stats['rows_written']} rows to {output_path}")
    print(
        "Summary: "
        f"mapping_rows={stats['mapping_rows']}, "
        f"missing_game_date={stats['missing_game_date']}, "
        f"missing_event_file={stats['missing_event_file']}, "
        f"missing_player_points_bookmaker={stats['missing_player_points_bookmaker']}, "
        f"events_with_bookmaker={stats['events_with_bookmaker']}, "
        f"candidate_player_lines={stats['candidate_player_lines']}, "
        f"unmatched_current_player={stats['unmatched_current_player']}, "
        f"insufficient_history={stats['insufficient_history']}"
    )
    if stats["mapping_rows"]:
        coverage = stats["rows_written"] / stats["mapping_rows"]
        print(f"Rows per mapped game: {coverage:.2f}")


def main():
    args = parse_args()
    rows, stats = build_baseline(args)
    output_path = Path(args.output)
    write_rows(output_path, rows)
    print_summary(stats, output_path)


if __name__ == "__main__":
    main()
