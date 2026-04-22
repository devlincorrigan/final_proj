#!/usr/bin/env python3

import csv
import math
import os
import statistics
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MPL_DIR = APP_DIR / ".matplotlib"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))

import matplotlib.pyplot as plt
import streamlit as st

PLAYERS_CSV = APP_DIR / "data" / "box_scores" / "players_filtered.csv"
DEFAULT_LOOKBACK_WINDOW = 80
FIGURE_SIZE = (14, 5.5)
HISTOGRAM_LABEL_SPACING = 3
HISTOGRAM_LEFT_PADDING = 0.5
DEFAULT_OVER_UNDER = 20.0
MAX_OVER_UNDER = 100.0

plt.style.use("ggplot")


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def build_full_name(first_name, family_name):
    return " ".join(part for part in [first_name, family_name] if part)


def parse_minutes_float(value):
    text = clean_text(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_points(value):
    text = clean_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


@st.cache_data
def load_players():
    players_by_id = {}

    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {PLAYERS_CSV}")

        for row in reader:
            person_id = clean_text(row.get("personId"))
            if not person_id or person_id in players_by_id:
                continue

            first_name = clean_text(row.get("firstName"))
            family_name = clean_text(row.get("familyName"))
            full_name = build_full_name(first_name, family_name)

            players_by_id[person_id] = {
                "personId": person_id,
                "fullName": full_name,
            }

    return sorted(
        players_by_id.values(),
        key=lambda player: (player["fullName"].casefold(), player["personId"]),
    )


@st.cache_data
def load_player_games(person_id):
    games = []

    with PLAYERS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {PLAYERS_CSV}")

        for row in reader:
            if clean_text(row.get("personId")) != person_id:
                continue

            minutes_float = parse_minutes_float(row.get("minutesFloat"))
            if minutes_float is None or minutes_float <= 0:
                continue

            games.append(
                {
                    "gameDate": clean_text(row.get("gameDate")),
                    "gameId": clean_text(row.get("gameId")),
                    "points": parse_points(row.get("points")),
                }
            )

    return sorted(games, key=lambda game: (game["gameDate"], game["gameId"]))


def format_player_option(player):
    return player["fullName"] or player["personId"]


def render_player_selector(players):
    selector_col, lookback_col = st.columns([3, 1])

    with selector_col:
        selected_player = st.selectbox(
            "Player",
            players,
            format_func=format_player_option,
            index=None,
            placeholder="Select a player",
        )

    selected_games = []
    max_games = 1
    if selected_player is not None:
        selected_games = load_player_games(selected_player["personId"])
        if selected_games:
            max_games = len(selected_games)

    with lookback_col:
        lookback_window = st.number_input(
            "Lookback Window",
            min_value=1,
            max_value=max_games,
            value=min(DEFAULT_LOOKBACK_WINDOW, max_games),
            step=1,
            disabled=selected_player is None or not selected_games,
        )

    return selected_player, selected_games, int(lookback_window)


def build_histogram_bins(min_points, max_points):
    return [point - 0.5 for point in range(min_points, max_points + 2)]


def build_histogram_ticks(min_points, max_points):
    tick_start = (min_points // HISTOGRAM_LABEL_SPACING) * HISTOGRAM_LABEL_SPACING
    tick_end = (
        (max_points + HISTOGRAM_LABEL_SPACING - 1)
        // HISTOGRAM_LABEL_SPACING
        * HISTOGRAM_LABEL_SPACING
    )
    return list(range(tick_start, tick_end + 1, HISTOGRAM_LABEL_SPACING))


def summarize_points(games):
    point_values = [game["points"] for game in games]
    return {
        "games_available": len(point_values),
        "mean_points": sum(point_values) / len(point_values),
        "median_points": statistics.median(point_values),
        "median_bin": statistics.median_low(point_values),
    }


def summarize_over_under(games, over_under_value):
    total_games = len(games)
    under_games = sum(1 for game in games if game["points"] < over_under_value)
    push_games = sum(1 for game in games if game["points"] == over_under_value)
    over_games = total_games - under_games - push_games

    return {
        "total_games": total_games,
        "under_games": under_games,
        "push_games": push_games,
        "over_games": over_games,
        "under_pct": (under_games / total_games) * 100,
        "push_pct": (push_games / total_games) * 100,
        "over_pct": (over_games / total_games) * 100,
    }


def build_points_figure(games, over_under_value):
    game_numbers = list(range(1, len(games) + 1))
    point_values = [game["points"] for game in games]
    min_points = min(point_values)
    max_points = max(point_values)
    histogram_bins = build_histogram_bins(min_points, max_points)
    histogram_ticks = build_histogram_ticks(min_points, max_points)
    summary = summarize_points(games)

    fig, (ax_line, ax_hist) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    ax_line.plot(
        game_numbers,
        point_values,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#1f77b4",
    )
    ax_line.axhline(
        sum(point_values) / len(point_values),
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label="Mean",
    )
    ax_line.axhline(
        over_under_value,
        linestyle=":",
        linewidth=2,
        color="#2ca02c",
        label="O/U",
    )
    ax_line.set_title("Points Per Game")
    ax_line.set_xlabel("Game Number")
    ax_line.set_ylabel("Points")
    ax_line.legend(loc="best")

    ax_hist.hist(
        point_values,
        bins=histogram_bins,
        alpha=0.45,
        color="#4c78a8",
        edgecolor="white",
        rwidth=0.9,
    )
    ax_hist.set_title("Points Distribution")
    ax_hist.set_xlabel("Points")
    ax_hist.set_ylabel("Number of Games")
    ax_hist.set_xlim(min_points - 0.5 - HISTOGRAM_LEFT_PADDING, max_points + 0.5)
    ax_hist.set_xticks(histogram_ticks)
    ax_hist.axvline(
        summary["median_bin"],
        linestyle="--",
        linewidth=2,
        color="#ff7f0e",
        label=f"Median = {summary['median_points']:.1f}",
    )
    ax_hist.axvline(
        over_under_value,
        linestyle=":",
        linewidth=2,
        color="#2ca02c",
        label="O/U",
    )
    ax_hist.legend(loc="upper right")

    fig.tight_layout()
    return fig


def render_player_charts(player_name, games, over_under_value):
    fig = build_points_figure(games, over_under_value)
    summary = summarize_points(games)

    st.divider()
    st.markdown(
        f"""
        <div style="text-align: center;">
            <div style="font-size: 1.4rem; font-weight: 600;">{player_name} - Points</div>
            <div style="font-size: 0.95rem; color: #666; margin-top: 0.2rem;">
                Games available: {summary["games_available"]} |
                Mean points: {summary["mean_points"]:.1f} |
                Median points: {summary["median_points"]:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_over_under_input():
    return st.number_input(
        "Over / Under",
        min_value=0.5,
        max_value=MAX_OVER_UNDER,
        value=DEFAULT_OVER_UNDER,
        step=0.5,
        key="over_under_value",
    )


def render_over_under_summary(games, over_under_value):
    summary = summarize_over_under(games, over_under_value)
    st.markdown(
        (
            f"Empirical O/U at {over_under_value:g} points: "
            f"Under = {summary['under_games']} ({summary['under_pct']:.1f}%) | "
            f"Push = {summary['push_games']} ({summary['push_pct']:.1f}%) | "
            f"Over = {summary['over_games']} ({summary['over_pct']:.1f}%)"
        )
    )


def main():
    st.set_page_config(page_title="Player Selector", page_icon="🏀")
    st.title("Player Selector")
    st.caption("Choose a player from the box score dataset.")

    players = load_players()
    if not players:
        st.warning("No players were found in the source CSV.")
        return

    selected_player, selected_games, lookback_window = render_player_selector(players)

    if selected_player is None:
        st.info("Select a player to view their scoring charts.")
        return

    if not selected_games:
        st.warning("No games with positive minutes were found for this player.")
        return

    recent_games = selected_games[-lookback_window:]
    over_under_value = st.session_state.get("over_under_value", DEFAULT_OVER_UNDER)
    render_player_charts(selected_player["fullName"], recent_games, over_under_value)
    render_over_under_input()
    render_over_under_summary(recent_games, over_under_value)


if __name__ == "__main__":
    main()
