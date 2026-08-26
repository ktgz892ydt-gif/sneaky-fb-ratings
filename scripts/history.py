"""
The week-by-week record: what the board said, and how it turned out.

THIS FILE'S DATA CANNOT BE REGENERATED. Everything else in data/ is derived
from the source and can be rebuilt by re-scraping. `data/history.jsonl` is the
one exception, and the exception matters: a prediction is what the board said
*at a moment in time*, before the game was played. Recomputing it afterwards
from a model that has since seen the result is not a prediction, it is a
retrofit, and it would flatter the record without anyone noticing. So the log
is append-only, and if it is lost the track record is genuinely gone.

That property is also why this is the honest foundation for comparing against
anyone else's model. A claim to have beaten another forecaster needs their
week-by-week calls archived at the time too; without that, any comparison is
reconstruction. What can be done is to start keeping score now, publicly, and
let the record accumulate.

Shape
-----
One JSON object per line, one line per (season, throughWeek). Each line is
self-contained -- it carries its own key list rather than referring to another
line -- because an append-only log should not have cross-line dependencies that
a truncated write could corrupt.

    season, throughWeek, generatedAt
    tuned        the constants that produced these numbers, so a change in the
                 model is visible in the record rather than silently mixed in
    keys         team names, in a fixed order
    teams        [rating, rank, playoffOdds] parallel to keys
    pred         [homeKey, awayKey, week, predictedMargin, homeWinProb]

Predictions are stored for the NEXT unplayed week only. That is the claim worth
being held to -- "we called this Friday right" -- and it keeps the file small
enough to live in the repo for years.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

LEAD_WEEKS = 1          # how far ahead predictions are recorded


def load(path):
    """Read the log. A corrupt trailing line is skipped, not fatal.

    A half-written line is what an interrupted commit looks like, and losing
    the whole history to one bad append would be a poor trade.
    """
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# Two kinds of record, and they must never be pooled.
#
#   live      captured before the games were played, by the build that was
#             running that week. Genuinely out of sample: there was no result
#             for the model to have seen.
#
#   backtest  replayed afterwards from committed scores -- fit on weeks 1..N,
#             predict week N+1. The walk-forward is honest, but the model's
#             CONSTANTS were tuned on these same seasons, so a backtest cannot
#             be evidence in the way a live week is. It is shown so the page is
#             not blank for the first ten weeks of a season, labelled as what
#             it is, and reported separately -- averaging the two together
#             would launder the weaker number into the stronger one.
KIND_LIVE = "live"
KIND_BACKTEST = "backtest"


def build_snapshot(season, through_week, generated_at, tuned, team_rows,
                   schedule, lead=LEAD_WEEKS, kind=KIND_LIVE,
                   include_teams=True):
    """Capture what the board is claiming right now.

    `team_rows` are the payload's team dicts; `schedule` its predicted fixtures
    with resolved names.
    """
    # The per-team block exists for the trend lines, and only a LIVE capture
    # needs to carry it: a backtest is replayable by definition, so storing its
    # ratings is duplicating something `backfill_history.py` can regenerate in
    # two seconds. Keeping them cost 1.2 MB of the log's first 1.7.
    if include_teams:
        ohio = [t for t in team_rows if t.get("inOhio") and t.get("games", 0) > 0]
        keys = [t["name"] for t in ohio]
        teams = [[t.get("rating"), t.get("rank"), t.get("playoffOdds")] for t in ohio]
    else:
        keys, teams = [], []

    horizon = through_week + lead
    pred = []
    for g in schedule:
        if not g.get("predicted"):
            continue
        if not (through_week < g["week"] <= horizon):
            continue
        pred.append([g["homeName"], g["awayName"], g["week"],
                     g["predictedHomeMargin"], g["homeWinProb"]])

    return {
        "season": season,
        "throughWeek": through_week,
        "kind": kind,
        "generatedAt": generated_at,
        "tuned": tuned,
        "keys": keys,
        "teams": teams,
        "pred": pred,
    }


def append_if_new(path, snap):
    """Append unless this (season, throughWeek) is already recorded.

    Idempotent on purpose. The workflow builds twice a week and a developer may
    build many times a day; only the first capture of a given week is the
    prediction, and later ones would have seen more of the season. Re-recording
    would quietly replace a forecast with hindsight.
    """
    for existing in load(path):
        if (existing.get("season") == snap["season"]
                and existing.get("throughWeek") == snap["throughWeek"]):
            return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, separators=(",", ":")) + "\n")
    return True


def score(snapshots, results_by_season):
    """Grade every recorded prediction against what actually happened.

    `results_by_season` maps season -> {(week, home, away): margin}. A
    prediction with no matching result is simply not yet scorable and is
    skipped -- never counted as a miss, which would punish the board for games
    the source has not posted.
    """
    per_season = defaultdict(lambda: {"n": 0, "correct": 0, "ll": 0.0,
                                      "absMargin": 0.0, "brier": 0.0,
                                      "kind": KIND_LIVE})
    bins = defaultdict(lambda: [0, 0])

    for snap in snapshots:
        got = results_by_season.get(snap.get("season")) or {}
        for home, away, week, margin, prob in snap.get("pred", []):
            actual = got.get((week, home, away))
            if actual is None or actual == 0:
                continue          # unplayed, or a tie: no favourite to be right about
            s = per_season[(snap.get("kind", KIND_LIVE), snap["season"])]
            p = min(max(float(prob), 1e-6), 1 - 1e-6)
            won = actual > 0
            s["n"] += 1
            s["correct"] += int((margin > 0) == won)
            s["ll"] += -(math.log(p) if won else math.log(1 - p))
            s["absMargin"] += abs(margin - actual)
            s["brier"] += (p - (1.0 if won else 0.0)) ** 2
            # Calibration, folded onto the favourite so 30% and 70% are one bin.
            fav_p = p if p >= 0.5 else 1 - p
            fav_won = won if p >= 0.5 else not won
            b = min(round(fav_p, 1), 1.0)
            bins[b][0] += int(fav_won)
            bins[b][1] += 1

    out = {}
    for (kind, yr), s in per_season.items():
        if not s["n"]:
            continue
        out.setdefault(kind, {})[str(yr)] = {
            "games": s["n"],
            "accuracy": round(s["correct"] / s["n"], 4),
            "logloss": round(s["ll"] / s["n"], 4),
            "brier": round(s["brier"] / s["n"], 4),
            "meanMarginError": round(s["absMargin"] / s["n"], 2),
        }
    def _pool(seasons):
        n = sum(v["games"] for v in seasons.values())
        if not n:
            return {}
        return {
            "games": n,
            "accuracy": round(sum(v["accuracy"] * v["games"] for v in seasons.values()) / n, 4),
            "logloss": round(sum(v["logloss"] * v["games"] for v in seasons.values()) / n, 4),
            "brier": round(sum(v["brier"] * v["games"] for v in seasons.values()) / n, 4),
            "meanMarginError": round(
                sum(v["meanMarginError"] * v["games"] for v in seasons.values()) / n, 2),
        }

    return {
        "bySeason": out,
        # Deliberately two totals, never one. See the note on KIND_LIVE.
        "overall": {k: _pool(v) for k, v in out.items()},
        "calibration": {f"{k:.1f}": {"predicted": k, "actual": round(v[0] / v[1], 4),
                                     "n": v[1]}
                        for k, v in sorted(bins.items()) if v[1] >= 20},
    }


def trends(snapshots, season, names):
    """-> {team name: {"w": [weeks], "rating": [...], "odds": [...]}}.

    Only teams asked for, so a payload does not carry a series for every school
    that ever appeared.
    """
    wanted = set(names)
    series = {n: {"w": [], "rating": [], "odds": []} for n in wanted}
    for snap in sorted((s for s in snapshots if s.get("season") == season),
                       key=lambda s: s.get("throughWeek", 0)):
        wk = snap.get("throughWeek")
        for key, row in zip(snap.get("keys", []), snap.get("teams", [])):
            if key not in wanted:
                continue
            series[key]["w"].append(wk)
            series[key]["rating"].append(row[0])
            series[key]["odds"].append(row[2])
    return {n: v for n, v in series.items() if len(v["w"]) >= 2}
