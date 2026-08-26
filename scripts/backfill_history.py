"""
Replay past seasons into the history log, marked as backtests.

Walk-forward, the same protocol tune.py uses: fit on weeks 1..N, predict week
N+1, never letting the fit see the week being predicted. That much is honest.

What it is NOT is a live track record. The model's constants were fitted on
these seasons, so a backtest here is a weaker claim than a week captured before
the games were played. Every line written carries kind="backtest" and the
scorecard reports the two separately; see the note on KIND_LIVE in history.py.

Run once. It is idempotent -- a (season, week) already in the log is skipped --
so re-running cannot overwrite a live capture with a replay.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harbin import LAST_REGULAR_WEEK  # noqa: E402
from history import (KIND_BACKTEST, append_if_new,  # noqa: E402
                     build_snapshot)
from ratings import RatingConfig, rate, win_probability  # noqa: E402
from resolve import load_games, load_roster, resolve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def config_from_tuned():
    import json
    p = os.path.join(DATA, "tuned.json")
    cfg = RatingConfig()
    if not os.path.exists(p):
        return cfg, None
    with open(p, encoding="utf-8") as fh:
        tb = json.load(fh)
    best = tb.get("best") or {}
    ps = tb.get("probScale") or {}
    return RatingConfig(
        squash_scale=float(best.get("squash_scale", cfg.squash_scale)),
        prior_games=float(best.get("prior_games", cfg.prior_games)),
        division_weight=float(best.get("division_weight", cfg.division_weight)),
        prob_scale_a=float(ps.get("a", cfg.prob_scale_a)),
        prob_scale_b=float(ps.get("b", cfg.prob_scale_b)),
    ), {k: best.get(k) for k in ("squash_scale", "prior_games", "carry",
                                 "division_weight")}


def replay(season, path, cfg, tuned_meta):
    g = os.path.join(DATA, f"games_{season}.csv")
    r = os.path.join(DATA, f"roster_{season}.csv")
    if not (os.path.exists(g) and os.path.exists(r)):
        print(f"  {season}: no data on disk, skipping", file=sys.stderr)
        return 0
    res = resolve(load_roster(r), load_games(g))
    ids = sorted(res.teams)
    by_week = defaultdict(list)
    for gm in res.games:
        by_week[gm.get("week", 1)].append(gm)

    written = 0
    for through in range(1, LAST_REGULAR_WEEK):
        train = [gm for gm in res.games if gm.get("week", 1) <= through]
        test = by_week.get(through + 1, [])
        if len(train) < 50 or not test:
            continue

        result = rate(ids, train, cfg)
        idx = {t: i for i, t in enumerate(ids)}
        played = defaultdict(int)
        for gm in train:
            played[gm["home"]] += 1
            played[gm["away"]] += 1

        # Shaped exactly like build.py's payload rows and schedule entries, so
        # a replayed line and a live line are the same thing on disk.
        team_rows = [{
            "name": res.teams[t].name,
            "inOhio": res.teams[t].in_ohio,
            "games": played[t],
            "rating": round(float(result.bt_margin[idx[t]]), 2),
            "rank": None,
            "playoffOdds": None,
        } for t in ids]

        schedule = []
        for gm in test:
            h, a = gm["home"], gm["away"]
            margin = (result.bt_margin[idx[h]] - result.bt_margin[idx[a]]
                      + (0.0 if gm.get("neutral") else result.hfa_margin))
            est = min(played[h], played[a])
            schedule.append({
                "predicted": True,
                "week": through + 1,
                "homeName": res.teams[h].name,
                "awayName": res.teams[a].name,
                "predictedHomeMargin": round(float(margin), 1),
                "homeWinProb": round(float(win_probability(margin, est, cfg)), 3),
            })

        snap = build_snapshot(season, through, f"backtest:{season}w{through}",
                              tuned_meta, team_rows, schedule,
                              kind=KIND_BACKTEST, include_teams=False)
        if append_if_new(path, snap):
            written += 1
    print(f"  {season}: wrote {written} backtest week(s)", file=sys.stderr)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--out", default=os.path.join(DATA, "history.jsonl"))
    a = ap.parse_args()
    cfg, tuned_meta = config_from_tuned()
    print(f"replaying with scale={cfg.squash_scale} prior={cfg.prior_games}",
          file=sys.stderr)
    total = sum(replay(int(y), a.out, cfg, tuned_meta)
                for y in a.seasons.split(","))
    print(f"-> {total} backtest weeks in {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
