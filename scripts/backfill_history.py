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

from build import (expected_total_points, projected_score,  # noqa: E402
                   scoring_profile)
from harbin import LAST_REGULAR_WEEK  # noqa: E402
from history import (KIND_BACKTEST, append_if_new,  # noqa: E402
                     build_snapshot)
from ratings import (RatingConfig, expected_margin, rate,  # noqa: E402
                     win_probability)
from resolve import load_games, load_roster, resolve  # noqa: E402
from tune import full_season_ratings, prior_for  # noqa: E402

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
        margin_scale=float(ps.get("marginScale") or cfg.margin_scale),
    ), {k: best.get(k) for k in ("squash_scale", "prior_games", "carry",
                                 "division_weight")}


def _season(year):
    g = os.path.join(DATA, f"games_{year}.csv")
    r = os.path.join(DATA, f"roster_{year}.csv")
    if not (os.path.exists(g) and os.path.exists(r)):
        return None
    return resolve(load_roster(r), load_games(g))


def replay(season, path, cfg, tuned_meta, carry):
    res = _season(season)
    if res is None:
        print(f"  {season}: no data on disk, skipping", file=sys.stderr)
        return 0
    ids = sorted(res.teams)
    by_week = defaultdict(list)
    for gm in res.games:
        by_week[gm.get("week", 1)].append(gm)

    # The live build rates with a preseason prior carried from the previous
    # season, and tune.py's walk-forward does the same. A replay that skips it
    # is not measuring the model that ships -- and the gap is widest in exactly
    # the early weeks a backtest covers, because that is when the prior does
    # most of the work. Without this the published track record graded a
    # prior-free model.
    prev = _season(season - 1)
    prior = None
    if prev is not None:
        prior_pts = prior_for(res, full_season_ratings(prev, cfg), carry,
                              cfg.division_weight)
        prior = {t: float(prior_pts[i]) for i, t in enumerate(ids)}
        print(f"  {season}: prior carried from {season - 1}", file=sys.stderr)
    else:
        print(f"  {season}: no {season - 1} data, replaying without a prior "
              f"(as the live build would if it had none)", file=sys.stderr)

    written = 0
    for through in range(1, LAST_REGULAR_WEEK):
        train = [gm for gm in res.games if gm.get("week", 1) <= through]
        test = by_week.get(through + 1, [])
        if len(train) < 50 or not test:
            continue

        result = rate(ids, train, cfg, priors=prior)
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

        # Same score layer the live build publishes, so a replayed row and a
        # live row are the same shape on disk.
        prof = scoring_profile(type("R", (), {"games": train})(), ids)
        schedule = []
        for gm in test:
            h, a = gm["home"], gm["away"]
            margin = (result.bt_margin[idx[h]] - result.bt_margin[idx[a]]
                      + (0.0 if gm.get("neutral") else result.hfa_margin))
            est = min(played[h], played[a])
            shown = round(float(expected_margin(margin, cfg)), 1)
            proj_h, proj_a = projected_score(
                shown, expected_total_points(h, a, prof, idx))
            schedule.append({
                "predicted": True,
                "week": through + 1,
                "homeName": res.teams[h].name,
                "awayName": res.teams[a].name,
                # Same split as the live build: the probability from the raw
                # difference, the published margin calibrated.
                "predictedHomeMargin": shown,
                "homeWinProb": round(float(win_probability(margin, est, cfg)), 3),
                "projectedHomeScore": proj_h,
                "projectedAwayScore": proj_a,
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
    carry = float((tuned_meta or {}).get("carry") or 0.5)
    total = sum(replay(int(y), a.out, cfg, tuned_meta, carry)
                for y in a.seasons.split(","))
    print(f"-> {total} backtest weeks in {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
