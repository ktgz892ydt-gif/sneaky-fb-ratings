"""
Fit the model's constants against history instead of guessing them.

Three numbers drive this system, and until now all three were my judgement:

    squash_scale   how fast margin saturates into a fractional win
    prior_games    how heavily last season's rating pulls on this one
    carry          how much of last season's rating survives into the next

This script replaces them with values chosen by how well they *predict games
the model has not seen*.

Protocol
--------
Walk-forward, which is the only honest way to score a rating system:

    for each evaluation season S
      fit season S-1 in full  ->  regress by `carry`  ->  preseason prior
      for each holdout week w
        fit on season S weeks 1..w-1 only
        predict every game in week w
        score the predictions

Nothing in the fit ever sees the week being predicted, so a combination cannot
win by memorising. Selection is by log loss rather than raw accuracy: log loss
punishes confident wrong calls, which is exactly the failure mode a rating
system should be penalised for. Accuracy alone would reward a model that is
right often but wildly overconfident.

The reported numbers are what the board actually achieves, not what it hopes
to. If accuracy comes out near 70% that is roughly the ceiling for high school
football, where roster variance is large; if calibration is off, the win
probabilities are not trustworthy even when the ranking is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ratings import RatingConfig, fit_bradley_terry, squash  # noqa: E402
from resolve import load_games, load_roster, resolve  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# ---------------------------------------------------------------------------
# Season loading
# ---------------------------------------------------------------------------

def load_season(year):
    g = os.path.join(DATA, f"games_{year}.csv")
    r = os.path.join(DATA, f"roster_{year}.csv")
    if not (os.path.exists(g) and os.path.exists(r)):
        return None
    res = resolve(load_roster(r), load_games(g))
    return res


def stable_key(team):
    """Identity that survives across seasons (divisions and regions do not)."""
    return (team.school_id or "").strip() or team.name


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit(team_ids, games, cfg, prior_pts=None):
    """Fit only the headline model. Returns ratings in points, and HFA."""
    index = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    hi = np.array([index[g["home"]] for g in games])
    ai = np.array([index[g["away"]] for g in games])
    margin = np.array([g["home_score"] - g["away_score"] for g in games], dtype=float)
    neutral = np.array([bool(g.get("neutral")) for g in games])

    y = squash(margin, cfg.squash_scale, cfg.margin_cap)
    prior_logit = None
    if prior_pts is not None:
        prior_logit = prior_pts / cfg.squash_scale
    r, hfa, _ = fit_bradley_terry(n, hi, ai, y, neutral, cfg, prior=prior_logit)
    return r * cfg.squash_scale, hfa * cfg.squash_scale


def full_season_ratings(res, cfg):
    ids = sorted(res.teams)
    pts, hfa = _fit(ids, res.games, cfg)
    played = defaultdict(int)
    for g in res.games:
        played[g["home"]] += 1
        played[g["away"]] += 1
    out = {}
    for i, t in enumerate(ids):
        if played[t] >= 4 and res.teams[t].in_ohio:
            out[stable_key(res.teams[t])] = float(pts[i])
    return {"ratings": out, "hfa": float(hfa)}


def prior_for(res, prev_ratings, carry, clip=14.0):
    """Map last season's ratings onto this season's team ids."""
    ids = sorted(res.teams)
    prev_ratings = (prev_ratings or {}).get("ratings", {})
    if not prev_ratings:
        return np.zeros(len(ids))
    mean = float(np.mean(list(prev_ratings.values())))
    out = np.zeros(len(ids))
    for i, t in enumerate(ids):
        v = prev_ratings.get(stable_key(res.teams[t]))
        if v is not None:
            out[i] = float(np.clip((v - mean) * carry, -clip, clip))
    return out - out.mean()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate(res, prev_ratings, cfg, carry, holdouts):
    ids = sorted(res.teams)
    index = {t: i for i, t in enumerate(ids)}
    prior = prior_for(res, prev_ratings, carry)
    prev_hfa = (prev_ratings or {}).get("hfa")

    by_week = defaultdict(list)
    for g in res.games:
        by_week[g["week"]].append(g)

    n = ll = correct = total = 0
    abs_err = 0.0
    bins = defaultdict(lambda: [0, 0])

    for w in holdouts:
        train = [g for g in res.games if g["week"] < w]
        test = by_week.get(w, [])
        if not test:
            continue
        if len(train) < 50:
            # Nothing in-season yet: the prior *is* the prediction.
            pts, hfa = prior, (prev_hfa if prev_hfa is not None else 1.5)
        else:
            pts, hfa = _fit(ids, train, cfg, prior)

        for g in test:
            h, a = index[g["home"]], index[g["away"]]
            pred = pts[h] - pts[a] + (0.0 if g.get("neutral") else hfa)
            p = 1.0 / (1.0 + np.exp(-pred / cfg.squash_scale))
            p = min(max(p, 1e-6), 1 - 1e-6)
            m = g["home_score"] - g["away_score"]
            y = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)

            ll += -(y * np.log(p) + (1 - y) * np.log(1 - p))
            if m != 0:
                correct += int((pred > 0) == (m > 0))
                total += 1
            abs_err += abs(pred - m)
            n += 1
            b = round(min(max(p, 0.5), 1.0), 1) if p >= 0.5 else round(1 - p, 1)
            fav_won = (p >= 0.5) == (y >= 0.5)
            bins[b][0] += int(fav_won)
            bins[b][1] += 1

    if n == 0:
        return None
    return {
        "n": n,
        "logloss": ll / n,
        "accuracy": correct / total if total else float("nan"),
        "mae_margin": abs_err / n,
        "calibration": {str(k): (v[0] / v[1], v[1]) for k, v in sorted(bins.items())},
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023,2024,2025",
                    help="comma separated, oldest first")
    ap.add_argument("--holdouts", default="6,8,10")
    ap.add_argument("--out", default=os.path.join(DATA, "tuned.json"))
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()

    years = [int(y) for y in args.seasons.split(",")]
    holdouts = [int(w) for w in args.holdouts.split(",")]

    loaded = {}
    for y in years:
        s = load_season(y)
        if s is None:
            print(f"  season {y}: no data on disk, skipping", file=sys.stderr)
            continue
        loaded[y] = s
        print(f"  season {y}: {len(s.games)} games, {len(s.teams)} teams", file=sys.stderr)

    evals = [y for y in years if y in loaded and (y - 1) in loaded]
    if not evals:
        raise SystemExit(
            "Need at least two consecutive seasons on disk to tune. "
            "Run the backfill first."
        )
    print(f"  evaluating on: {evals}\n", file=sys.stderr)

    if args.quick:
        grid_scale, grid_pg, grid_carry = [7.0, 9.0, 12.0], [1.0, 2.0], [0.4, 0.6]
    else:
        grid_scale = [6.0, 8.0, 10.0, 13.0]
        grid_pg = [0.75, 1.5, 3.0]
        grid_carry = [0.3, 0.45, 0.6, 0.75]

    prev_cache = {}
    results = []
    combos = list(itertools.product(grid_scale, grid_pg, grid_carry))
    for i, (scale, pg, carry) in enumerate(combos, 1):
        cfg = RatingConfig(squash_scale=scale, prior_games=pg)
        agg = {"n": 0, "ll": 0.0, "corr": 0.0, "mae": 0.0}
        for S in evals:
            ck = (S - 1, scale, pg)
            if ck not in prev_cache:
                prev_cache[ck] = full_season_ratings(loaded[S - 1], cfg)
            m = evaluate(loaded[S], prev_cache[ck], cfg, carry, holdouts)
            if not m:
                continue
            agg["n"] += m["n"]
            agg["ll"] += m["logloss"] * m["n"]
            agg["corr"] += m["accuracy"] * m["n"]
            agg["mae"] += m["mae_margin"] * m["n"]
        if agg["n"] == 0:
            continue
        results.append({
            "squash_scale": scale, "prior_games": pg, "carry": carry,
            "logloss": agg["ll"] / agg["n"],
            "accuracy": agg["corr"] / agg["n"],
            "mae_margin": agg["mae"] / agg["n"],
            "n": agg["n"],
        })
        print(f"  [{i:>3}/{len(combos)}] scale={scale:<5} prior={pg:<5} carry={carry:<5} "
              f"logloss={results[-1]['logloss']:.4f} acc={results[-1]['accuracy']:.1%}",
              file=sys.stderr)

    results.sort(key=lambda r: r["logloss"])
    best = results[0]

    # Calibration for the winner, so the report says whether the probabilities
    # mean anything, not just whether the ordering is good.
    cfg = RatingConfig(squash_scale=best["squash_scale"], prior_games=best["prior_games"])
    S = evals[-1]
    ck = (S - 1, best["squash_scale"], best["prior_games"])
    detail = evaluate(loaded[S], prev_cache[ck], cfg, best["carry"], holdouts)

    payload = {
        "tunedOn": evals,
        "holdoutWeeks": holdouts,
        "best": best,
        "calibration": detail["calibration"] if detail else None,
        "leaderboard": results[:10],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\n" + "=" * 64, file=sys.stderr)
    print(f"BEST  squash_scale={best['squash_scale']}  "
          f"prior_games={best['prior_games']}  carry={best['carry']}", file=sys.stderr)
    print(f"      log loss {best['logloss']:.4f} | accuracy {best['accuracy']:.1%} "
          f"| margin error {best['mae_margin']:.1f} pts | {best['n']} games",
          file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
