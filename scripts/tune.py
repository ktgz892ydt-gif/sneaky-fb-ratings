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
    out, by_div = {}, defaultdict(list)
    for i, t in enumerate(ids):
        tm = res.teams[t]
        if played[t] >= 4 and tm.in_ohio:
            out[stable_key(tm)] = float(pts[i])
            if tm.division:
                by_div[tm.division].append(float(pts[i]))
    # Measured division ladder, centred so it adds no overall level.
    eff = {d: sum(v) / len(v) for d, v in by_div.items() if len(v) >= 10}
    if eff:
        c = sum(eff.values()) / len(eff)
        eff = {d: v - c for d, v in eff.items()}
    return {"ratings": out, "hfa": float(hfa), "divEffects": eff,
            "divOf": {stable_key(res.teams[t]): res.teams[t].division for t in ids}}


def prior_for(res, prev, carry, division_weight=1.0, clip=14.0):
    """Compose each team's starting point.

        prior = division_weight * (its division's measured baseline)
              + carry * (what it personally earned above that baseline)

    Both parts come from measurement. The division part is estimated from
    last season's cross-division results, never from enrollment.
    """
    ids = sorted(res.teams)
    prev = prev or {}
    ratings = prev.get("ratings", {})
    eff = prev.get("divEffects", {}) or {}
    div_of_prev = prev.get("divOf", {}) or {}
    if not ratings and not eff:
        return np.zeros(len(ids))
    mean = float(np.mean(list(ratings.values()))) if ratings else 0.0

    out = np.zeros(len(ids))
    for i, t in enumerate(ids):
        tm = res.teams[t]
        if not tm.in_ohio:
            continue
        base = eff.get(tm.division, 0.0) * division_weight
        key = stable_key(tm)
        v = ratings.get(key)
        dev = 0.0
        if v is not None:
            dev = ((v - mean) - eff.get(div_of_prev.get(key), 0.0)) * carry
            dev = float(np.clip(dev, -clip, clip))
        out[i] = base + dev
    return out - out.mean()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate(res, prev_ratings, cfg, carry, holdouts, division_weight=1.0):
    ids = sorted(res.teams)
    index = {t: i for i, t in enumerate(ids)}
    prior = prior_for(res, prev_ratings, carry, division_weight)
    prev_hfa = (prev_ratings or {}).get("hfa")

    by_week = defaultdict(list)
    for g in res.games:
        by_week[g["week"]].append(g)

    n = ll = correct = total = 0
    ll_sq = 0.0
    abs_err = 0.0
    bins = defaultdict(lambda: [0, 0])
    week_stats = defaultdict(lambda: {"n": 0, "ll": 0.0, "correct": 0, "total": 0})

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

            gll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            ll += gll
            ll_sq += gll * gll
            if m != 0:
                correct += int((pred > 0) == (m > 0))
                total += 1
            abs_err += abs(pred - m)
            n += 1
            ws = week_stats[w]
            ws["n"] += 1
            ws["ll"] += gll
            if m != 0:
                ws["total"] += 1
                ws["correct"] += int((pred > 0) == (m > 0))

            b = round(min(max(p, 0.5), 1.0), 1) if p >= 0.5 else round(1 - p, 1)
            fav_won = (p >= 0.5) == (y >= 0.5)
            bins[b][0] += int(fav_won)
            bins[b][1] += 1

    if n == 0:
        return None
    per_week = dict(week_stats)
    return {
        "n": n,
        "logloss": ll / n,
        "ll_sum": ll,
        "ll_sumsq": ll_sq,
        "perWeek": per_week,
        # correct/total are returned raw so callers can aggregate across
        # seasons without reweighting a ratio whose denominator (non-tie
        # games) differs from n (all games).
        "correct": correct,
        "total": total,
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
        grid_dw = [0.0, 1.0]
    else:
        # Widened twice, each time because the optimum landed on a boundary.
        grid_scale = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        grid_pg = [0.05, 0.1, 0.25, 0.5, 0.75, 1.5]
        # carry stops at 1.0 deliberately. Above 1.0 the model would amplify
        # last season's estimate rather than regress it -- claiming this year's
        # team is *more* extreme than last year's measurement. A backtest can
        # reward that (last season's ratings are themselves shrunk, so
        # un-shrinking them fits better) but it is not a claim about football,
        # and it compounds badly when a program actually collapses.
        grid_carry = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        grid_dw = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    prev_cache = {}
    results = []
    combos = list(itertools.product(grid_scale, grid_pg, grid_carry, grid_dw))
    for i, (scale, pg, carry, dw) in enumerate(combos, 1):
        cfg = RatingConfig(squash_scale=scale, prior_games=pg, division_weight=dw)
        agg = {"n": 0, "ll": 0.0, "llsq": 0.0, "correct": 0, "total": 0,
               "mae": 0.0, "perSeason": {}}
        for S in evals:
            ck = (S - 1, scale, pg)
            if ck not in prev_cache:
                prev_cache[ck] = full_season_ratings(loaded[S - 1], cfg)
            m = evaluate(loaded[S], prev_cache[ck], cfg, carry, holdouts, dw)
            if not m:
                continue
            agg["n"] += m["n"]
            agg["ll"] += m["logloss"] * m["n"]
            agg["llsq"] += m["ll_sumsq"]
            agg["perSeason"][str(S)] = round(m["logloss"], 4)
            agg["correct"] += m["correct"]
            agg["total"] += m["total"]
            agg["mae"] += m["mae_margin"] * m["n"]
        if agg["n"] == 0:
            continue
        results.append({
            "ll_sumsq": agg["llsq"],
            "perSeason": agg["perSeason"],
            "squash_scale": scale, "prior_games": pg, "carry": carry,
            "division_weight": dw,
            "logloss": agg["ll"] / agg["n"],
            "accuracy": (agg["correct"] / agg["total"]) if agg["total"] else float("nan"),
            "decided": agg["total"],
            "mae_margin": agg["mae"] / agg["n"],
            "n": agg["n"],
        })
        print(f"  [{i:>3}/{len(combos)}] scale={scale:<5} prior={pg:<5} carry={carry:<5} "
              f"div={dw:<4} "
              f"logloss={results[-1]['logloss']:.4f} acc={results[-1]['accuracy']:.1%}",
              file=sys.stderr)

    results.sort(key=lambda r: r["logloss"])
    raw_best = results[0]

    # A grid search on two evaluation seasons will happily chase noise to the
    # edge of the grid. So: compute the standard error of the best score, then
    # among every configuration statistically indistinguishable from it, take
    # the most conservative one. This is the one-standard-error rule, and it is
    # the difference between "the best number we saw" and "the best number we
    # can defend".
    var = max(raw_best["ll_sumsq"] / raw_best["n"] - (raw_best["logloss"] ** 2), 0.0)
    se = (var / raw_best["n"]) ** 0.5 if raw_best["n"] else 0.0
    threshold = raw_best["logloss"] + se

    DEFAULTS = {"squash_scale": 9.0, "prior_games": 1.5, "carry": 0.5,
                "division_weight": 1.0}

    def conservatism(r):
        """Distance from the documented defaults, scaled per parameter."""
        return (abs(r["carry"] - DEFAULTS["carry"]) / 0.5
                + abs(r["division_weight"] - DEFAULTS["division_weight"]) / 1.0
                + abs(r["squash_scale"] - DEFAULTS["squash_scale"]) / 9.0
                + abs(r["prior_games"] - DEFAULTS["prior_games"]) / 1.5)

    within = [r for r in results if r["logloss"] <= threshold]
    best = min(within, key=conservatism) if within else raw_best
    best["selectedBy"] = ("one-standard-error rule" if best is not raw_best
                          else "outright best")
    best["seLogloss"] = round(se, 5)
    best["candidatesWithinOneSE"] = len(within)

    # Calibration for the winner, so the report says whether the probabilities
    # mean anything, not just whether the ordering is good.
    cfg = RatingConfig(squash_scale=best["squash_scale"], prior_games=best["prior_games"],
                       division_weight=best["division_weight"])
    S = evals[-1]
    ck = (S - 1, best["squash_scale"], best["prior_games"])
    detail = evaluate(loaded[S], prev_cache[ck], cfg, best["carry"], holdouts,
                      best["division_weight"])

    # An optimum on the edge of the search means the grid constrained the
    # answer. Say so rather than presenting it as settled.
    edges = []
    for key, grid in (("squash_scale", grid_scale), ("prior_games", grid_pg),
                      ("carry", grid_carry), ("division_weight", grid_dw)):
        if len(grid) > 1 and best[key] in (min(grid), max(grid)):
            edges.append(f"{key}={best[key]} (grid {min(grid)}..{max(grid)})")
    if edges:
        print("\n  NOTE: the best point sits on the edge of the grid for: "
              + "; ".join(edges), file=sys.stderr)
        print("  Widen those ranges and re-run before treating this as final.",
              file=sys.stderr)

    # Stability report. If the constants that win on one season are beaten
    # badly on another, the fit is chasing that season, not the sport.
    stability = []
    for S in evals:
        per = [(r["perSeason"].get(str(S)), r) for r in results
               if r.get("perSeason", {}).get(str(S)) is not None]
        if not per:
            continue
        per.sort(key=lambda x: x[0])
        w = per[0][1]
        stability.append({
            "season": S,
            "bestHere": {k: w[k] for k in ("squash_scale", "prior_games",
                                           "carry", "division_weight")},
            "bestHereLogloss": per[0][0],
            "chosenConfigLogloss": best.get("perSeason", {}).get(str(S)),
        })

    print("\n  per-season stability:", file=sys.stderr)
    for st in stability:
        b = st["bestHere"]
        gap = ((st["chosenConfigLogloss"] or 0) - st["bestHereLogloss"])
        print(f"    {st['season']}: best here carry={b['carry']} "
              f"div={b['division_weight']} scale={b['squash_scale']} "
              f"(logloss {st['bestHereLogloss']:.4f}); "
              f"chosen config is {gap:+.4f} off that", file=sys.stderr)

    if detail and detail.get("perWeek"):
        print("\n  by holdout week (chosen config):", file=sys.stderr)
        for w in sorted(detail["perWeek"], key=int):
            ws = detail["perWeek"][w]
            acc = ws["correct"] / ws["total"] if ws["total"] else float("nan")
            print(f"    week {w:>2}: logloss {ws['ll']/ws['n']:.4f}  "
                  f"accuracy {acc:.1%}  ({ws['n']} games)", file=sys.stderr)

    payload = {
        "tunedOn": evals,
        "atGridEdge": edges,
        "stability": stability,
        "perWeek": (detail or {}).get("perWeek"),
        "selection": {
            "rule": best.get("selectedBy"),
            "standardError": best.get("seLogloss"),
            "candidatesWithinOneSE": best.get("candidatesWithinOneSE"),
            "outrightBest": {k: raw_best[k] for k in
                             ("squash_scale", "prior_games", "carry",
                              "division_weight", "logloss")},
        },
        "holdoutWeeks": holdouts,
        "best": best,
        "calibration": detail["calibration"] if detail else None,
        "leaderboard": results[:10],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("\n" + "=" * 64, file=sys.stderr)
    if best is not raw_best:
        print(f"Outright best was scale={raw_best['squash_scale']} "
              f"prior={raw_best['prior_games']} carry={raw_best['carry']} "
              f"div={raw_best['division_weight']} "
              f"(logloss {raw_best['logloss']:.4f}).", file=sys.stderr)
        print(f"{len(within)} configurations sit within one standard error "
              f"({se:.4f}) of it; taking the most conservative.", file=sys.stderr)
        print("-" * 64, file=sys.stderr)
    print(f"BEST  squash_scale={best['squash_scale']}  "
          f"prior_games={best['prior_games']}  carry={best['carry']}  "
          f"division_weight={best['division_weight']}", file=sys.stderr)
    print(f"      log loss {best['logloss']:.4f} | accuracy {best['accuracy']:.1%} "
          f"| margin error {best['mae_margin']:.1f} pts | {best['n']} games",
          file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
