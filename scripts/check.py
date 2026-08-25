"""
Verification pass. Runs after every build; the workflow fails if it fails.

The point is to catch the failure modes that produce a page which *looks*
fine: a scrape that silently returned a third of the games, a name collision
that merged two schools, a solver that didn't converge, a rating column full
of NaN. Every one of those still renders a neat table.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ratings import RatingConfig, squash  # noqa: E402
from tune import SCHEMA_VERSION as TUNED_SCHEMA_VERSION  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATINGS = os.path.join(ROOT, "site", "ratings.json")

fails, warns = [], []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def warn(cond, msg):
    if not cond:
        warns.append(msg)


def main():
    # ---- model invariants (these are pure math and must always hold)
    cfg = RatingConfig()
    s, cap = cfg.squash_scale, cfg.margin_cap
    check(abs(squash(np.array([0.0]), s, cap)[0] - 0.5) < 1e-12,
          "squash(0) must be exactly 0.5")
    for m in (1, 3, 7, 14, 21, 35, 60):
        a = squash(np.array([float(m)]), s, cap)[0]
        b = squash(np.array([float(-m)]), s, cap)[0]
        check(abs((a + b) - 1.0) < 1e-12,
              f"squash must be symmetric: squash({m}) + squash({-m}) != 1")
    inc = [squash(np.array([float(m)]), s, cap)[0] for m in range(0, 50)]
    check(all(inc[i] <= inc[i + 1] for i in range(len(inc) - 1)),
          "squash must be monotonically increasing in margin")
    check(squash(np.array([200.0]), s, cap)[0] == squash(np.array([float(cap)]), s, cap)[0],
          "margins beyond the cap must be clipped, not extrapolated")

    if not os.path.exists(RATINGS):
        fails.append(f"{RATINGS} does not exist -- build.py did not run")
        report()
        return

    with open(RATINGS, encoding="utf-8") as fh:
        d = json.load(fh)

    teams = d["teams"]
    ohio = [t for t in teams if t["inOhio"]]

    # ---- solver health
    check(d["converged"], "the Bradley-Terry solver reported non-convergence")

    for t in teams:
        for k in ("rating", "btBinary", "massey", "sos"):
            v = t[k]
            check(v is not None and not math.isnan(v) and not math.isinf(v),
                  f"{t['id']}.{k} is not a finite number ({v})")

    # ---- scale sanity: ratings are centered and in a plausible points range
    rs = np.array([t["rating"] for t in teams])
    check(abs(rs.mean()) < 0.5, f"ratings should centre near zero, mean is {rs.mean():.3f}")
    check(rs.max() < 80 and rs.min() > -80,
          f"rating range {rs.min():.1f}..{rs.max():.1f} is implausible for points")

    # ---- coverage
    check(len(ohio) > 600,
          f"only {len(ohio)} Ohio teams resolved; OHSAA fields ~708 in 11-man")
    warn(len(ohio) >= 690, f"{len(ohio)} Ohio teams resolved, expected ~700+")
    check(d["gameCount"] > 200, f"only {d['gameCount']} games -- scrape looks partial")

    weeks = d["weeksLoaded"]
    expected_min = 300 * len(weeks)
    warn(d["gameCount"] >= expected_min * 0.75,
         f"{d['gameCount']} games across {len(weeks)} week(s) is low; "
         f"expected roughly {expected_min}")

    # ---- the collision guard: no team may play twice in one week
    integrity = [w for w in d["warnings"] if w.startswith("INTEGRITY")]
    check(not integrity, "integrity warnings from the resolver: " + "; ".join(integrity))

    # ---- ranks are dense, unique and ordered by rating
    ranked = sorted([t for t in ohio if t["rank"]], key=lambda t: t["rank"])
    playedn = [t for t in ohio if t["games"] > 0]
    check(len(ranked) == len(playedn),
          f"every Ohio team that has played should carry a rank "
          f"({len(ranked)} ranked vs {len(playedn)} played)")
    check(all(not t["rank"] for t in ohio if t["games"] == 0),
          "a team with no games this season must not hold a rank")
    check(all(t.get("unplayed") == (t["games"] == 0) for t in ohio),
          "the unplayed flag must agree with the game count")
    check([t["rank"] for t in ranked] == list(range(1, len(ranked) + 1)),
          "ranks must be 1..N with no gaps or duplicates")
    for a, b in zip(ranked, ranked[1:]):
        check(a["rating"] >= b["rating"] - 1e-9,
              f"rank order violates rating order at {a['name']} / {b['name']}")

    # ---- record consistency against the games actually loaded
    for t in teams:
        check(t["w"] + t["l"] + t["t"] == t["games"],
              f"{t['id']}: W+L+T ({t['w']}+{t['l']}+{t['t']}) != games ({t['games']})")

    # ---- home field advantage should be small and positive-ish
    hfa = d["hfa"]["rating"]
    warn(-1.0 < hfa < 6.0,
         f"fitted home-field advantage of {hfa:.2f} pts is outside the usual range")

    # ---- ambiguity should be reported, not hidden
    amb = [t for t in ohio if t["ambiguous"]]
    warn(len(amb) < 40, f"{len(amb)} Ohio teams flagged ambiguous -- resolver is struggling")
    for t in amb:
        check(bool(t["note"]), f"{t['id']} is flagged ambiguous but carries no explanation")

    # ---- the tuned metadata must have been produced by the current script
    tpath = os.path.join(ROOT, "data", "tuned.json")
    if os.path.exists(tpath):
        try:
            with open(tpath, encoding="utf-8") as fh:
                tb = json.load(fh)
        except (json.JSONDecodeError, OSError):
            tb = None
        if tb is not None:
            ver = tb.get("schema", 1)
            warn(ver >= TUNED_SCHEMA_VERSION,
                 f"data/tuned.json was written by an older tune.py "
                 f"(schema {ver}, current {TUNED_SCHEMA_VERSION}). Its field "
                 f"names no longer match the script or the README -- re-run "
                 f"the workflow with 'Re-fit the model constants' ticked.")
            if ver >= 2:
                check("outrightBestAtGridEdge" in tb and
                      "selectedConfigAtGridEdge" in tb,
                      "a schema 2 tuned.json must carry both edge fields")

    # ---- connectivity must be reported honestly
    c = d["connectivity"]
    check(c["level"] in ("none", "low", "medium", "high"), "bad connectivity level")
    check(c["largestComponent"] <= d["teamCount"], "largest component exceeds team count")
    avg = c["avgGamesPerTeam"]
    if avg < 2:
        check(c["level"] == "none",
              f"with {avg} games per team the board must be labelled 'not yet meaningful'")

    report()


def report():
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} check(s) failed.")
        sys.exit(1)
    print(f"All checks passed ({len(warns)} warning(s)).")


if __name__ == "__main__":
    main()
