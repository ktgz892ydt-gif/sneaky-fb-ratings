"""
Turn a finished season's ratings into a preseason prior for the next one.

Why a team-level prior and not a division-level one
--------------------------------------------------
It is tempting to start every Division I team high and every Division VII team
low. Don't. That hard-codes the assumption that enrollment equals quality, and
then the season's results can only argue with it at the margins -- exactly the
distortion that makes Harbin unsuitable as a strength rating.

Measured on Week 1 2026, even a modest three-point division ladder moved
Division VI from 17 of the top 50 to zero, and Division I from 3 to 18. Early
in the season the prior does not *inform* the ranking, it *is* the ranking. So
what goes in the prior matters enormously, and it should be the thing we
actually know: how each individual team played last year.

That way Kirtland starts where Kirtland earned, and a mediocre Division I team
starts where it earned, and the enrollment of their buildings never enters into
it.

Regression to the mean
----------------------
Rosters graduate. Last year's rating is evidence about this year's team, not a
measurement of it, so it is shrunk toward league average before use. The
default keeps 50%, which is in the usual range for year-over-year carryover in
high school sport, where turnover is high.

Matching across seasons
-----------------------
Teams are matched on the school ID published on the ranking pages, which is
stable year to year. Where an ID is missing we fall back to the "School (City)"
key, which is also stable and unique. Teams with no history at all get zero --
league average -- rather than a division guess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def build_prior(prev, carry=0.5, clip=None):
    """prev: a ratings.json payload from the previous season."""
    teams = [t for t in prev.get("teams", []) if t.get("inOhio")]
    played = [t for t in teams if (t.get("games") or 0) > 0]
    if not played:
        raise SystemExit("previous season has no completed games; nothing to carry forward")

    mean = sum(t["rating"] for t in played) / len(played)

    out, skipped = {}, 0
    for t in played:
        # Only carry a team we actually measured. A team with one or two games
        # is barely evidence, and its rating is mostly last year's own prior.
        if (t.get("games") or 0) < 4:
            skipped += 1
            continue
        val = (t["rating"] - mean) * carry
        if clip:
            val = max(-clip, min(clip, val))
        key = (t.get("schoolId") or "").strip() or t["name"]
        out[key] = round(val, 3)
    return out, {
        "season": prev.get("season"),
        "carry": carry,
        "teamsCarried": len(out),
        "skippedTooFewGames": skipped,
        "sourceMean": round(mean, 3),
    }


def apply_prior(prior_map, teams):
    """Map the prior onto this season's team ids.

    `teams` is the current season's team rows (needing id, schoolId, name).
    Returns {team_id: prior_points}.
    """
    out = {}
    for t in teams:
        key = (t.get("schoolId") or "").strip()
        if key and key in prior_map:
            out[t["id"]] = prior_map[key]
        elif t["name"] in prior_map:
            out[t["id"]] = prior_map[t["name"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-ratings", required=True,
                    help="ratings.json produced for the previous season")
    ap.add_argument("--out", default=os.path.join(DATA, "prior.json"))
    ap.add_argument("--carry", type=float, default=0.5,
                    help="fraction of last year's rating to keep (default 0.5)")
    ap.add_argument("--clip", type=float, default=14.0,
                    help="cap on prior magnitude in points; 0 disables")
    args = ap.parse_args()

    with open(args.from_ratings, encoding="utf-8") as fh:
        prev = json.load(fh)

    prior, meta = build_prior(prev, args.carry, args.clip or None)
    payload = {"meta": meta, "prior": prior}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    vals = sorted(prior.values())
    print(f"carried {meta['teamsCarried']} teams from {meta['season']} "
          f"(skipped {meta['skippedTooFewGames']} with under 4 games)",
          file=sys.stderr)
    if vals:
        print(f"prior range {vals[0]:+.1f} to {vals[-1]:+.1f} pts, "
              f"median {vals[len(vals)//2]:+.1f}", file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
