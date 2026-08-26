"""
Record another forecaster's upcoming-week picks, once per week.

Run BEFORE the games, like our own capture, and for the same reason: a
prediction recorded afterwards is not a prediction.

Fails soft, always. This is the only step in the pipeline that depends on a
site we do not control, and a board that refuses to publish because someone
else's server is down would be a bad trade. A miss costs one week of
comparison data and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rivals import (PICKS_URL, SOURCE, append_if_new, fetch,  # noqa: E402
                    match_picks, parse_picks)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=os.path.join(ROOT, "site", "ratings.json"),
                    help="our own build, for the fixture list to match against")
    ap.add_argument("--out", default=os.path.join(DATA, "rivals.jsonl"))
    ap.add_argument("--url", default=PICKS_URL)
    ap.add_argument("--delay", type=float, default=None)
    a = ap.parse_args()

    try:
        with open(a.ratings, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {a.ratings} ({exc}); nothing to match against",
              file=sys.stderr)
        return 0

    try:
        flat = fetch(url=a.url, **({"delay": a.delay} if a.delay is not None else {}))
    except Exception as exc:                      # noqa: BLE001 - deliberate
        print(f"could not reach {a.url}: {exc}. Skipping this week -- the "
              f"board publishes regardless.", file=sys.stderr)
        return 0

    week, picks = parse_picks(flat)
    if not week or not picks:
        print(f"parsed {len(picks)} picks for week {week}; the page format may "
              f"have moved. Skipping rather than recording nothing.",
              file=sys.stderr)
        return 0

    names = {i: t["name"] for i, t in enumerate(payload["teams"])}

    def side(v):
        return names[v] if isinstance(v, int) else v

    fixtures = {(side(g["h"]), side(g["a"])): g
                for g in payload.get("schedule", [])
                if g.get("w") == week and "m" in g}
    if not fixtures:
        print(f"we hold no predicted fixtures for week {week}; skipping",
              file=sys.stderr)
        return 0

    matched, report = match_picks(picks, fixtures)
    print(f"week {week}: {report['matched']}/{report['picks']} picks matched "
          f"({report['coverage']:.1%}), {report['ambiguous']} ambiguous",
          file=sys.stderr)
    if report["unmatchedSample"]:
        for pair in report["unmatchedSample"]:
            print(f"   unmatched: {pair[0]} / {pair[1]}", file=sys.stderr)

    record = {
        "source": SOURCE,
        "season": payload.get("season"),
        "week": week,
        "capturedAt": payload.get("generatedAt"),
        "report": report,
        "picks": matched,
    }
    if append_if_new(a.out, record):
        print(f"-> recorded {len(matched)} picks to {a.out}", file=sys.stderr)
    else:
        print(f"week {week} already recorded; left alone", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
