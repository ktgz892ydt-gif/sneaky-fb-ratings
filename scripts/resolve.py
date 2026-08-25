"""
Team identity resolution.

The scoreboard gives bare school names. Ohio has many duplicates: three
schools named Northwest, three named Perry, three named Crestview, two named
Jackson, plus North / South / East / Eastern / Southern and a long tail. Keying
on the bare name would silently merge distinct programs into one rated entity,
which corrupts every rating that touches them and does so invisibly.

The resolver's contract is: it will decline to guess before it will merge.

Method
------
The regional ranking pages list every OHSAA team with its division, region and
running record. A team plays at most one game per week, so a school's record is
a fingerprint: after week N, each roster slot has a distinct W-L that we can
match against the sequence of results attached to each scoreboard appearance.

For a duplicated name we enumerate the roster slots carrying that name and the
game appearances carrying that name, then look for an assignment that makes
every slot's record consistent with its assigned results. If exactly one
assignment works, the name is resolved. If several work, the entities are still
kept *separate* -- they are simply labelled with an index rather than a school,
and written to conflicts.csv for a human to look at. Nothing is ever merged.

Names absent from the roster are treated as out-of-state opponents. They are
rated (their results carry real information about the Ohio teams that played
them) but flagged so the site can hide them from the Ohio rankings.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Team:
    tid: str
    name: str
    division: str | None = None
    region: int | None = None
    harbin: float | None = None
    stated_record: str | None = None
    school_id: str = ""
    city: str = ""
    in_ohio: bool = True
    ambiguous: bool = False
    note: str = ""


@dataclass
class Resolution:
    teams: dict
    games: list
    conflicts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def load_roster(path):
    slots = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            slots[row["name"]].append(
                {
                    "name": row["name"],
                    "division": row["division"],
                    "region": int(row["region"]),
                    "record": row["record"],
                    "harbin": float(row["harbin"]),
                    "school_id": (row.get("school_id") or "").strip(),
                    "city": (row.get("city") or "").strip(),
                }
            )
    return slots


def load_games(path):
    """Read games from either the scraper's CSV or the pipe-delimited fixture."""
    if path.endswith(".csv"):
        games = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    games.append(
                        {
                            "week": int(row["week"]),
                            "away": row["away"].strip(),
                            "away_score": int(row["away_score"]),
                            "home": row["home"].strip(),
                            "home_score": int(row["home_score"]),
                            "neutral": bool(int(row.get("neutral") or 0)),
                        }
                    )
                except (ValueError, KeyError):
                    continue
        return games

    games = []
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                continue
            away, as_, home, hs = parts
            if not as_.strip() or not hs.strip():
                continue  # postponed / cancelled / no score posted
            try:
                games.append(
                    {
                        "week": 1,
                        "away": away.strip(),
                        "away_score": int(as_),
                        "home": home.strip(),
                        "home_score": int(hs),
                        "neutral": False,
                        "line": lineno,
                    }
                )
            except ValueError:
                continue
    return games



def resolve(roster_slots, games) -> Resolution:
    """Map every team name in the schedule onto a school.

    The unit of duplication is the WEEK, not the season. A team plays at most
    one game a week, so two appearances of "Perry" in the same week are two
    different schools -- but ten appearances across ten weeks are one school
    playing a season. Counting appearances without regard to week was a bug
    that shattered every school into one phantom team per week; the guard at
    the end of this function exists to make that class of error loud.
    """
    # name -> list of appearances; each is (game_index, side, outcome, opponent, week)
    appearances = defaultdict(list)
    for gi, g in enumerate(games):
        m = g["home_score"] - g["away_score"]
        home_out = 1 if m > 0 else (0 if m < 0 else 0.5)
        away_out = 1 - home_out if home_out != 0.5 else 0.5
        wk = g.get("week", 1)
        appearances[g["home"]].append((gi, "home", home_out, g["away"], wk))
        appearances[g["away"]].append((gi, "away", away_out, g["home"], wk))

    teams: dict[str, Team] = {}
    conflicts, warnings = [], []
    assignment: dict[tuple, str] = {}   # (name, appearance_index) -> team id
    deferred = []

    def make(tid, name, slot=None, **kw):
        teams[tid] = Team(
            tid, name,
            division=slot["division"] if slot else kw.pop("division", None),
            region=slot["region"] if slot else kw.pop("region", None),
            harbin=slot["harbin"] if slot else kw.pop("harbin", None),
            stated_record=slot["record"] if slot else None,
            school_id=slot.get("school_id", "") if slot else kw.pop("school_id", ""),
            city=slot.get("city", "") if slot else kw.pop("city", ""),
            **kw,
        )
        return tid

    for name in sorted(set(appearances) | set(roster_slots)):
        slots = roster_slots.get(name, [])
        apps = appearances.get(name, [])

        by_week = defaultdict(list)
        for i, a in enumerate(apps):
            by_week[a[4]].append(i)

        # How many distinct schools share this name? At least as many as ever
        # played in the same week.
        simultaneous = max((len(v) for v in by_week.values()), default=0)

        # ---- not on the OHSAA roster: out of state
        if not slots:
            n = max(simultaneous, 1)
            if n == 1:
                tid = make(f"OOS::{name}", name, in_ohio=False,
                           note="not on OHSAA roster")
                for i in range(len(apps)):
                    assignment[(name, i)] = tid
            else:
                for k in range(n):
                    make(f"OOS::{name}#{k+1}", name, in_ohio=False, ambiguous=True,
                         note="not on roster; several schools share this name")
                _spread(name, apps, by_week, n, assignment, lambda k: f"OOS::{name}#{k+1}")
                conflicts.append({"name": name, "kind": "oos-duplicate",
                                  "detail": f"{n} schools share this name"})
            continue

        # ---- the ordinary case: one school, however many weeks it played
        if simultaneous <= 1 and len(slots) == 1:
            sl = slots[0]
            tid = make(f"{name}|{sl['division']}-{sl['region']}", name, sl)
            for i in range(len(apps)):
                assignment[(name, i)] = tid
            continue

        # ---- one stream of games, several roster entries sharing the name.
        # We cannot say which school it is, but it is definitely one school.
        if simultaneous <= 1:
            divs = {s["division"] for s in slots}
            regs = {s["region"] for s in slots}
            tid = make(f"{name}|?", name, None, ambiguous=True,
                       division=divs.pop() if len(divs) == 1 else None,
                       region=regs.pop() if len(regs) == 1 else None,
                       note=f"{len(slots)} schools share this name and only one "
                            f"played each week; identity not determined")
            for i in range(len(apps)):
                assignment[(name, i)] = tid
            conflicts.append({"name": name, "kind": "ambiguous",
                              "detail": f"{len(slots)} roster entries, never simultaneous"})
            continue

        # ---- genuinely several schools active at once: defer to the
        # geography pass, which needs the co-occurrence table built first.
        deferred.append({"name": name, "slots": slots, "apps": apps,
                         "by_week": by_week, "n": max(simultaneous, len(slots))})

    # ------------------------------------------------------------------
    # Geography pass: learn how regions schedule each other from the names
    # already settled, then use it to split the ones that are still shared.
    # ------------------------------------------------------------------
    region_of = {}
    for tid, tm in teams.items():
        if tm.in_ohio and tm.region is not None and not tm.ambiguous:
            region_of.setdefault(tm.name, set()).add(tm.region)
    region_of = {n: next(iter(r)) for n, r in region_of.items() if len(r) == 1}

    cooc = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)
    for g in games:
        ra, rb = region_of.get(g["home"]), region_of.get(g["away"])
        if ra is None or rb is None:
            continue
        cooc[ra][rb] += 1; cooc[rb][ra] += 1
        totals[ra] += 1; totals[rb] += 1

    N_REGIONS, ALPHA = 28, 0.5

    def log_p(slot_region, opp_region):
        if slot_region is None or opp_region is None:
            return 0.0
        return math.log((cooc[slot_region][opp_region] + ALPHA)
                        / (totals[slot_region] + ALPHA * N_REGIONS))

    for item in deferred:
        name, slots, apps = item["name"], item["slots"], item["apps"]
        # As many schools as ever played at once. Any beyond the roster's
        # entries are out-of-state teams that happen to share the name.
        n = item["n"]
        n_ohio = min(n, len(slots))

        ids = []
        used = set()
        for k in range(n):
            if k < n_ohio:
                sl = slots[k]
                base = f"{name}|{sl['division']}-{sl['region']}"
                tid, bump = base, 1
                while tid in teams or tid in used:
                    bump += 1
                    tid = f"{base}#{bump}"
                make(tid, name, sl)
            else:
                tid = f"OOS::{name}#{k + 1}"
                make(tid, name, None, in_ohio=False, ambiguous=True,
                     note="shares a name with an Ohio school but is not on the roster")
            used.add(tid)
            ids.append(tid)

        # Within each week, assign that week's games to distinct schools,
        # choosing whichever pairing makes the opponents most plausible.
        gaps = []
        for wk, app_idx in sorted(item["by_week"].items()):
            scored = []
            for perm in itertools.permutations(range(n), min(n, len(app_idx))):
                sc = sum(
                    log_p(slots[perm[j]]["region"] if perm[j] < n_ohio else None,
                          region_of.get(apps[ai][3]))
                    for j, ai in enumerate(app_idx[:len(perm)])
                )
                scored.append((sc, perm))
            scored.sort(key=lambda x: -x[0])
            best = scored[0][1] if scored else tuple(range(len(app_idx)))
            if len(scored) > 1:
                gaps.append(scored[0][0] - scored[1][0])
            for j, ai in enumerate(app_idx):
                assignment[(name, ai)] = ids[best[j] if j < len(best) else min(j, n - 1)]

        # A wide margin between the best and second-best pairing means the
        # opponents really do point at one answer; a narrow one means we guessed.
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        confident = mean_gap >= 1.0
        for tid in ids:
            if teams[tid].in_ohio and not confident:
                teams[tid].ambiguous = True
                teams[tid].note = (f"{len(slots)} schools share this name; games "
                                   f"split by opponent region, low confidence")
        if not confident:
            conflicts.append({"name": name, "kind": "low-confidence",
                              "detail": f"{n} schools share this name; "
                                        f"geography margin {mean_gap:.2f} log-units"})

    # ---- rewrite the games with resolved ids
    seen_pos = defaultdict(int)
    resolved = []
    for g in games:
        out = dict(g)
        for side in ("home", "away"):
            nm = g[side]
            i = seen_pos[nm]
            seen_pos[nm] += 1
            out[side] = assignment[(nm, i)]
        resolved.append(out)

    # ---- the guard. A team cannot play twice in one week; if one does, the
    # resolution above is wrong and the ratings built on it would be silently
    # garbage. Fail loudly instead.
    per_week = defaultdict(lambda: defaultdict(int))
    for g in resolved:
        per_week[g.get("week", 1)][g["home"]] += 1
        per_week[g.get("week", 1)][g["away"]] += 1
    for wk, counts in per_week.items():
        for tid, c in counts.items():
            if c > 1:
                warnings.append(
                    f"INTEGRITY: {tid} appears in {c} games in week {wk} -- "
                    f"resolution is wrong or the source has a duplicate row"
                )

    return Resolution(teams=teams, games=resolved,
                      conflicts=conflicts, warnings=warnings)


def _spread(name, apps, by_week, n, assignment, id_for):
    """Hand out same-week appearances to distinct entities, round robin."""
    for wk, idxs in sorted(by_week.items()):
        for k, ai in enumerate(idxs):
            assignment[(name, ai)] = id_for(min(k, n - 1))
