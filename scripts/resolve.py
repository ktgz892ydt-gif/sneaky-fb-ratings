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


def _record_tuple(rec: str):
    parts = rec.split("-")
    w = int(parts[0])
    l = int(parts[1])
    t = int(parts[2]) if len(parts) > 2 else 0
    return (w, l, t)


def resolve(roster_slots, games) -> Resolution:
    # Collect every appearance of every bare name.
    # outcome: 1 = won, 0 = lost, 0.5 = tie
    appearances = defaultdict(list)
    for gi, g in enumerate(games):
        m = g["home_score"] - g["away_score"]
        home_out = 1 if m > 0 else (0 if m < 0 else 0.5)
        away_out = 1 - home_out if home_out != 0.5 else 0.5
        appearances[g["home"]].append((gi, "home", home_out, g["away"]))
        appearances[g["away"]].append((gi, "away", away_out, g["home"]))

    teams: dict[str, Team] = {}
    conflicts = []
    warnings = []
    pending = []
    # maps (name, appearance_position) -> team id
    assignment: dict[tuple, str] = {}

    all_names = set(appearances) | set(roster_slots)

    for name in sorted(all_names):
        slots = roster_slots.get(name, [])
        apps = appearances.get(name, [])

        # ---- Case 1: name not on the OHSAA roster -> out of state
        if not slots:
            if len(apps) <= 1:
                tid = f"OOS::{name}"
                teams[tid] = Team(tid, name, in_ohio=False, note="not on OHSAA roster")
                for pos, _ in enumerate(apps):
                    assignment[(name, pos)] = tid
            else:
                # Same name, multiple games in one week -> definitely different
                # schools. Split them; never merge.
                for pos, _ in enumerate(apps):
                    tid = f"OOS::{name}#{pos + 1}"
                    teams[tid] = Team(
                        tid, name, in_ohio=False, ambiguous=True,
                        note="not on roster; multiple same-week games so split",
                    )
                    assignment[(name, pos)] = tid
                conflicts.append(
                    {"name": name, "kind": "oos-duplicate",
                     "detail": f"{len(apps)} same-week games, no roster entry"}
                )
            continue

        # ---- Case 2: unique roster slot and at most one appearance
        if len(slots) == 1 and len(apps) <= 1:
            s = slots[0]
            tid = f"{name}|{s['division']}-{s['region']}"
            teams[tid] = Team(
                tid, name, s["division"], s["region"], s["harbin"], s["record"]
            )
            for pos, _ in enumerate(apps):
                assignment[(name, pos)] = tid
            continue

        # ---- Case 3: duplicates. Try to match records to results.
        n_slots, n_apps = len(slots), len(apps)

        if n_apps > n_slots:
            warnings.append(
                f"{name}: {n_apps} games but only {n_slots} roster slots; "
                f"{n_apps - n_slots} appearance(s) treated as out-of-state"
            )

        # Candidate assignments: which appearance goes to which slot.
        # A slot's stated record must be consistent with the outcome of the
        # game assigned to it (for a single week: 1-0 means it won).
        def consistent(slot, outcome):
            w, l, t = _record_tuple(slot["record"])
            if outcome == 1:
                return w >= 1
            if outcome == 0:
                return l >= 1
            return t >= 1

        viable = []
        k = min(n_slots, n_apps)
        for slot_perm in itertools.permutations(range(n_slots), k):
            ok = all(
                consistent(slots[slot_perm[i]], apps[i][2]) for i in range(k)
            )
            if ok:
                viable.append(slot_perm)
            if len(viable) > 1:
                break  # ambiguous; no need to enumerate further

        unique = len(viable) == 1

        if unique:
            perm = viable[0]
            used = set()
            for i in range(k):
                s = slots[perm[i]]
                tid = f"{name}|{s['division']}-{s['region']}"
                # two schools can share a name AND a region (it happens);
                # disambiguate the id further in that case
                bump = 1
                base = tid
                while tid in teams and tid in used:
                    bump += 1
                    tid = f"{base}#{bump}"
                used.add(tid)
                teams[tid] = Team(
                    tid, name, s["division"], s["region"], s["harbin"], s["record"]
                )
                assignment[(name, i)] = tid
            for i in range(k, n_apps):
                tid = f"OOS::{name}#{i + 1}"
                teams[tid] = Team(tid, name, in_ohio=False, ambiguous=True,
                                  note="surplus appearance beyond roster slots")
                assignment[(name, i)] = tid
        else:
            # Record alone cannot separate them. Defer to the geography pass.
            pending.append({"name": name, "slots": slots, "apps": apps,
                            "consistent": consistent})

    # ------------------------------------------------------------------
    # Geography pass
    #
    # Records alone leave ties: if two schools called Perry both went 1-0,
    # either could own either win. But Ohio teams overwhelmingly play close to
    # home, so *who they played* is informative. We learn the region-vs-region
    # scheduling distribution from the games we already resolved, then score
    # each candidate assignment by how plausible its opponents are.
    #
    # This is a likelihood, not a certainty. Assignments that stay close are
    # reported as low-confidence rather than asserted.
    # ------------------------------------------------------------------
    region_of_name = {}
    for tid, tm in teams.items():
        if tm.in_ohio and tm.region is not None and not tm.ambiguous:
            # only names that resolved to exactly one entity
            region_of_name.setdefault(tm.name, set()).add(tm.region)
    region_of_name = {n: next(iter(r)) for n, r in region_of_name.items() if len(r) == 1}

    cooc = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)
    for g in games:
        ra = region_of_name.get(g["home"])
        rb = region_of_name.get(g["away"])
        if ra is None or rb is None:
            continue
        cooc[ra][rb] += 1.0
        cooc[rb][ra] += 1.0
        totals[ra] += 1.0
        totals[rb] += 1.0

    N_REGIONS = 28
    ALPHA = 0.5  # Laplace smoothing; keeps unseen region pairs merely unlikely

    def log_p(slot_region, opp_region):
        if slot_region is None or opp_region is None:
            return 0.0
        num = cooc[slot_region][opp_region] + ALPHA
        den = totals[slot_region] + ALPHA * N_REGIONS
        return math.log(num / den)

    for item in pending:
        name, slots, apps = item["name"], item["slots"], item["apps"]
        consistent = item["consistent"]
        n_slots, n_apps = len(slots), len(apps)
        k = min(n_slots, n_apps)

        scored = []
        for perm in itertools.permutations(range(n_slots), k):
            if not all(consistent(slots[perm[i]], apps[i][2]) for i in range(k)):
                continue
            score = 0.0
            for i in range(k):
                opp_region = region_of_name.get(apps[i][3])
                score += log_p(slots[perm[i]]["region"], opp_region)
            scored.append((score, perm))

        scored.sort(key=lambda x: -x[0])

        if not scored:
            # No record-consistent assignment at all: something is off upstream.
            best_perm = tuple(range(k))
            gap = 0.0
            confident = False
            conflicts.append({"name": name, "kind": "no-consistent-assignment",
                              "detail": f"{n_slots} slots, {n_apps} games"})
        else:
            best_perm = scored[0][1]
            gap = (scored[0][0] - scored[1][0]) if len(scored) > 1 else float("inf")
            # A gap of 1 log unit means the winner is ~2.7x more likely.
            confident = gap >= 1.0

        used = set()
        for i in range(k):
            s = slots[best_perm[i]]
            base = f"{name}|{s['division']}-{s['region']}"
            tid = base
            bump = 1
            while tid in teams or tid in used:
                bump += 1
                tid = f"{base}#{bump}"
            used.add(tid)
            teams[tid] = Team(
                tid, name, s["division"], s["region"], s["harbin"], s["record"],
                ambiguous=not confident,
                note="" if confident else
                     f"shares a name with {n_slots - 1} other school(s); "
                     f"assigned by opponent geography, low confidence",
            )
            assignment[(name, i)] = tid

        for i in range(k, n_apps):
            tid = f"OOS::{name}#{i + 1}"
            teams[tid] = Team(tid, name, in_ohio=False, ambiguous=True,
                              note="surplus appearance beyond roster slots")
            assignment[(name, i)] = tid

        if not confident:
            conflicts.append({
                "name": name,
                "kind": "low-confidence",
                "detail": f"{n_slots} schools share this name; "
                          f"geography margin {gap:.2f} log-units",
            })

    # Rewrite games to use resolved ids.
    seen_pos = defaultdict(int)
    resolved_games = []
    for gi, g in enumerate(games):
        out = dict(g)
        for side in ("home", "away"):
            nm = g[side]
            pos = seen_pos[nm]
            seen_pos[nm] += 1
            out[side] = assignment[(nm, pos)]
        resolved_games.append(out)

    # Safety net: no team may appear twice in the same week.
    per_week = defaultdict(lambda: defaultdict(int))
    for g in resolved_games:
        per_week[g["week"]][g["home"]] += 1
        per_week[g["week"]][g["away"]] += 1
    for wk, counts in per_week.items():
        for tid, c in counts.items():
            if c > 1:
                warnings.append(
                    f"INTEGRITY: {tid} appears in {c} games in week {wk} -- "
                    f"resolution is wrong or the source has a duplicate row"
                )

    return Resolution(teams=teams, games=resolved_games,
                      conflicts=conflicts, warnings=warnings)
