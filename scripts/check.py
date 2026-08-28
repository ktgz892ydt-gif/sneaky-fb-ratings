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

from build import IMPLAUSIBLE_SCORES  # noqa: E402
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


# ---- date-based completeness -------------------------------------------
#
# These are pure so tests can drive them directly; main() only reports.
#
# The question they answer is one a week number cannot: has a game been played
# yet, or was it played and missed? Both look identical in a fixture list. With
# the kickoff date on every record, a fixture whose date has passed is a game
# the board should already hold a score for, and does not.


def _name(payload, ref):
    """A schedule row points at a team by index, or carries a bare name when
    the team could not be resolved. Both have to render in a warning."""
    if isinstance(ref, int):
        teams = payload.get("teams") or []
        return teams[ref]["name"] if 0 <= ref < len(teams) else f"#{ref}"
    return str(ref)


def overdue_fixtures(schedule, today):
    """Fixtures whose kickoff date has passed but which still have no score.

    `today` is the build's own generated-at date, not the wall clock, so the
    result is a property of the build and a pinned rebuild gives the same
    answer.
    """
    return [g for g in schedule if g.get("d") and g["d"] < today]


def week_coverage(schedule, results, today):
    """-> {week: (overdue, captured)} for every week with an overdue fixture.

    A handful of overdue games is ordinary -- a postponement, or the source
    posting a score late. HALF A WEEK is not: nobody postpones 180 games, so
    that shape means the week's scoreboard did not parse.
    """
    out = {}
    for g in overdue_fixtures(schedule, today):
        o, c = out.get(g["w"], (0, 0))
        out[g["w"]] = (o + 1, c)
    for r in results or []:
        if r["w"] in out:
            o, c = out[r["w"]]
            out[r["w"]] = (o, c + 1)
    return out


def week_date_spans(results):
    """-> {week: (earliest, latest)} over results that carry a date."""
    spans = {}
    for r in results or []:
        d = r.get("d")
        if not d:
            continue
        lo, hi = spans.get(r["w"], (d, d))
        spans[r["w"]] = (min(lo, d), max(hi, d))
    return spans


def weeks_out_of_order(spans):
    """Weeks whose dates overlap the following week's.

    Week numbering is the spine of the whole model -- the prior, the
    walk-forward tuning and the track record are all keyed on it. A game filed
    under the wrong week is invisible in a fixture list and shifts a rating.
    """
    bad = []
    for w in sorted(spans):
        if w + 1 in spans and spans[w][1] >= spans[w + 1][0]:
            bad.append((w, spans[w], w + 1, spans[w + 1]))
    return bad


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

    # ---- the published page must carry everything a consumer needs to turn a
    # rating difference into a win probability. The simulator reads this; if it
    # silently vanishes, every projection quietly falls back to a wrong scale.
    cps = (d.get("config") or {}).get("probScale")
    check(isinstance(cps, dict) and all(k in cps for k in ("a", "b", "flatScale")),
          "ratings.json config is missing the probScale block")

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
            if ver >= 4:
                sel = tb.get("selection") or {}
                check("standardErrorPaired" in sel,
                      "a schema 4 tuned.json must report the paired standard "
                      "error the selection rule actually used")
                # The paired SE removes the game noise every configuration
                # shares, so it must come out smaller than the marginal one.
                # If it doesn't, the pairing is not lining up the same games.
                sp, sm = sel.get("standardErrorPaired"), sel.get("standardError")
                if sp is not None and sm is not None:
                    check(sp <= sm,
                          f"the paired standard error ({sp}) is not smaller "
                          f"than the marginal one ({sm}) -- the per-game "
                          f"vectors are not aligned across configurations")
            # Gated on the block being *present*, not on the schema number: a
            # `tune.py --calibrate-only` pass adds probScale to a tuned.json of
            # any vintage without claiming the current schema, and the curve
            # still has to be sane wherever it came from.
            ps = tb.get("probScale")
            if ver >= 3:
                check(isinstance(ps, dict) and "a" in ps and "b" in ps,
                      "a schema 3 tuned.json must carry a probScale block")
            if isinstance(ps, dict) and "a" in ps and "b" in ps:
                # The curve must steepen as teams play, and stay inside sane
                # bounds at both ends of a season. One that runs backwards
                # means the fit found noise, not information.
                def _s(g):
                    return math.sqrt(max(float(ps["a"]), 0.1)
                                     + max(float(ps["b"]), 0.0) / g)
                early, late = _s(1), _s(10)
                check(late < early,
                      f"the probability curve does not steepen through the "
                      f"season ({early:.2f} at 1 game, {late:.2f} at 10) -- "
                      f"the fit is backwards")
                warn(3.0 <= late <= 20.0 and 3.0 <= early <= 20.0,
                     f"probability scale leaves the sane range "
                     f"({early:.2f} at 1 game, {late:.2f} at 10)")
                cv = ps.get("crossValidated") or {}
                if cv.get("meanLoglossFitted") and cv.get("meanLoglossFlat"):
                    warn(cv["meanLoglossFitted"] <= cv["meanLoglossFlat"],
                         f"the fitted probability curve is worse out of sample "
                         f"than a flat scale ({cv['meanLoglossFitted']:.4f} vs "
                         f"{cv['meanLoglossFlat']:.4f}) -- do not ship it")

    # ---- the two scales must not have been swapped
    #
    # predictedHomeMargin is the rating difference times marginScale, while the
    # probability is computed from the difference BEFORE that multiplication.
    # Dividing the scale back out of the published margin must therefore
    # reproduce the published probability. If someone ever feeds the calibrated
    # margin into the probability -- the obvious mistake, since both are
    # "the margin" -- every probability shifts and nothing else looks wrong.
    cfgb = d.get("config") or {}
    mscale = cfgb.get("marginScale")
    cps2 = cfgb.get("probScale") or {}
    if mscale and cps2.get("a") is not None:
        def _scale_for(g, standin):
            if standin:
                return cps2["max"]
            if g < 1:
                return cps2["flatScale"]
            return min(max(math.sqrt(cps2["a"] + cps2["b"] / g),
                           cps2["min"]), cps2["max"])

        teams_by_pos = d["teams"]

        def _games(v):
            return teams_by_pos[v]["games"] if isinstance(v, int) else 0

        mismatch, worst = 0, 0.0
        for g in d.get("schedule") or []:
            if "m" not in g:
                continue
            gb = min(_games(g["h"]), _games(g["a"]))
            raw = g["m"] / mscale
            p_expected = 1.0 / (1.0 + math.exp(-raw / _scale_for(gb, bool(g.get("e")))))
            err = abs(p_expected - g["p"])
            worst = max(worst, err)
            if err > 0.01:
                mismatch += 1
        check(mismatch == 0,
              f"{mismatch} fixtures where the published probability cannot be "
              f"reproduced from the published margin divided by marginScale "
              f"(worst {worst:.4f}) -- the calibrated margin has most likely "
              f"been fed into the probability")
        check(0.5 <= mscale <= 3.0,
              f"marginScale {mscale} is outside any plausible range")

    # ---- the remaining schedule
    #
    # Predictions are a separate list from results on purpose. The failure that
    # matters is leakage in either direction: a fixture counted as a result
    # would be a phantom 0-0 tie, and a result still listed as a fixture would
    # show a team two opponents in one week.
    sched = d.get("schedule") or []
    check(len(sched) == d.get("scheduleGameCount", len(sched)),
          "scheduleGameCount disagrees with the schedule it describes")

    n_teams = len(d["teams"])
    played_pairs = set()
    for r in d.get("results") or []:
        played_pairs.add((r["w"], r["h"], r["a"]))

    per_week = defaultdict(lambda: defaultdict(int))
    for r in d.get("results") or []:
        per_week[r["w"]][r["h"]] += 1
        per_week[r["w"]][r["a"]] += 1

    bad_index = est_without_basis = 0
    for g in sched:
        for side in ("h", "a"):
            v = g[side]
            if isinstance(v, int):
                if not (0 <= v < n_teams):
                    bad_index += 1
                else:
                    per_week[g["w"]][v] += 1
            elif not isinstance(v, str):
                bad_index += 1

        if "m" in g:
            check(0.0 < g["p"] < 1.0,
                  f"win probability out of range in week {g['w']}: {g['p']}")
            # A probability that disagrees with its own margin means the two
            # were computed from different numbers.
            check((g["p"] > 0.5) == (g["m"] > 0) or abs(g["m"]) < 1e-9,
                  f"week {g['w']}: probability {g['p']} contradicts margin {g['m']}")
            check("ph" in g and "pa" in g,
                  f"week {g['w']}: predicted fixture has no projected score")
            if "ph" in g and "pa" in g:
                check(isinstance(g["ph"], int) and isinstance(g["pa"], int),
                      f"week {g['w']}: projected score must be integer points")
                check(g["ph"] >= 0 and g["pa"] >= 0,
                      f"week {g['w']}: projected score cannot be negative "
                      f"({g['ph']}-{g['pa']})")
                check(abs((g["ph"] - g["pa"]) - g["m"]) <= 1.1,
                      f"week {g['w']}: projected score {g['ph']}-{g['pa']} "
                      f"does not match margin {g['m']}")
                # A projected score has to be one football can produce.
                # Measured over 37,240 real team-scores, these five occur in
                # under 0.4% of games, and 4 in under 0.03%. Publishing
                # "48-1" is not a bold call, it is an impossible one, and it
                # reads as a bug to exactly the audience this board is for.
                check(g["ph"] not in IMPLAUSIBLE_SCORES
                      and g["pa"] not in IMPLAUSIBLE_SCORES,
                      f"week {g['w']}: projected score {g['ph']}-{g['pa']} "
                      f"contains a total football does not produce")
                # A stated favourite must not be shown level or losing.
                if abs(g["m"]) >= 0.5:
                    # Explicit both ways: `(ph > pa) == (m > 0)` passes a TIE
                    # when the margin is negative, which is how 22-22 shipped
                    # against a -0.5 favourite.
                    ok = (g["ph"] > g["pa"]) if g["m"] > 0 else (g["ph"] < g["pa"])
                    check(ok,
                          f"week {g['w']}: margin {g['m']:+.1f} names a "
                          f"favourite but the projected score is "
                          f"{g['ph']}-{g['pa']}")
            if g.get("e"):
                est_without_basis += 0 if d.get("fallbackRating") else 1
        else:
            check(bool(g.get("x")),
                  f"week {g['w']}: a fixture with no prediction must say why")
            check("ph" not in g and "pa" not in g,
                  f"week {g['w']}: unpredicted fixture carries a projected score")

        # Source quality, not a code bug: warn loudly, but do not stop the
        # week's ratings from publishing over it.
        warn((g["w"], g["h"], g["a"]) not in played_pairs,
             f"week {g['w']}: a game already played is still listed as a fixture")

    check(bad_index == 0,
          f"{bad_index} schedule entries point at a team that does not exist")
    check(est_without_basis == 0,
          "a prediction is flagged as estimated but no stand-in rating is published")

    # A team plays at most once a week. This is the same invariant the resolver
    # enforces on results, extended across the fixtures -- a duplicated fixture
    # would quietly inflate every projected record built on it.
    #
    # A warning rather than a failure: it distorts projections, which are the
    # newest and least load-bearing part of the page, and the ratings beneath
    # them are unaffected. Blocking the week's publish over it would trade a
    # working board for a broken one.
    doubled = [(wk, t) for wk, counts in per_week.items()
               for t, c in counts.items() if c > 1]
    warn(not doubled,
         f"{len(doubled)} team-weeks hold more than one game "
         f"(first: team index {doubled[0][1]} in week {doubled[0][0]}) -- "
         f"projected records for those teams will be overstated"
         if doubled else "")

    # Projections must be arithmetic on the games actually listed.
    for t in d["teams"]:
        if "remaining" not in t:
            continue
        total = t["projWins"] + t["projLosses"]
        expect = t["w"] + t["l"] + t["remaining"]
        check(abs(total - expect) < 0.02,
              f"{t['name']}: projected record adds to {total:.1f} games but the "
              f"team has {t['w']}-{t['l']} decided with {t['remaining']} left")
        check(t["projWins"] >= t["w"] - 0.05,
              f"{t['name']}: projected wins {t['projWins']} is below the "
              f"{t['w']} already banked")

    # ---- a season has to be a possible length.
    #
    # Ten regular-season games plus at most five playoff rounds. Anything past
    # that means fixtures belonging to more than one school have landed on this
    # team. The arithmetic check above cannot see it -- an inflated schedule
    # adds up perfectly well, and Salem was published at 13.3-5.7 over
    # nineteen games with every other assertion here passing. The per-week
    # duplicate warning can't see it either when the two schools never share a
    # week. This is the check that catches it.
    MAX_SEASON = 16
    for t in d["teams"]:
        if "remaining" not in t:
            continue
        total = t["w"] + t["l"] + t["t"] + t["remaining"]
        check(total <= MAX_SEASON,
              f"{t['name']}: {total} games this season "
              f"({t['w']}-{t['l']}-{t['t']} played, {t['remaining']} scheduled) "
              f"is more than the {MAX_SEASON} any team can play -- fixtures "
              f"from another school are landing on this one")

    # ---- every OHSAA team must have a season to play
    #
    # The mirror of the too-long check above, and the one that would have
    # caught a whole class of quiet scraper failures. Ohio plays a ten-game
    # regular season; a team the parser cannot read simply has no games, and
    # nothing else here notices -- the ratings table renders, the ranks are
    # dense, the arithmetic all agrees. Two schools sat at zero games for a
    # full season because their names ran one character past a length limit.
    #
    # Zero is a hard failure: there is no innocent reading of it.
    sched_by_team = defaultdict(int)
    for r in d.get("results") or []:
        sched_by_team[r["h"]] += 1
        sched_by_team[r["a"]] += 1
    for g in sched:
        for side in ("h", "a"):
            if isinstance(g[side], int):
                sched_by_team[g[side]] += 1

    empty = [t["name"] for i, t in enumerate(d["teams"])
             if t["inOhio"] and sched_by_team[i] == 0]
    check(not empty,
          f"{len(empty)} OHSAA team(s) have no game this season, played or "
          f"scheduled -- the scraper is dropping them: {empty[:6]}")

    # A short season is a warning, not a failure: a genuine bye or a fixture
    # the source has not posted yet both look like this, and neither should
    # stop the week's ratings publishing. A *rising* count is the signal.
    #
    # Measured on the 2026 board after the parser fixes: 672 of 699 Ohio teams
    # hold a full ten, 25 hold nine (verified against the source pages as real
    # byes), and two are genuinely thin -- East Technical at eight and
    # Jefferson Township at four. So the honest baseline here is 2, and the
    # threshold is set well above it rather than snugly, because byes move
    # year to year. Before the fixes this number was 59.
    short = [t["name"] for i, t in enumerate(d["teams"])
             if t["inOhio"] and 0 < sched_by_team[i] < 9]
    warn(len(short) <= 25,
         f"{len(short)} OHSAA teams have fewer than 9 games this season -- "
         f"if that number is climbing, the scoreboard format has moved "
         f"(first few: {short[:5]})")

    # ---- every game that should have been played has been captured
    #
    # The check that the date column exists for. Before it, a game the parser
    # dropped and a game not yet played were the same thing: a row in the
    # fixture list. Now a fixture carrying a date that has already passed is,
    # by definition, a score the board should be holding and is not.
    #
    # `today` is the build's own timestamp rather than the wall clock, so a
    # pinned rebuild reproduces the same verdict.
    today = (d.get("generatedAt") or "")[:10]
    results = d.get("results") or []
    if today:
        late = overdue_fixtures(sched, today)
        # A warning, not a failure. A postponement and a score the source has
        # not posted yet both look exactly like this, and neither should stop a
        # good week's ratings from publishing -- same trade as the rest of the
        # schedule checks. Measured on the 2026 board mid-week 2 the honest
        # baseline was 1 (Dunbar at Stivers, played Thursday, unposted Friday).
        # The list is printed rather than just the count so the games can
        # actually be looked up on the source.
        shown = [f"w{g['w']} {g['d']} "
                 f"{_name(d, g['a'])} at {_name(d, g['h'])}" for g in late[:5]]
        warn(not late,
             f"{len(late)} fixture(s) are past their kickoff date with no "
             f"score -- either the source has not posted them or the parser "
             f"missed them: {shown}")

        # A whole week is different in kind, and it does fail. Postponements
        # do not come in hundreds; a week that is mostly still 'unplayed' days
        # after its date means the scoreboard for that week did not parse, and
        # every rating built on the weeks around it is standing on a hole.
        for wk, (missing, got) in sorted(week_coverage(sched, results, today).items()):
            total = missing + got
            check(missing < 20 or missing <= total * 0.5,
                  f"week {wk}: {missing} of {total} games are past their date "
                  f"with no score, and only {got} were captured -- that is not "
                  f"late posting, that week's scoreboard did not parse")

    # Dates only arrived with parser version 2; a season scraped before that
    # carries none, so this reports coverage rather than demanding it.
    dated = sum(1 for r in results if r.get("d"))
    if dated:
        warn(dated >= len(results) * 0.95,
             f"only {dated} of {len(results)} results carry a kickoff date -- "
             f"expected nearly all of them once the season is re-scraped")
        # Week numbering is the spine of the model: the prior, the
        # walk-forward tuning and the track record are all keyed on it. If
        # week N's games run into week N+1's dates, something is filed wrong.
        # A postponement replayed later can do this innocently, so it warns.
        overlap = weeks_out_of_order(week_date_spans(results))
        warn(not overlap,
             f"{len(overlap)} week(s) hold games dated into the following "
             f"week: {overlap[:3]}")

    # ---- the playoff model
    #
    # These are conservation laws, and they are the strongest check available
    # on a Monte Carlo: however the simulation behaves, exactly N teams per
    # region qualify in EVERY simulated season, so the odds across a region
    # must sum to N. If they do not, the ranking inside the simulation is
    # wrong, and no amount of plausible-looking percentages would show it.
    po = d.get("playoffs") or {}
    if po:
        per = po["qualifiersPerRegion"]
        byes = po["firstRoundByes"]
        regions = defaultdict(list)
        for t in d["teams"]:
            if t["inOhio"] and t.get("playoffOdds") is not None and t["region"]:
                regions[t["region"]].append(t)
        check(len(regions) == 28,
              f"{len(regions)} regions carry playoff odds, expected 28")
        for r, ts in sorted(regions.items()):
            tot = sum(x["playoffOdds"] for x in ts)
            check(abs(tot - per) < 0.02,
                  f"region {r}: playoff odds sum to {tot:.2f}, but exactly "
                  f"{per} teams qualify from every simulated season")
            tb = sum(x["byeOdds"] for x in ts)
            check(abs(tb - byes) < 0.02,
                  f"region {r}: first-round-bye odds sum to {tb:.2f}, expected {byes}")
            t1 = sum(x["topSeedOdds"] for x in ts)
            check(abs(t1 - 1.0) < 0.02,
                  f"region {r}: top-seed odds sum to {t1:.2f}, expected exactly 1")

        for t in d["teams"]:
            if t.get("playoffOdds") is None:
                continue
            for k in ("playoffOdds", "byeOdds", "topSeedOdds"):
                check(0.0 <= t[k] <= 1.0, f"{t['name']}.{k} = {t[k]} is not a probability")
            # A bye is a subset of qualifying, and the top seed a subset of a bye.
            check(t["byeOdds"] <= t["playoffOdds"] + 1e-6,
                  f"{t['name']}: bye odds {t['byeOdds']} exceed playoff odds {t['playoffOdds']}")
            check(t["topSeedOdds"] <= t["byeOdds"] + 1e-6,
                  f"{t['name']}: top-seed odds exceed bye odds")
            wd = t.get("winDist") or {}
            if wd:
                sm = sum(wd.values())
                check(abs(sm - 1.0) < 0.02,
                      f"{t['name']}: win distribution sums to {sm:.3f}, not 1")

        # ---- what-ifs are conditionals, so they obey the law of total
        # probability. P(qualify) is a weighted average of P(qualify | win) and
        # P(qualify | lose), which means the unconditional figure must lie
        # BETWEEN them. This is the check that catches the mistake that matters:
        # inverting a road game, so the odds shown for winning are really the
        # odds for losing. Nothing about such a payload looks wrong otherwise.
        inverted = between = 0
        for t in d["teams"]:
            odds = t.get("playoffOdds")   # not `po`: that is the playoffs block
            check(odds is not None or not t.get("whatIf"),
                  f"{t['name']} carries what-if odds but no playoff odds")
            if odds is None:
                continue
            for r in (t.get("whatIf") or []):
                lo, hi = min(r["win"], r["lose"]), max(r["win"], r["lose"])
                if not (lo - 0.02 <= odds <= hi + 0.02):
                    inverted += 1
                between += 1
                check(0.0 <= r["win"] <= 1.0 and 0.0 <= r["lose"] <= 1.0,
                      f"{t['name']} week {r['w']}: what-if odds out of range")
            for r in (t.get("watch") or []):
                check(r["for"] in ("home", "away"),
                      f"{t['name']}: watch entry roots for {r['for']!r}")
                check(0.0 <= r["sw"] <= 1.0,
                      f"{t['name']}: watch swing {r['sw']} is not a probability")
        check(inverted == 0,
              f"{inverted} of {between} what-if pairs do not bracket the team's "
              f"own playoff odds -- a conditional is inverted, most likely a "
              f"home/away mix-up")
        warn(between > 0 or not any(t.get("remaining") for t in d["teams"]),
             "no what-if numbers were published even though games remain")

        # A team must never be told to watch a game it is playing in -- that is
        # not scoreboard watching, and it would silently double-count.
        pos = {t["id"]: i for i, t in enumerate(d["teams"])}
        for t in d["teams"]:
            me = pos[t["id"]]
            for r in (t.get("watch") or []):
                check(me not in (r["h"], r["a"]),
                      f"{t['name']} is told to watch a game it plays in "
                      f"(week {r['w']})")

        # The published Harbin implementation must still agree with the source.
        ag = po.get("harbinAgreement") or {}
        if ag.get("exactFraction") is not None:
            warn(ag["exactFraction"] >= 0.80,
                 f"our Harbin matches the source's published column exactly for "
                 f"only {ag['exactFraction']:.1%} of comparable teams "
                 f"(was 86% when the formula was recovered) -- OHSAA may have "
                 f"changed the ladder")

    # ---- the track record
    #
    # The one thing that must never happen here is a backtest being presented
    # as a live result. A replayed week was produced by constants fitted on
    # that very season; a live week was captured before the games. Pooling them
    # launders the weaker number into the stronger one, and the resulting
    # headline would look entirely reasonable.
    sc = d.get("scorecard")
    if sc:
        check(isinstance(sc.get("overall"), dict),
              "scorecard.overall must be split by kind, not a single figure")
        for kind, v in (sc.get("overall") or {}).items():
            check(kind in ("live", "backtest"),
                  f"scorecard reports an unknown kind {kind!r}")
            if not v:
                continue
            check(v["games"] > 0, f"scorecard[{kind}] claims a record over no games")
            check(0.0 <= v["accuracy"] <= 1.0,
                  f"scorecard[{kind}] accuracy {v['accuracy']} is not a fraction")
            check(0.0 <= v["brier"] <= 1.0,
                  f"scorecard[{kind}] brier {v['brier']} out of range")
            # A forecast worse than a coin flip would mean the sign is inverted
            # somewhere -- far more likely than a genuinely anti-predictive model.
            warn(v["accuracy"] >= 0.55,
                 f"scorecard[{kind}] calls only {v['accuracy']:.1%} of games "
                 f"correctly over {v['games']} games -- check the sign")
            warn(v["brier"] <= 0.25,
                 f"scorecard[{kind}] Brier {v['brier']:.3f} is no better than "
                 f"always saying 50%")
        # Per-season figures must belong to a kind, never sit loose at the top.
        for kind, seasons in (sc.get("bySeason") or {}).items():
            check(kind in ("live", "backtest"),
                  f"scorecard.bySeason has an unknown kind {kind!r}")
            for yr, v in seasons.items():
                check(str(yr).isdigit(), f"scorecard season key {yr!r} is not a year")
                check(v["games"] > 0, f"scorecard {kind} {yr} covers no games")
        for k, v in (sc.get("calibration") or {}).items():
            check(0.0 <= v["actual"] <= 1.0,
                  f"calibration bin {k}: actual {v['actual']} is not a fraction")
            check(v["n"] > 0, f"calibration bin {k} has no games")

    # ---- the head-to-head comparison
    #
    # Two things must hold. Attribution cannot be separated from the numbers:
    # the source grants reuse "provided that they credit the source", so a
    # payload carrying his figures without his name is a licence problem, not a
    # cosmetic one. And a difference must not be reported as evidence unless it
    # actually is -- over a few hundred games a couple of points of accuracy is
    # noise, and saying otherwise is the whole trap this feature could fall into.
    h2h = d.get("headToHead")
    if h2h:
        check(bool(h2h.get("sourceName")) and bool(h2h.get("sourceUrl")),
              "headToHead carries figures from another site with no attribution")
        check(h2h["sharedGames"] > 0,
              "headToHead reports a comparison over no shared games")
        for who in ("ours", "theirs"):
            v = h2h[who]
            check(0.0 <= v["accuracy"] <= 1.0,
                  f"headToHead.{who}.accuracy {v['accuracy']} is not a fraction")
            check(v["logloss"] >= 0, f"headToHead.{who}.logloss is negative")
        dis = h2h["disagreements"]
        total = sum(dis.values())
        check(total == h2h["sharedGames"],
              f"disagreement counts sum to {total} but {h2h['sharedGames']} "
              f"games were shared -- every game must fall in exactly one bucket")
        # The gap must be the discordant pairs over the shared games.
        implied = (dis["weWereRight"] - dis["theyWereRight"]) / h2h["sharedGames"]
        check(abs(implied - h2h["accuracyGap"]) < 0.001,
              f"accuracyGap {h2h['accuracyGap']} does not match the disagreement "
              f"counts (implies {implied:.4f})")
        for k in ("accuracyVerdict", "loglossVerdict"):
            check(h2h[k] in ("indistinguishable", "leaning", "clear"),
                  f"headToHead.{k} is {h2h[k]!r}")
        check(0.0 <= h2h["accuracyPValue"] <= 1.0,
              f"headToHead p-value {h2h['accuracyPValue']} out of range")
        # A tiny sample can never be reported as a clear win.
        if dis["weWereRight"] + dis["theyWereRight"] < 10:
            check(h2h["accuracyVerdict"] != "clear",
                  f"only {dis['weWereRight'] + dis['theyWereRight']} games were "
                  f"disagreed about, which cannot support a 'clear' verdict")

    # Trend series must be internally consistent -- three parallel arrays that
    # disagree in length would silently plot a team's odds against the wrong week.
    for t in d["teams"]:
        tr = t.get("trend")
        if not tr:
            continue
        n = len(tr["w"])
        check(len(tr["rating"]) == n and len(tr["odds"]) == n,
              f"{t['name']}: trend arrays disagree in length "
              f"({n} weeks, {len(tr['rating'])} ratings, {len(tr['odds'])} odds)")
        check(tr["w"] == sorted(tr["w"]),
              f"{t['name']}: trend weeks are not in order")

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
