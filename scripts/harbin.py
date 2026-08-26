"""
The OHSAA playoff qualifier, as OHSAA actually computes it.

Harbin is not a strength rating and this module does not treat it as one. It is
the *rule*: the thing that decides who plays in week 11. The board's own rating
answers "who is best"; this answers "who qualified", and the two disagree often
enough that showing both is the most interesting thing on the page.

The formula
-----------
    Level 1   for each win, points set by the DEFEATED opponent's division
    Level 2   plus those same points for every win each defeated opponent has
    Harbin    (Level 1 + Level 2) / games played

Only the regular season counts -- weeks 1 through 10. Playoff results do not
feed back into the qualifier that produced the playoffs.

Where these numbers come from
-----------------------------
They were not taken from memory or from a description of the rules. They were
recovered from the source's own published Harbin column by least squares over
702 teams of a completed season, and then verified against it. The fitted
per-division win values came out

    6.54  6.10  5.47  5.02  4.48  3.94  3.40

which is a monotone ladder in steps of ~0.5, and rounding it to the obvious
clean values reproduces the published figure EXACTLY for 86% of teams that have
no out-of-state opponent anywhere in their two-level tree, with a median error
of 0.0000.

`validate()` at the bottom re-runs that check, and tests/test_harbin.py fails
the build if the agreement ever degrades. If OHSAA changes the ladder, that is
how we find out.
"""

from __future__ import annotations

from collections import defaultdict

# Points for beating a team in this division. Recovered from published data,
# not assumed -- see the module docstring.
DIVISION_POINTS = {
    "I": 6.5, "II": 6.0, "III": 5.5, "IV": 5.0, "V": 4.5, "VI": 4.0, "VII": 3.5,
}

# An out-of-state opponent has no OHSAA division. OHSAA assigns one by
# enrollment, which the scoreboard does not publish, so it cannot be read off
# the page. Least squares over the 2025 season put the effective value at 5.98
# -- indistinguishable from Division II -- so that is the stand-in, and every
# figure that leans on it is reported as approximate rather than exact.
OUT_OF_STATE_POINTS = 6.0

# The regular season. Playoff weeks do not count toward the qualifier.
LAST_REGULAR_WEEK = 10

# How many teams per region reach the playoffs. Read off the brackets rather
# than assumed: counting distinct Ohio teams appearing in week 11 and later
# gives exactly 16 per region in 2023 and 2024, and exactly 12 in all 28
# regions in 2025 -- OHSAA cut the field. Under 12 the top four seeds take a
# first-round bye, which is why only 8 teams per region play in week 11.
QUALIFIERS_PER_REGION = 12
FIRST_ROUND_BYES = 4


def division_value(team, out_of_state=OUT_OF_STATE_POINTS):
    """Points earned for beating this team."""
    return DIVISION_POINTS.get(team.division, out_of_state)


def win_tables(games, last_week=LAST_REGULAR_WEEK):
    """-> (wins, games_played), where wins[t] lists the teams t defeated.

    A tie counts as a game played and a win for nobody, which is how OHSAA
    treats it and also why `games_played` is tracked separately rather than
    derived from the win list.
    """
    wins = defaultdict(list)
    played = defaultdict(int)
    for g in games:
        if g.get("week", 1) > last_week:
            continue
        h, a = g["home"], g["away"]
        played[h] += 1
        played[a] += 1
        m = g["home_score"] - g["away_score"]
        if m > 0:
            wins[h].append(a)
        elif m < 0:
            wins[a].append(h)
    return wins, played


def harbin_points(teams, games, last_week=LAST_REGULAR_WEEK,
                  out_of_state=OUT_OF_STATE_POINTS):
    """-> {team_id: harbin average}, over the regular season only.

    `teams` maps id -> an object with a `.division`; `games` is the resolved
    game list. A team with no games scores 0.0 rather than dividing by zero.
    """
    wins, played = win_tables(games, last_week)
    value = {t: division_value(tm, out_of_state) for t, tm in teams.items()}

    # Level 1 first for everyone, because level 2 is a sum of other teams' level
    # 1 -- computing them in one pass would read a half-built table.
    level1 = {t: sum(value[o] for o in wins[t] if o in value) for t in teams}

    out = {}
    for t in teams:
        n = played[t]
        out[t] = (level1[t] + sum(level1[o] for o in wins[t] if o in level1)) / n if n else 0.0
    return out


def qualifiers(teams, harbin, per_region, region_of=None):
    """-> {team_id: seed}, the top `per_region` of each region by Harbin.

    Ohio-only: an out-of-state team on the scoreboard is not in a region and
    cannot qualify. Ties are broken by team id so a run is reproducible; OHSAA
    breaks them with a coin flip and a real tie at the cut line is rare enough
    that inventing a rule here would be more misleading than arbitrary.
    """
    by_region = defaultdict(list)
    for t, tm in teams.items():
        if not tm.in_ohio:
            continue
        r = (region_of or {}).get(t, tm.region)
        if r is None:
            continue
        by_region[r].append(t)

    seeds = {}
    for r, members in by_region.items():
        members.sort(key=lambda t: (-harbin.get(t, 0.0), t))
        for i, t in enumerate(members[:per_region], 1):
            seeds[t] = i
    return seeds


def leans_on_out_of_state(teams, games, last_week=LAST_REGULAR_WEEK):
    """-> {team_id: True} where the Harbin figure is an approximation.

    An out-of-state opponent has no OHSAA division on the scoreboard, so this
    module stands one in (see OUT_OF_STATE_POINTS). Measured against the
    source's published column over the 2025 season, that stand-in leaves a
    small positive bias -- we overstate -- and it grows with exposure:

        out-of-state teams in the two-level tree    mean error   mean |error|
                                              0       -0.017        0.017
                                              1       +0.094        0.181
                                              2       +0.125        0.310
                                              3       +0.273        0.383
                                             4+       +0.299        0.851

    Roughly one Ohio team in ten is in the last two rows. That is usually well
    inside the gap at a regional cut line and occasionally is not, so the teams
    it touches are marked rather than quietly presented as exact.
    """
    wins, _ = win_tables(games, last_week)
    flagged = {}
    for t in teams:
        touched = list(wins[t]) + [o2 for o in wins[t] for o2 in wins.get(o, [])]
        if any(not teams[x].in_ohio for x in touched if x in teams):
            flagged[t] = True
    return flagged


def validate(teams, games, published, last_week=LAST_REGULAR_WEEK):
    """Score this implementation against the source's own published column.

    Returns a report dict. `published` maps team id -> the site's Harbin value.
    Only teams with no out-of-state team anywhere in their two-level tree are
    scored strictly: everywhere else the enrollment-based division of an
    out-of-state opponent is unknown to us, so a mismatch there is a gap in the
    input, not a bug in the formula.
    """
    wins, _ = win_tables(games, last_week)
    got = harbin_points(teams, games, last_week)

    clean, exact, errors = 0, 0, []
    for t in teams:
        if t not in published or not teams[t].in_ohio:
            continue
        touched = list(wins[t]) + [o2 for o in wins[t] for o2 in wins.get(o, [])]
        if not all(teams[x].in_ohio for x in touched if x in teams):
            continue
        clean += 1
        err = got[t] - published[t]
        errors.append(err)
        if abs(err) < 0.005:
            exact += 1

    n = len(errors)
    return {
        "comparableTeams": clean,
        "exactMatches": exact,
        "exactFraction": round(exact / n, 4) if n else None,
        "maxAbsError": round(max((abs(e) for e in errors), default=0.0), 4),
        "meanAbsError": round(sum(abs(e) for e in errors) / n, 4) if n else None,
    }
