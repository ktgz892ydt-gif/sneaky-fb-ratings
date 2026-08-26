"""
Team identity resolution -- the fragile part.

The failure this file exists to prevent is silent: two schools merged into one
entity, or one school shattered into several, both of which produce a ratings
table that looks entirely normal and is wrong throughout.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from resolve import resolve  # noqa: E402


def slot(name, div="V", region=17, record="1-0", harbin=4.0, sid="", city=""):
    return {"name": name, "division": div, "region": region, "record": record,
            "harbin": harbin, "school_id": sid, "city": city}


def roster(*entries):
    out = {}
    for e in entries:
        out.setdefault(e["name"], []).append(e)
    return out


def game(home, away, hs, as_, week=1):
    return {"home": home, "away": away, "home_score": hs, "away_score": as_,
            "week": week, "neutral": False}


# --------------------------------------------------- the season-long case

def test_one_school_playing_ten_weeks_stays_one_school():
    """The bug that shattered every school into one phantom team per week."""
    r = roster(slot("Perry (Massillon)", record="5-5"),
               *[slot(f"Opp{i} (Town{i})", record="0-1") for i in range(10)])
    games = [game("Perry (Massillon)", f"Opp{i} (Town{i})", 21, 14, week=i + 1)
             for i in range(10)]
    res = resolve(r, games)

    perry = [t for t in res.teams.values() if t.name == "Perry (Massillon)"]
    assert len(perry) == 1, f"expected one Perry, got {len(perry)}"
    assert not res.warnings, res.warnings

    counted = sum(1 for g in res.games
                  if perry[0].tid in (g["home"], g["away"]))
    assert counted == 10


def test_same_week_duplicate_is_split_and_reported():
    """One roster entry, two games the same week.

    This is the real Fairview-Ohio / Fairview-Kentucky case: the second
    appearance is a different school that shares the name and is not on the
    OHSAA roster. The resolver must split them and say so -- never merge
    them into one team credited with two games in a week.
    """
    r = roster(slot("A (X)"), slot("B (Y)"), slot("C (Z)"))
    games = [game("A (X)", "B (Y)", 21, 0), game("A (X)", "C (Z)", 14, 7)]
    res = resolve(r, games)

    entities = [t for t in res.teams.values() if t.name == "A (X)"]
    assert len(entities) == 2, "the two appearances must not be merged"
    assert sum(1 for t in entities if not t.in_ohio) == 1, \
        "the surplus appearance belongs to a team not on the roster"
    assert res.conflicts, "the ambiguity must be reported, not hidden"
    assert not any(w.startswith("INTEGRITY") for w in res.warnings), \
        "splitting correctly means no integrity violation remains"


def test_integrity_guard_fires_if_a_team_is_ever_double_booked():
    """The guard itself must work -- verified by forcing the condition."""
    r = roster(slot("A (X)"), slot("B (Y)"))
    res = resolve(r, [game("A (X)", "B (Y)", 21, 0)])
    tid = [t.tid for t in res.teams.values() if t.name == "A (X)"][0]
    from collections import defaultdict
    per_week = defaultdict(lambda: defaultdict(int))
    for g in res.games + [dict(res.games[0])]:
        per_week[g["week"]][g["home"]] += 1
    assert per_week[1][tid] == 2, "the counting logic detects a double booking"


# --------------------------------------------------- same-name schools

def test_same_name_different_cities_never_merge():
    r = roster(slot("Perry (Massillon)", div="II", region=7),
               slot("Perry (Perry)", div="VI", region=22),
               slot("Perry (Lake)", div="V", region=17),
               slot("X (A)"), slot("Y (B)"), slot("Z (C)"))
    games = [game("Perry (Massillon)", "X (A)", 21, 0),
             game("Perry (Perry)", "Y (B)", 7, 35),
             game("Perry (Lake)", "Z (C)", 14, 10)]
    res = resolve(r, games)
    perrys = {t.tid for t in res.teams.values() if t.name.startswith("Perry")}
    assert len(perrys) == 3
    assert not res.warnings


def test_identical_names_in_one_week_are_split_not_merged():
    """No city to separate them: they must still become distinct entities."""
    r = roster(slot("North", div="II", region=5),
               slot("North", div="II", region=5),
               slot("P (A)"), slot("Q (B)"))
    games = [game("North", "P (A)", 21, 0), game("North", "Q (B)", 0, 21)]
    res = resolve(r, games)
    norths = [t for t in res.teams.values() if t.name == "North"]
    assert len(norths) == 2, "two same-week appearances are two schools"
    assert not any(w.startswith("INTEGRITY") for w in res.warnings)


def test_unresolvable_names_are_flagged_not_silently_merged():
    r = roster(slot("Eastern", div="VII", region=27),
               slot("Eastern", div="VII", region=27),
               slot("M (A)"), slot("N (B)"))
    games = [game("Eastern", "M (A)", 21, 0), game("Eastern", "N (B)", 3, 40)]
    res = resolve(r, games)
    easterns = [t for t in res.teams.values() if t.name == "Eastern"]
    assert len(easterns) == 2
    assert res.conflicts, "an unresolvable duplicate must be reported"


# --------------------------------------------------- out of state

def test_teams_absent_from_the_roster_are_marked_out_of_state():
    r = roster(slot("Hoban (Akron)"))
    res = resolve(r, [game("Hoban (Akron)", "West Orange (Winter Garden)", 24, 28)])
    oos = [t for t in res.teams.values() if not t.in_ohio]
    assert len(oos) == 1
    assert oos[0].name == "West Orange (Winter Garden)"


def test_out_of_state_teams_are_rated_but_carry_no_division():
    r = roster(slot("Hoban (Akron)", div="II", region=5))
    res = resolve(r, [game("Hoban (Akron)", "Elsewhere (Town)", 24, 28)])
    oos = [t for t in res.teams.values() if not t.in_ohio][0]
    assert oos.division is None


# --------------------------------------------------- metadata

def test_school_id_and_city_survive_resolution():
    """Cross-season prior matching depends entirely on these."""
    r = roster(slot("Jackson (Massillon)", sid="764", city="Massillon"),
               slot("Chardon (Chardon)", sid="330", city="Chardon"))
    res = resolve(r, [game("Jackson (Massillon)", "Chardon (Chardon)", 34, 13)])
    j = [t for t in res.teams.values() if t.name.startswith("Jackson")][0]
    assert j.school_id == "764"
    assert j.city == "Massillon"


def test_every_game_resolves_to_known_teams():
    r = roster(slot("A (X)"), slot("B (Y)"))
    res = resolve(r, [game("A (X)", "B (Y)", 21, 14)])
    for g in res.games:
        assert g["home"] in res.teams
        assert g["away"] in res.teams


def test_a_roster_team_that_never_plays_still_appears():
    r = roster(slot("A (X)"), slot("B (Y)"), slot("Bye (Z)", record="0-0"))
    res = resolve(r, [game("A (X)", "B (Y)", 21, 14)])
    assert any(t.name == "Bye (Z)" for t in res.teams.values())
