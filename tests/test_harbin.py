"""
The OHSAA qualifier, and the odds built on it.

The formula in scripts/harbin.py was not taken from a description of the rules.
It was recovered from the source's own published Harbin column and then checked
against it. These tests are that check, run on every build: if OHSAA changes
the ladder, or the scraper starts feeding it different games, the agreement
degrades and this file says so.
"""

import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from harbin import (DIVISION_POINTS, QUALIFIERS_PER_REGION,  # noqa: E402
                    harbin_points, qualifiers, validate, win_tables)
from resolve import load_games, load_roster, resolve  # noqa: E402
from simulate import simulate_season  # noqa: E402


class T:
    def __init__(self, division=None, region=None, in_ohio=True):
        self.division, self.region, self.in_ohio = division, region, in_ohio
        self.harbin = None


def game(home, away, hs, as_, week=1):
    return {"home": home, "away": away, "home_score": hs, "away_score": as_,
            "week": week}


# --------------------------------------------------------------- the ladder

def test_the_division_ladder_is_monotone():
    """Beating a bigger school is worth more. Recovered, not assumed."""
    v = [DIVISION_POINTS[d] for d in ("I", "II", "III", "IV", "V", "VI", "VII")]
    assert all(a > b for a, b in zip(v, v[1:])), v


def test_the_ladder_steps_evenly():
    v = [DIVISION_POINTS[d] for d in ("I", "II", "III", "IV", "V", "VI", "VII")]
    steps = {round(a - b, 6) for a, b in zip(v, v[1:])}
    assert steps == {0.5}, steps


# ------------------------------------------------------------ the arithmetic

def test_a_win_earns_the_beaten_team_s_division():
    teams = {"a": T("III"), "b": T("I")}
    h = harbin_points(teams, [game("a", "b", 21, 0)])
    assert h["a"] == pytest.approx(DIVISION_POINTS["I"])   # 1 game, level 2 = 0
    assert h["b"] == 0.0


def test_level_two_pays_for_your_opponent_s_wins():
    """Beat someone who beat someone: their level 1 becomes your level 2."""
    teams = {"a": T("V"), "b": T("V"), "c": T("I")}
    games = [game("b", "c", 10, 7, week=1), game("a", "b", 14, 0, week=2)]
    h = harbin_points(teams, games)
    # a beat b (V = 4.5); b's level 1 is a win over c (I = 6.5); a played 1 game
    assert h["a"] == pytest.approx((4.5 + 6.5) / 1)


def test_a_loss_is_worth_nothing_however_narrow():
    teams = {"a": T("I"), "b": T("I")}
    h = harbin_points(teams, [game("a", "b", 20, 21)])
    assert h["a"] == 0.0


def test_a_tie_counts_as_a_game_but_not_a_win():
    teams = {"a": T("I"), "b": T("I"), "c": T("I")}
    h = harbin_points(teams, [game("a", "b", 7, 7), game("a", "c", 20, 0, week=2)])
    _, played = win_tables([game("a", "b", 7, 7), game("a", "c", 20, 0, week=2)])
    assert played["a"] == 2
    assert h["a"] == pytest.approx(6.5 / 2)     # one win, divided by two games


def test_the_playoffs_do_not_feed_the_qualifier_that_produced_them():
    """Weeks 11+ must not count. Verified against the real record column in
    test_it_matches_the_published_column below; this pins the mechanism."""
    teams = {"a": T("I"), "b": T("I")}
    regular = harbin_points(teams, [game("a", "b", 21, 0, week=10)])
    with_playoff = harbin_points(teams, [game("a", "b", 21, 0, week=10),
                                         game("a", "b", 35, 0, week=11)])
    assert regular["a"] == with_playoff["a"]


def test_a_team_with_no_games_does_not_divide_by_zero():
    assert harbin_points({"a": T("I")}, [])["a"] == 0.0


# ------------------------------------------------- against the real published data

def _season(year):
    res = resolve(load_roster(os.path.join(ROOT, "data", f"roster_{year}.csv")),
                  load_games(os.path.join(ROOT, "data", f"games_{year}.csv")))
    pub_by_name = {r["name"]: float(r["harbin"]) for r in
                   csv.DictReader(open(os.path.join(ROOT, "data", f"roster_{year}.csv")))}
    pub = {t: pub_by_name[res.teams[t].name] for t in res.teams
           if res.teams[t].name in pub_by_name}
    return res, pub


def test_it_matches_the_published_column():
    """The load-bearing test. 86% exact when the formula was recovered."""
    res, pub = _season(2025)
    rep = validate(res.teams, res.games, pub)
    assert rep["comparableTeams"] > 400
    assert rep["exactFraction"] >= 0.80, rep
    assert rep["meanAbsError"] < 0.05, rep


@pytest.mark.parametrize("year,per,expected", [(2023, 16, 448), (2024, 16, 448),
                                               (2025, 12, 336)])
def test_it_picks_the_teams_that_actually_made_the_playoffs(year, per, expected):
    """The only test that matters to a reader: does the rule name the real field?

    Scoring our Harbin over the regular season alone against the teams that
    actually appeared in week 11 and later. The published Harbin itself only
    reaches 99.3-100%, so this is close to the ceiling.
    """
    res, _ = _season(year)
    actual = {g[s] for g in res.games if g.get("week", 1) > 10 for s in ("home", "away")
              if g[s] in res.teams and res.teams[g[s]].in_ohio}
    ours = set(qualifiers(res.teams, harbin_points(res.teams, res.games), per))
    assert len(ours) == expected, f"{len(ours)} qualifiers, expected {expected}"
    assert len(ours & actual) / len(actual) >= 0.98


# --------------------------------------------------------------- the odds

def _toy():
    """Two regions of four, everyone plays everyone, nothing played yet."""
    teams = {f"t{i}": T("III", region=1 + i // 4) for i in range(8)}
    rem, probs = [], []
    for r in (1, 2):
        mem = [t for t in teams if teams[t].region == r]
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                rem.append((mem[i], mem[j])); probs.append(0.5)
    return teams, rem, probs


def test_the_odds_across_a_region_sum_to_the_places_available():
    """The conservation law. Exactly `per_region` qualify in every simulated
    season, so the odds must add to that -- whatever the simulation does."""
    teams, rem, probs = _toy()
    sim = simulate_season(sorted(teams), teams, [], rem, probs, per_region=2,
                          n_sims=2000)
    for r in (1, 2):
        tot = sum(v["playoffOdds"] for t, v in sim.items() if teams[t].region == r)
        assert tot == pytest.approx(2.0, abs=0.02), (r, tot)


def test_evenly_matched_teams_get_even_odds():
    teams, rem, probs = _toy()
    sim = simulate_season(sorted(teams), teams, [], rem, probs, per_region=2,
                          n_sims=4000)
    odds = [v["playoffOdds"] for v in sim.values()]
    assert max(odds) - min(odds) < 0.12, odds


def test_a_certain_winner_is_certain_to_qualify():
    teams, rem, _ = _toy()
    probs = [1.0 if rem[k][0] == "t0" else (0.0 if rem[k][1] == "t0" else 0.5)
             for k in range(len(rem))]
    sim = simulate_season(sorted(teams), teams, [], rem, probs, per_region=2,
                          n_sims=2000)
    assert sim["t0"]["playoffOdds"] > 0.99


def test_the_simulation_is_reproducible():
    """A fixed seed, because the deterministic-build check depends on it."""
    teams, rem, probs = _toy()
    a = simulate_season(sorted(teams), teams, [], rem, probs, 2, n_sims=500)
    b = simulate_season(sorted(teams), teams, [], rem, probs, 2, n_sims=500)
    assert a == b


def test_a_win_distribution_is_a_distribution():
    teams, rem, probs = _toy()
    sim = simulate_season(sorted(teams), teams, [], rem, probs, 2, n_sims=2000)
    for t, v in sim.items():
        assert sum(v["winDist"].values()) == pytest.approx(1.0, abs=0.02)
        assert all(0 <= k <= 16 for k in v["winDist"])


def test_out_of_state_teams_cannot_qualify():
    teams = {"oh": T("III", region=1), "oos": T(None, None, in_ohio=False)}
    sim = simulate_season(sorted(teams), teams, [game("oh", "oos", 21, 0)], [], [],
                          per_region=QUALIFIERS_PER_REGION, n_sims=100)
    assert "oos" not in sim and "oh" in sim
