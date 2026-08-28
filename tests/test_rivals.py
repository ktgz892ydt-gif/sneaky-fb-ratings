"""
Comparing this board against another public forecaster.

Two properties this file protects.

1. The comparison is PAIRED and on the INTERSECTION. Each site publishes its
   own accuracy over its own game set -- he predicted 345 games in week 1 of
   2026 where this board's scrape found 400 -- so comparing headline figures
   measures the schedules, not the models.

2. The parser reads his convention, not ours. On his page the first team named
   is the FAVOURITE and "at" means the favourite is away. On the scoreboard the
   first team named is always the visitor. Reading one as the other inverts
   the home side of every pick and still looks entirely plausible.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from rivals import (SOURCE, append_if_new, head_to_head,  # noqa: E402
                    load, match_picks, parse_picks)


# Verbatim from the real flattened page, 2026 week 2.
REAL = ("Picks for week #2 Updated Sun 23-Aug-2026 12:09 PM Week 1 complete "
        "Thursday Deer Park (1-0) by 24 (89%) at Shroder (0-1) "
        "Dunbar (1-0) by 47 (99%) at Dayton Stivers (0-1) "
        "#45 Perkins (1-0) by 49 (99%) at Scott (0-1) "
        "Friday Ada (1-0) by 27 (91%) at Arcadia (1-0) "
        "Africentric Early Coll. (0-1) by 8 (66%) vs Buckeye Valley (0-1)")


# ----------------------------------------------------------------- parsing

def test_the_week_is_read_from_the_page():
    week, _ = parse_picks(REAL)
    assert week == 2


def test_every_pick_on_the_sample_parses():
    _, picks = parse_picks(REAL)
    assert len(picks) == 5, [p["fav"] for p in picks]


def test_at_means_the_favourite_is_away():
    """His convention, not the scoreboard's. Getting this backwards would
    invert the home side of every pick."""
    _, picks = parse_picks(REAL)
    deer = picks[0]
    assert deer["fav"] == "Deer Park"
    assert deer["favHome"] is False          # "by 24 at Shroder"
    afri = picks[-1]
    assert afri["favHome"] is True           # "by 8 vs Buckeye Valley"


def test_a_day_heading_is_not_read_as_part_of_a_name():
    """Flattening the page glues 'Friday' onto the next team."""
    _, picks = parse_picks(REAL)
    names = [p["fav"] for p in picks]
    assert "Ada" in names and not any(n.startswith("Friday") for n in names)
    assert not any("Picks for week" in n for n in names)


def test_the_probability_is_read_as_a_fraction():
    _, picks = parse_picks(REAL)
    assert picks[0]["favProb"] == pytest.approx(0.89)
    assert picks[0]["margin"] == 24


# ---------------------------------------------------------------- matching

FIXTURES = {
    ("Shroder (Cincinnati)", "Deer Park (Cincinnati)"): {},
    ("Stivers (Dayton)", "Dunbar (Dayton)"): {},
    ("Scott (Toledo)", "Perkins (Sandusky)"): {},
    ("Arcadia (Arcadia)", "Ada (Ada)"): {},
    ("Africentric Early College (Columbus)", "Buckeye Valley (Delaware)"): {},
}


def test_his_bare_names_match_our_school_city_names():
    _, picks = parse_picks(REAL)
    matched, report = match_picks(picks, FIXTURES)
    assert report["ambiguous"] == 0
    assert report["matched"] == 5, report


def test_a_city_prefixed_name_matches():
    """'Dayton Stivers' is our 'Stivers (Dayton)' with the city moved."""
    _, picks = parse_picks(REAL)
    matched, _ = match_picks(picks, FIXTURES)
    got = [m for m in matched if m["home"] == "Stivers (Dayton)"]
    assert len(got) == 1


def test_an_abbreviated_name_matches():
    """'Africentric Early Coll.' for 'Africentric Early College (Columbus)'."""
    _, picks = parse_picks(REAL)
    matched, _ = match_picks(picks, FIXTURES)
    assert any(m["home"].startswith("Africentric") for m in matched)


def test_a_pick_is_restated_from_the_home_team_s_view():
    _, picks = parse_picks(REAL)
    matched, _ = match_picks(picks, FIXTURES)
    deer = [m for m in matched if m["home"] == "Shroder (Cincinnati)"][0]
    # Deer Park favoured by 24 while away, so the home margin is negative.
    assert deer["homeMargin"] == -24
    assert deer["homeProb"] == pytest.approx(0.11, abs=0.01)


def test_an_ambiguous_pick_is_dropped_not_guessed():
    """The resolver's contract, applied to someone else's data."""
    # The pick is "Perry ... at X", so Perry is the VISITOR: the fixture key is
    # (home, away) = (X, Perry). Building it the other way round is the exact
    # confusion these tests exist to catch, and it caught me writing them.
    fixtures = {("X (X)", "Perry (Massillon)"): {},
                ("X (X)", "Perry (Perry)"): {}}
    _, picks = parse_picks("Picks for week #3 Perry (1-0) by 7 (70%) at X (0-1)")
    matched, report = match_picks(picks, fixtures)
    assert matched == [] and report["ambiguous"] == 1


# ------------------------------------------------------------ head to head

def _ours(season, week, games):
    return [{"season": season, "pred": [[h, a, week, m, p] for h, a, m, p in games]}]


def _theirs(season, week, games):
    return [{"source": SOURCE, "season": season, "week": week,
             "picks": [{"home": h, "away": a, "homeMargin": m, "homeProb": p}
                       for h, a, m, p in games]}]


def test_only_games_both_predicted_are_scored():
    ours = _ours(2026, 2, [("A", "B", 7, 0.7), ("C", "D", 7, 0.7)])
    theirs = _theirs(2026, 2, [("A", "B", 7, 0.7)])       # only one in common
    res = {2026: {(2, "A", "B"): 10, (2, "C", "D"): 10}}
    h = head_to_head(theirs, ours, res)
    assert h["sharedGames"] == 1


def test_mcnemar_counts_only_the_games_they_disagreed_on():
    ours = _ours(2026, 2, [("A", "B", 7, 0.7), ("C", "D", 7, 0.7),
                           ("E", "F", 7, 0.7)])
    theirs = _theirs(2026, 2, [("A", "B", -7, 0.3),        # we right, they wrong
                               ("C", "D", 7, 0.7),         # both right
                               ("E", "F", 7, 0.7)])        # both wrong
    res = {2026: {(2, "A", "B"): 10, (2, "C", "D"): 10, (2, "E", "F"): -10}}
    h = head_to_head(theirs, ours, res)
    d = h["disagreements"]
    assert (d["weWereRight"], d["theyWereRight"]) == (1, 0)
    assert (d["bothRight"], d["bothWrong"]) == (1, 1)
    assert h["accuracyGap"] == pytest.approx(1 / 3, abs=1e-4)  # payload is rounded


def test_a_tiny_edge_over_few_games_is_called_indistinguishable():
    """The honest headline. One extra correct call is not a better model."""
    games = [(f"H{i}", f"A{i}", 7, 0.7) for i in range(40)]
    ours = _ours(2026, 2, games)
    theirs = _theirs(2026, 2, [(h, a, (-7 if i == 0 else 7), 0.7)
                               for i, (h, a, _, _) in enumerate(games)])
    res = {2026: {(2, h, a): 10 for h, a, _, _ in games}}
    h = head_to_head(theirs, ours, res)
    assert h["disagreements"]["weWereRight"] == 1
    assert h["accuracyVerdict"] == "indistinguishable", h


def test_a_consistent_edge_is_called_clear():
    games = [(f"H{i}", f"A{i}", 7, 0.7) for i in range(200)]
    ours = _ours(2026, 2, games)
    # They get a quarter of them backwards.
    theirs = _theirs(2026, 2, [(h, a, (-7 if i % 4 == 0 else 7), 0.7)
                               for i, (h, a, _, _) in enumerate(games)])
    res = {2026: {(2, h, a): 10 for h, a, _, _ in games}}
    h = head_to_head(theirs, ours, res)
    assert h["accuracyVerdict"] == "clear", h
    assert h["ours"]["accuracy"] > h["theirs"]["accuracy"]


def test_attribution_travels_with_the_numbers():
    """The source grants reuse 'provided that they credit the source', so the
    credit must not be separable from the figures."""
    ours = _ours(2026, 2, [("A", "B", 7, 0.7)])
    theirs = _theirs(2026, 2, [("A", "B", 7, 0.7)])
    h = head_to_head(theirs, ours, {2026: {(2, "A", "B"): 10}})
    assert h["sourceName"] and h["sourceUrl"]


def test_nothing_scorable_yields_nothing():
    assert head_to_head([], [], {}) is None


def test_a_week_is_recorded_once(tmp_path):
    p = str(tmp_path / "r.jsonl")
    rec = {"source": SOURCE, "season": 2026, "week": 2, "picks": []}
    assert append_if_new(p, rec) is True
    assert append_if_new(p, rec) is False
    assert len(load(p)) == 1


def test_no_verdict_beyond_indistinguishable_on_a_tiny_sample():
    """A z-test needs a sample to approximate.

    On two shared games the two per-game log-loss differences happened to be
    close, the standard error came out tiny, z was 8, and the payload reported
    "clear". That is not evidence, it is two numbers agreeing with each other.
    Accuracy was already protected by an exact binomial; this is the floor for
    the continuous measure.
    """
    games = [("A", "B", 20.0, 0.95), ("C", "D", 20.0, 0.95)]
    ours = _ours(2026, 2, games)
    theirs = _theirs(2026, 2, [(h, a, m, 0.55) for h, a, m, _ in games])
    res = {2026: {(2, "A", "B"): 10, (2, "C", "D"): 10}}
    h = head_to_head(theirs, ours, res)
    assert h["sharedGames"] == 2
    assert h["loglossVerdict"] == "indistinguishable", h
    assert h["accuracyVerdict"] == "indistinguishable", h


def test_a_real_sample_can_still_earn_a_verdict():
    games = [(f"H{i}", f"A{i}", 20.0, 0.95) for i in range(60)]
    ours = _ours(2026, 2, games)
    theirs = _theirs(2026, 2, [(h, a, m, 0.55) for h, a, m, _ in games])
    res = {2026: {(2, h, a): 10 for h, a, _, _ in games}}
    r = head_to_head(theirs, ours, res)
    assert r["sharedGames"] == 60
    assert r["loglossVerdict"] in ("leaning", "clear"), r
