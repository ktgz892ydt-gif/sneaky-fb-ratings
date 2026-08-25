"""
Team identity across state lines.

Ohio has a Salem; so does New Jersey. The scoreboard writes both as
"Salem (Salem)" and separates them only by a trailing state tag. The scraper
was splitting that tag off into its own column and keying teams on the bare
name, which merged the two schools into one entity -- and the Ohio team
inherited New Jersey's entire remaining schedule.

Nothing about that failure looks wrong on the page. The team simply appears
to be playing eighteen games. Seven Ohio schools were affected before this
was caught: Salem, Middletown, Bellevue, Bluffton, Celina, Lancaster and
Marietta.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from build import load_schedule  # noqa: E402
from resolve import load_games, team_identity  # noqa: E402


def test_an_ohio_team_keeps_its_bare_name():
    """Ohio teams carry no tag, so the roster join must be unaffected."""
    assert team_identity("Salem (Salem)", "") == "Salem (Salem)"
    assert team_identity("Salem (Salem)", None) == "Salem (Salem)"


def test_an_out_of_state_team_carries_its_state():
    assert team_identity("Salem (Salem)", "NJ") == "Salem (Salem) [NJ]"


def test_the_two_salems_are_different_teams():
    assert team_identity("Salem (Salem)", "") != team_identity("Salem (Salem)", "NJ")


def test_two_out_of_state_namesakes_stay_apart():
    assert team_identity("Lincoln (Lincoln)", "NE") != team_identity("Lincoln (Lincoln)", "IL")


def test_the_tag_is_normalised():
    assert team_identity("X (Y)", "nj") == "X (Y) [NJ]"
    assert team_identity("  X (Y)  ", " nj ") == "X (Y) [NJ]"


# ------------------------------------------------------------- through the loaders

def _write(tmp_path, name, header, rows):
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return str(p)


GAME_COLS = ["week", "away", "away_score", "home", "home_score",
             "neutral", "away_state", "home_state"]
SCHED_COLS = ["week", "date", "time", "away", "home",
              "neutral", "away_state", "home_state"]


def test_completed_games_separate_the_namesakes(tmp_path):
    path = _write(tmp_path, "games_2026.csv", GAME_COLS, [
        {"week": 1, "away": "Salem (Salem)", "away_score": 7,
         "home": "Poland Seminary (Poland)", "home_score": 21,
         "neutral": 0, "away_state": "", "home_state": ""},
        {"week": 1, "away": "Salem (Salem)", "away_score": 14,
         "home": "Woodstown (Woodstown)", "home_score": 10,
         "neutral": 0, "away_state": "NJ", "home_state": "NJ"},
    ])
    names = {side for g in load_games(path) for side in (g["home"], g["away"])}
    assert "Salem (Salem)" in names
    assert "Salem (Salem) [NJ]" in names


def test_fixtures_separate_the_namesakes(tmp_path):
    path = _write(tmp_path, "schedule_2026.csv", SCHED_COLS, [
        {"week": 5, "date": "2026-09-25", "time": "7pm",
         "away": "Salem (Salem)", "home": "Beaver Local (East Liverpool)",
         "neutral": 0, "away_state": "", "home_state": ""},
        {"week": 5, "date": "2026-09-25", "time": "7pm",
         "away": "Salem (Salem)", "home": "Arthur P Schalick (Pittsgrove)",
         "neutral": 0, "away_state": "NJ", "home_state": "NJ"},
    ])
    aways = [f["away"] for f in load_schedule(path)]
    assert aways == ["Salem (Salem)", "Salem (Salem) [NJ]"], aways


def test_the_ohio_team_does_not_inherit_the_namesake_schedule(tmp_path):
    """The bug in one assertion: one Ohio Salem, one fixture, not two."""
    path = _write(tmp_path, "schedule_2026.csv", SCHED_COLS, [
        {"week": w, "date": "", "time": "",
         "away": "Salem (Salem)", "home": f"Ohio Opponent {w} (Town)",
         "neutral": 0, "away_state": "", "home_state": ""}
        for w in range(2, 11)
    ] + [
        {"week": w, "date": "", "time": "",
         "away": "Salem (Salem)", "home": f"Jersey Opponent {w} (Town)",
         "neutral": 0, "away_state": "NJ", "home_state": "NJ"}
        for w in range(2, 11)
    ])
    fixtures = load_schedule(path)
    ohio = [f for f in fixtures if f["away"] == "Salem (Salem)"]
    assert len(ohio) == 9, f"Ohio Salem should have 9 fixtures, got {len(ohio)}"
    assert len(fixtures) == 18, "both schools' fixtures should survive, kept apart"


def test_a_team_with_no_state_column_still_loads(tmp_path):
    """Older CSVs predate the column; they must not crash or gain a tag."""
    path = _write(tmp_path, "schedule_2026.csv",
                  ["week", "date", "time", "away", "home", "neutral"], [
                      {"week": 2, "date": "", "time": "", "away": "A (X)",
                       "home": "B (Y)", "neutral": 0}])
    f = load_schedule(path)[0]
    assert f["away"] == "A (X)" and f["home"] == "B (Y)"
