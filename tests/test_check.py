"""
Completeness: has every game that should have been played been captured?

A week number cannot answer that. In a fixture list, "not played yet" and
"played, and we failed to read it" are the same row -- which is how two schools
once sat at zero games for a whole season without anything going red. The
kickoff date is what separates them: a fixture whose date has passed is a score
the board should be holding and is not.

The thresholds here are deliberately lopsided. A handful of overdue games is
ordinary -- a postponement, or the source posting late on a Sunday -- and must
never stop a good week's ratings from publishing. Half a week is not ordinary:
nobody postpones 180 games, so that shape means the scoreboard did not parse,
and every rating built around it is standing on a hole.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from check import (_name, overdue_fixtures, week_coverage,  # noqa: E402
                   week_date_spans, weeks_out_of_order)
from resolve import load_games  # noqa: E402

TODAY = "2026-09-10"


def fx(week, date, home=0, away=1):
    return {"w": week, "d": date, "h": home, "a": away}


def rs(week, date=None, home=0, away=1):
    r = {"w": week, "h": home, "a": away, "hs": 21, "as": 14}
    if date:
        r["d"] = date
    return r


# ------------------------------------------------------- overdue fixtures

def test_a_fixture_past_its_date_is_a_missed_game():
    sched = [fx(3, "2026-09-04"), fx(4, "2026-09-11")]
    late = overdue_fixtures(sched, TODAY)
    assert [g["d"] for g in late] == ["2026-09-04"]


def test_a_fixture_on_todays_date_is_not_yet_overdue():
    """The build runs in the morning; that evening's games are still ahead."""
    assert overdue_fixtures([fx(4, TODAY)], TODAY) == []


def test_a_fixture_with_no_date_is_never_flagged():
    """Seasons scraped before the date column carry none. Absence of evidence
    is not evidence of a missed game."""
    assert overdue_fixtures([{"w": 3, "h": 0, "a": 1}], TODAY) == []


def test_the_verdict_follows_the_build_not_the_wall_clock():
    """`today` is the build's generatedAt, so a pinned rebuild reproduces it."""
    sched = [fx(3, "2026-09-04")]
    assert overdue_fixtures(sched, "2026-09-01") == []
    assert len(overdue_fixtures(sched, "2026-09-05")) == 1


# --------------------------------------------------------- week coverage

def test_a_few_late_scores_do_not_condemn_the_week():
    """The real 2026 baseline was one overdue game against a full week."""
    sched = [fx(2, "2026-09-04")]
    results = [rs(2) for _ in range(300)]
    (missing, got), = week_coverage(sched, results, TODAY).values()
    assert (missing, got) == (1, 300)
    assert missing < 20 or missing <= (missing + got) * 0.5   # the check.py rule


def test_half_a_week_missing_is_a_parse_failure():
    sched = [fx(2, "2026-09-04") for _ in range(180)]
    results = [rs(2) for _ in range(20)]
    (missing, got), = week_coverage(sched, results, TODAY).values()
    assert (missing, got) == (180, 20)
    assert not (missing < 20 or missing <= (missing + got) * 0.5)


def test_a_week_still_in_the_future_is_not_counted():
    assert week_coverage([fx(9, "2026-10-30")], [], TODAY) == {}


# ------------------------------------------------- weeks must not overlap

def test_week_spans_come_from_the_dates_present():
    spans = week_date_spans([rs(1, "2026-08-21"), rs(1, "2026-08-22"),
                             rs(2, "2026-08-28")])
    assert spans == {1: ("2026-08-21", "2026-08-22"), 2: ("2026-08-28", "2026-08-28")}


def test_clean_week_numbering_raises_nothing():
    spans = week_date_spans([rs(1, "2026-08-21"), rs(2, "2026-08-28"),
                             rs(3, "2026-09-04")])
    assert weeks_out_of_order(spans) == []


def test_a_game_filed_under_the_wrong_week_is_caught():
    """Week numbering is the spine of the model -- the prior, the walk-forward
    tuning and the track record are all keyed on it. A game filed a week early
    is invisible in a fixture list and quietly shifts a rating."""
    spans = week_date_spans([rs(1, "2026-08-21"), rs(2, "2026-08-28"),
                             rs(2, "2026-09-04"), rs(3, "2026-09-04")])
    bad = weeks_out_of_order(spans)
    assert [b[0] for b in bad] == [2]


def test_undated_results_are_simply_absent_from_the_spans():
    assert week_date_spans([rs(1), rs(2)]) == {}


# ---------------------------------------------------------- name rendering

def test_an_unresolved_fixture_still_renders_in_a_warning():
    payload = {"teams": [{"name": "Alpha (A)"}]}
    assert _name(payload, 0) == "Alpha (A)"
    assert _name(payload, "Someone (Nowhere)") == "Someone (Nowhere)"
    assert _name(payload, 99) == "#99"          # out of range, must not raise


# ------------------------------------------------- the CSV contract itself

def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_load_games_carries_the_kickoff_date(tmp_path):
    p = _write(tmp_path / "g.csv",
               ["week", "date", "away", "away_score", "home", "home_score",
                "neutral", "away_state", "home_state"],
               [[1, "2026-08-21", "Bravo (B)", 0, "Alpha (A)", 42, 0, "", ""]])
    g, = load_games(p)
    assert g["date"] == "2026-08-21"
    assert g["week"] == 1 and g["home_score"] == 42


def test_a_season_scraped_before_the_date_column_still_loads(tmp_path):
    """2023-2025 are committed without it and are finished history; the column
    must be optional or the whole backtest stops loading."""
    p = _write(tmp_path / "g.csv",
               ["week", "away", "away_score", "home", "home_score",
                "neutral", "away_state", "home_state"],
               [[1, "Bravo (B)", 0, "Alpha (A)", 42, 0, "", ""]])
    g, = load_games(p)
    assert g["date"] == ""
    assert g["home_score"] == 42
