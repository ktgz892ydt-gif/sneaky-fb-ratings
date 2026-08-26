"""
The track record.

The property this file exists to protect: a *backtest* must never be presented
as a *live* result. A replayed week was produced by constants fitted on that
same season; a live week was captured before the games were played. Pooling
them launders the weaker number into the stronger one, and the headline that
results would look completely reasonable.

The second property: the log is append-only and cannot be regenerated. Every
other file in data/ is derived from the source. A prediction is what the board
said at a moment in time, and recomputing it from a model that has since seen
the result is a retrofit, not a forecast.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from history import (KIND_BACKTEST, KIND_LIVE, append_if_new,  # noqa: E402
                     build_snapshot, load, score, trends)


def team(name, rating=1.0, rank=1, odds=0.5, games=1, ohio=True):
    return {"name": name, "rating": rating, "rank": rank, "playoffOdds": odds,
            "games": games, "inOhio": ohio}


def fixture(home, away, week, margin, prob):
    return {"predicted": True, "week": week, "homeName": home, "awayName": away,
            "predictedHomeMargin": margin, "homeWinProb": prob}


def snap(week=1, kind=KIND_LIVE, season=2026, preds=None, teams=None):
    return build_snapshot(season, week, "t", {}, teams or [team("A"), team("B")],
                          preds if preds is not None else [fixture("A", "B", week + 1, 7.0, 0.7)],
                          kind=kind)


# ------------------------------------------------------------- append-only

def test_a_week_is_recorded_once(tmp_path):
    """The first look at a week is the prediction. A later build has seen more
    of the season, so re-recording would swap a forecast for hindsight."""
    p = str(tmp_path / "h.jsonl")
    assert append_if_new(p, snap(1)) is True
    assert append_if_new(p, snap(1)) is False
    assert append_if_new(p, snap(2)) is True
    assert len(load(p)) == 2


def test_a_replay_cannot_overwrite_a_live_capture(tmp_path):
    p = str(tmp_path / "h.jsonl")
    append_if_new(p, snap(3, kind=KIND_LIVE))
    append_if_new(p, snap(3, kind=KIND_BACKTEST))
    kinds = [s["kind"] for s in load(p)]
    assert kinds == [KIND_LIVE], kinds


def test_a_truncated_line_does_not_destroy_the_log(tmp_path):
    """A half-written line is what an interrupted commit looks like. Losing the
    whole history to one bad append would be a poor trade."""
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps(snap(1)) + "\n" + '{"season":2026,"throughWe\n',
                 encoding="utf-8")
    assert len(load(str(p))) == 1


def test_a_missing_log_is_empty_not_an_error():
    assert load("/nonexistent/history.jsonl") == []


# --------------------------------------------------------- never pooled

def test_live_and_backtest_are_scored_separately():
    snaps = [snap(1, kind=KIND_LIVE, preds=[fixture("A", "B", 2, 7.0, 0.9)]),
             snap(1, kind=KIND_BACKTEST, season=2025,
                  preds=[fixture("A", "B", 2, 7.0, 0.9)])]
    results = {2026: {(2, "A", "B"): 10}, 2025: {(2, "A", "B"): -10}}
    sc = score(snaps, results)
    assert set(sc["overall"]) == {KIND_LIVE, KIND_BACKTEST}
    assert sc["overall"][KIND_LIVE]["accuracy"] == 1.0
    assert sc["overall"][KIND_BACKTEST]["accuracy"] == 0.0
    # and never merged into one figure
    assert "accuracy" not in sc["overall"]


def test_an_unplayed_prediction_is_not_counted_as_a_miss():
    """Punishing the board for games the source has not posted would make the
    record look worse every time the site runs late."""
    snaps = [snap(1, preds=[fixture("A", "B", 2, 7.0, 0.9),
                            fixture("C", "D", 2, 7.0, 0.9)])]
    sc = score(snaps, {2026: {(2, "A", "B"): 10}})
    assert sc["overall"][KIND_LIVE]["games"] == 1


def test_a_tie_is_not_scored():
    sc = score([snap(1, preds=[fixture("A", "B", 2, 7.0, 0.9)])],
               {2026: {(2, "A", "B"): 0}})
    assert sc["overall"] == {} or not sc["overall"].get(KIND_LIVE)


def test_a_confident_wrong_call_costs_more_than_a_hedged_one():
    confident = score([snap(1, preds=[fixture("A", "B", 2, 20.0, 0.95)])],
                      {2026: {(2, "A", "B"): -3}})
    hedged = score([snap(1, preds=[fixture("A", "B", 2, 1.0, 0.55)])],
                   {2026: {(2, "A", "B"): -3}})
    assert (confident["overall"][KIND_LIVE]["logloss"]
            > hedged["overall"][KIND_LIVE]["logloss"])


# ------------------------------------------------------------------ trends

def test_a_trend_needs_at_least_two_weeks():
    one = trends([snap(1)], 2026, ["A"])
    assert one == {}
    two = trends([snap(1), snap(2)], 2026, ["A"])
    assert two["A"]["w"] == [1, 2]


def test_trend_arrays_stay_parallel():
    series = trends([snap(1), snap(2), snap(3)], 2026, ["A", "B"])
    for v in series.values():
        assert len(v["w"]) == len(v["rating"]) == len(v["odds"])


def test_a_backtest_line_carries_no_team_block():
    """Backtests are replayable, so storing their ratings duplicates something
    a two-second script regenerates. It cost 1.2 MB of the log's first 1.7."""
    b = build_snapshot(2025, 3, "t", {}, [team("A")], [], kind=KIND_BACKTEST,
                       include_teams=False)
    assert b["keys"] == [] and b["teams"] == []
    live = build_snapshot(2026, 3, "t", {}, [team("A")], [])
    assert live["keys"] == ["A"]


def test_only_the_next_week_is_recorded():
    """The claim worth being held to is 'we called this Friday right'."""
    s = build_snapshot(2026, 4, "t", {}, [team("A")],
                       [fixture("A", "B", 5, 7.0, 0.7),
                        fixture("A", "C", 6, 7.0, 0.7),
                        fixture("A", "D", 4, 7.0, 0.7)])
    assert [p[2] for p in s["pred"]] == [5]
