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
                     build_snapshot, load, pred_keys, record, score, trends)


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
    """Without results to check against, a week is written once and never
    revised. `append_if_new` is the strict form, used by the backtest replay:
    a caller that cannot prove the forecast games are still unplayed does not
    get to rewrite anything."""
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


def test_a_snapshot_with_nothing_to_predict_is_refused(tmp_path):
    """A finished season has nothing to forecast.

    The workflow builds LAST season every run to derive the preseason prior.
    That build has no remaining fixtures, and without this guard it appended a
    line for 2025 week 16 marked "live" -- a completed season recorded as
    though it had been foreseen. It would have done so on every run, and the
    log's entire value is that its entries predate the games.
    """
    empty = build_snapshot(2025, 16, "t", {}, [team("A")], [])
    assert empty["pred"] == []
    # build.py refuses to append this; the log stays as it was.
    p = str(tmp_path / "h.jsonl")
    if empty["pred"]:
        append_if_new(p, empty)
    assert load(p) == []


def test_a_snapshot_with_predictions_is_still_recorded(tmp_path):
    p = str(tmp_path / "h.jsonl")
    live = snap(1)
    assert live["pred"]
    assert append_if_new(p, live) is True

# ------------------------------------------- revisable until the games start
#
# The log's rule is not "write once". It is "never revise a forecast for a game
# that has been played". Those differ, and the difference caused real damage:
# `through_week` is the highest week holding ANY result, so one Thursday-night
# game turns the week over while ~350 Friday fixtures are still to come. Under
# first-write-wins, whichever build ran first owned the week -- on 2026 week 2
# a manual Friday-morning run locked in week 3's predictions from a model that
# had seen 27 of week 2's 357 games, and Saturday's far better forecast was
# refused in silence. The record was not flattered; it was made incoherent,
# because the standard then depended on when somebody clicked "Run workflow".

def test_a_capture_is_improved_while_its_games_are_still_ahead(tmp_path):
    p = str(tmp_path / "h.jsonl")
    early = snap(2, preds=[fixture("A", "B", 3, 3.0, 0.55)])
    assert record(p, early, played={}) == "appended"

    # Week 2 has now finished; week 3 has not started. Revising a forecast for
    # a game nobody has played is not hindsight -- there is no result to see.
    later = snap(2, preds=[fixture("A", "B", 3, 21.0, 0.95)])
    played = {(2, "A", "B"): 14}
    assert record(p, later, played=played) == "replaced"

    lines = load(p)
    assert len(lines) == 1
    assert lines[0]["pred"][0][3] == 21.0


def test_a_capture_freezes_once_one_of_its_games_is_played(tmp_path):
    """The moment a forecast game kicks off, that line is a call made before
    kickoff and must survive untouched."""
    p = str(tmp_path / "h.jsonl")
    before = snap(2, preds=[fixture("A", "B", 3, 3.0, 0.55),
                            fixture("C", "D", 3, 3.0, 0.55)])
    assert record(p, before, played={}) == "appended"

    # Only ONE of the two has been played. The line still freezes whole: it is
    # one capture at one instant, not a bag of independent rows.
    played = {(3, "A", "B"): 10}
    after = snap(2, preds=[fixture("A", "B", 3, 21.0, 0.95),
                           fixture("C", "D", 3, 21.0, 0.95)])
    assert record(p, after, played=played) == "kept"
    assert load(p)[0]["pred"][0][3] == 3.0


def test_the_home_away_order_of_a_result_key_is_respected(tmp_path):
    """`pred_keys` must line up with the map build.py scores against, which is
    keyed (week, home, away). Reading it the other way round would make every
    played game look unplayed and the freeze would never fire."""
    s = snap(2, preds=[fixture("Home", "Away", 3, 7.0, 0.7)])
    assert pred_keys(s) == {(3, "Home", "Away")}
    p = str(tmp_path / "h.jsonl")
    record(p, s, played={})
    assert record(p, s, played={(3, "Away", "Home"): 7}) == "replaced"
    assert record(p, s, played={(3, "Home", "Away"): 7}) == "kept"


def test_without_results_nothing_is_revised(tmp_path):
    """A caller that cannot prove the games are unplayed gets the strict rule."""
    p = str(tmp_path / "h.jsonl")
    record(p, snap(2, preds=[fixture("A", "B", 3, 3.0, 0.55)]), played={})
    assert record(p, snap(2, preds=[fixture("A", "B", 3, 9.0, 0.8)])) == "kept"
    assert load(p)[0]["pred"][0][3] == 3.0


def test_a_replay_cannot_revise_a_live_capture(tmp_path):
    """Same property as the append-only version, through the new door. A
    backtest was produced by constants fitted on that very season."""
    p = str(tmp_path / "h.jsonl")
    record(p, snap(3, kind=KIND_LIVE), played={})
    assert record(p, snap(3, kind=KIND_BACKTEST), played={}) == "kept"
    assert [s["kind"] for s in load(p)] == [KIND_LIVE]


def test_a_revision_leaves_every_other_line_untouched_and_in_order(tmp_path):
    """A revision rewrites the file, so the rest of the log has to come through
    unchanged -- this is the one file in the repo that cannot be regenerated."""
    p = str(tmp_path / "h.jsonl")
    for wk in (1, 2, 3):
        record(p, snap(wk, preds=[fixture("A", "B", wk + 1, float(wk), 0.6)]),
               played={})
    assert record(p, snap(2, preds=[fixture("A", "B", 3, 99.0, 0.9)]),
                  played={}) == "replaced"

    lines = load(p)
    assert [l["throughWeek"] for l in lines] == [1, 2, 3]
    assert [l["pred"][0][3] for l in lines] == [1.0, 99.0, 3.0]
    # Written atomically, so no leftover temp file beside the log.
    assert not os.path.exists(p + ".tmp")


def test_the_2026_week_2_regression(tmp_path):
    """The concrete case this rule was written for.

    Friday 10:42am: 27 of week 2's games are in, `through_week` has flipped to
    2, and a manual run records week 3's forecast. Saturday 08:00: all of week
    2 is in and the model is materially better. Week 3 is still six days away,
    so Saturday's forecast must win.
    """
    p = str(tmp_path / "h.jsonl")
    friday = snap(2, preds=[fixture("A", "B", 3, 2.0, 0.52)])
    record(p, friday, played={(2, "X", "Y"): 7})          # a Thursday game

    week2_done = {(2, "X", "Y"): 7, (2, "A", "B"): 3, (2, "C", "D"): 21}
    saturday = snap(2, preds=[fixture("A", "B", 3, 17.0, 0.88)])
    assert record(p, saturday, played=week2_done) == "replaced"
    assert load(p)[0]["pred"][0][3] == 17.0

    # ...and once week 3 kicks off, it is frozen for good.
    week3_started = dict(week2_done)
    week3_started[(3, "A", "B")] = -6
    assert record(p, snap(2, preds=[fixture("A", "B", 3, -6.0, 0.1)]),
                  played=week3_started) == "kept"
    assert load(p)[0]["pred"][0][3] == 17.0
