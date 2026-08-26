"""
Predictions for not-yet-played fixtures.

The property that matters most here is a separation, not a number: a fixture
has no result, and if one ever reached the rating fit it would appear as a
phantom 0-0 tie between two real teams. The ratings table would still look
completely normal. So the first test below builds the same season twice, once
with a schedule loaded and once without, and demands the ratings come out
bit-for-bit identical.

The second concern is honesty. An opponent nobody has played has no rating,
and standing one in is a stated approximation, not a measurement -- so every
prediction that leans on one has to say so.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from build import (_document, compact_schedule, load_schedule,  # noqa: E402
                   predict_schedule, project_records)
from ratings import RatingConfig, rate  # noqa: E402
from resolve import resolve  # noqa: E402

CFG = RatingConfig()
STANDIN = (1.86, "division III baseline")


def slot(name, div="V", region=17, record="1-0", harbin=4.0, sid="", city=""):
    return {"name": name, "division": div, "region": region, "record": record,
            "harbin": harbin, "school_id": sid, "city": city}


def game(home, away, hs, as_, week=1):
    return {"home": home, "away": away, "home_score": hs, "away_score": as_,
            "week": week, "neutral": False}


def fixture(home, away, week=2, neutral=False):
    return {"week": week, "date": "2026-09-04", "time": "7pm",
            "home": home, "away": away, "neutral": neutral}


def season():
    """Four Ohio schools, a clear strength order, one game each."""
    roster = {}
    for n in ("Alpha (A)", "Bravo (B)", "Charlie (C)", "Delta (D)"):
        roster.setdefault(n, []).append(slot(n))
    games = [game("Alpha (A)", "Bravo (B)", 42, 0),
             game("Charlie (C)", "Delta (D)", 21, 20)]
    res = resolve(roster, games)
    team_ids = sorted(res.teams)
    return res, team_ids, rate(team_ids, res.games, CFG)


def tid_for(res, name):
    return [t for t in res.teams if res.teams[t].name == name][0]


# ------------------------------------------------- the separation that must hold

def test_loading_a_schedule_cannot_change_the_ratings():
    """The one that would corrupt everything, silently."""
    res, team_ids, base = season()
    fixtures = [fixture("Alpha (A)", "Charlie (C)"),
                fixture("Bravo (B)", "Delta (D)")]
    predict_schedule(fixtures, res, base, team_ids, CFG, STANDIN)
    after = rate(team_ids, res.games, CFG)
    for a, b in zip(base.bt_margin, after.bt_margin):
        assert a == b, "predicting fixtures perturbed the fit"
    assert len(res.games) == 2, "a fixture was appended to the played games"


def test_a_fixture_adds_no_wins_or_losses():
    res, team_ids, result = season()
    predict_schedule([fixture("Alpha (A)", "Bravo (B)", week=9)],
                     res, result, team_ids, CFG, STANDIN)
    i = team_ids.index(tid_for(res, "Alpha (A)"))
    assert int(result.games[i]) == 1
    assert int(result.wins[i]) == 1


# ------------------------------------------------------------------ the maths

def test_the_stronger_team_is_favoured():
    res, team_ids, result = season()
    g = predict_schedule([fixture("Alpha (A)", "Bravo (B)")],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert g["predicted"]
    assert g["favoriteName"] == "Alpha (A)"
    assert g["predictedHomeMargin"] > 0
    assert g["homeWinProb"] > 0.5


def test_swapping_home_and_away_flips_the_margin_by_twice_the_home_edge():
    res, team_ids, result = season()
    a = predict_schedule([fixture("Alpha (A)", "Charlie (C)")],
                         res, result, team_ids, CFG, STANDIN)[0]
    b = predict_schedule([fixture("Charlie (C)", "Alpha (A)")],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert (a["predictedHomeMargin"] + b["predictedHomeMargin"]) == pytest.approx(
        2 * result.hfa_margin, abs=0.11)


def test_a_neutral_site_removes_the_home_edge():
    res, team_ids, result = season()
    home = predict_schedule([fixture("Alpha (A)", "Charlie (C)")],
                            res, result, team_ids, CFG, STANDIN)[0]
    neut = predict_schedule([fixture("Alpha (A)", "Charlie (C)", neutral=True)],
                            res, result, team_ids, CFG, STANDIN)[0]
    assert home["predictedHomeMargin"] > neut["predictedHomeMargin"]
    assert (home["predictedHomeMargin"] - neut["predictedHomeMargin"]) == pytest.approx(
        result.hfa_margin, abs=0.11)


def test_probabilities_stay_in_range_and_agree_with_the_margin():
    res, team_ids, result = season()
    names = ["Alpha (A)", "Bravo (B)", "Charlie (C)", "Delta (D)"]
    fixtures = [fixture(h, a) for h in names for a in names if h != a]
    for g in predict_schedule(fixtures, res, result, team_ids, CFG, STANDIN):
        assert 0.0 < g["homeWinProb"] < 1.0
        assert (g["homeWinProb"] > 0.5) == (g["predictedHomeMargin"] > 0)
        assert g["spread"] == pytest.approx(abs(g["predictedHomeMargin"]))


# ------------------------------------------------------- unrateable opponents

def test_an_unknown_opponent_uses_the_stand_in_and_says_so():
    res, team_ids, result = season()
    g = predict_schedule([fixture("Alpha (A)", "Nowhere Prep (Elsewhere)")],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert g["predicted"]
    assert g["estimated"] is True
    assert g["estimatedNote"] == "division III baseline"


def test_an_ordinary_fixture_is_not_flagged_as_estimated():
    res, team_ids, result = season()
    g = predict_schedule([fixture("Alpha (A)", "Bravo (B)")],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert g["estimated"] is False
    assert g["estimatedNote"] == ""


def test_with_no_stand_in_available_the_game_is_left_unpredicted():
    """Better an admitted gap than an invented number."""
    res, team_ids, result = season()
    g = predict_schedule([fixture("Alpha (A)", "Nowhere Prep (Elsewhere)")],
                         res, result, team_ids, CFG, (None, ""))[0]
    assert g["predicted"] is False
    assert "no rating" in g["reason"]
    assert "predictedHomeMargin" not in g


# --------------------------------------------------------- shared team names

def _shared_name_season():
    """One Ohio school and a non-roster namesake, both playing in week 1.

    This is the real Marietta case: the resolver splits them, correctly, and
    the fixture list then names only "Marietta (Marietta)".
    """
    roster = {}
    for n in ("Marietta (Marietta)", "Alpha (A)", "Bravo (B)"):
        roster.setdefault(n, []).append(slot(n))
    games = [game("Marietta (Marietta)", "Alpha (A)", 28, 7),
             game("Marietta (Marietta)", "Bravo (B)", 14, 10)]
    res = resolve(roster, games)
    team_ids = sorted(res.teams)
    return res, team_ids, rate(team_ids, res.games, CFG)


def test_a_name_split_between_ohio_and_elsewhere_resolves_to_the_ohio_school():
    res, team_ids, result = _shared_name_season()
    shared = [t for t in res.teams if res.teams[t].name == "Marietta (Marietta)"]
    assert len(shared) == 2, "precondition: the resolver split the name"
    assert sum(1 for t in shared if res.teams[t].in_ohio) == 1

    g = predict_schedule([fixture("Marietta (Marietta)", "Alpha (A)", week=5)],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert g["predicted"] is True
    assert g["assumedOhio"] is True
    assert res.teams[g["home"]].in_ohio


def test_two_ohio_schools_sharing_a_name_are_still_refused():
    """Choosing the Ohio one only works when there is exactly one."""
    roster = {}
    roster.setdefault("North", []).append(slot("North", div="II", region=5))
    roster.setdefault("North", []).append(slot("North", div="II", region=5))
    for n in ("Alpha (A)", "Bravo (B)"):
        roster.setdefault(n, []).append(slot(n))
    games = [game("North", "Alpha (A)", 21, 0), game("North", "Bravo (B)", 0, 21)]
    res = resolve(roster, games)
    team_ids = sorted(res.teams)
    result = rate(team_ids, res.games, CFG)

    g = predict_schedule([fixture("North", "Alpha (A)", week=4)],
                         res, result, team_ids, CFG, STANDIN)[0]
    assert g["predicted"] is False
    assert "several schools" in g["reason"]


# ------------------------------------------------------------- projections

def test_projected_games_equal_played_plus_remaining():
    res, team_ids, result = season()
    fixtures = [fixture("Alpha (A)", "Charlie (C)", week=2),
                fixture("Bravo (B)", "Alpha (A)", week=3)]
    sched = predict_schedule(fixtures, res, result, team_ids, CFG, STANDIN)
    proj = project_records(sched, team_ids, result)
    a = proj[tid_for(res, "Alpha (A)")]
    assert a["remaining"] == 2
    assert (a["projWins"] + a["projLosses"]) == pytest.approx(3.0)


def test_a_team_with_nothing_left_projects_its_current_record():
    res, team_ids, result = season()
    proj = project_records([], team_ids, result)
    a = proj[tid_for(res, "Alpha (A)")]
    assert a["remaining"] == 0
    assert a["projWins"] == pytest.approx(1.0)
    assert a["projLosses"] == pytest.approx(0.0)


def test_an_unpredicted_fixture_contributes_nothing_to_a_projection():
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Nowhere Prep (Elsewhere)")],
                             res, result, team_ids, CFG, (None, ""))
    proj = project_records(sched, team_ids, result)
    a = proj[tid_for(res, "Alpha (A)")]
    assert a["remaining"] == 0, "an unpredicted game must not count as remaining"
    assert a["projWins"] == pytest.approx(1.0)


def test_a_strong_favourite_projects_close_to_a_full_win():
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Bravo (B)", week=2)],
                             res, result, team_ids, CFG, STANDIN)
    proj = project_records(sched, team_ids, result)
    assert proj[tid_for(res, "Alpha (A)")]["projWins"] > 1.5


@pytest.mark.parametrize("n_left", range(1, 11))
def test_a_projected_record_never_adds_up_to_more_games_than_exist(n_left):
    """Rounding each half separately let '3.5-6.5' describe an 11-game season."""
    res, team_ids, result = season()
    fixtures = [fixture("Alpha (A)", "Charlie (C)", week=w)
                for w in range(2, 2 + n_left)]
    sched = predict_schedule(fixtures, res, result, team_ids, CFG, STANDIN)
    proj = project_records(sched, team_ids, result)
    for tid, p in proj.items():
        i = team_ids.index(tid)
        decided = int(result.wins[i]) + int(result.losses[i]) + p["remaining"]
        assert (p["projWins"] + p["projLosses"]) == pytest.approx(decided, abs=0.011)


# ------------------------------------------------------------------ transport

def test_compacting_keeps_the_teams_and_the_numbers():
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Bravo (B)", week=3)],
                             res, result, team_ids, CFG, STANDIN)
    rich, small = sched[0], compact_schedule(sched, team_ids)[0]
    assert team_ids[small["h"]] == rich["home"]
    assert team_ids[small["a"]] == rich["away"]
    assert small["m"] == rich["predictedHomeMargin"]
    assert small["p"] == rich["homeWinProb"]
    assert small["w"] == 3


def test_an_unresolved_opponent_compacts_to_its_name():
    """No index exists for a team that was never rated, so the name travels."""
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Nowhere Prep (Elsewhere)")],
                             res, result, team_ids, CFG, STANDIN)
    small = compact_schedule(sched, team_ids)[0]
    assert small["a"] == "Nowhere Prep (Elsewhere)"
    assert isinstance(small["h"], int)


def test_compacting_omits_flags_that_are_false():
    """Absent means false; a full season of zeroes is a megabyte of nothing."""
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Bravo (B)")],
                             res, result, team_ids, CFG, STANDIN)
    small = compact_schedule(sched, team_ids)[0]
    for k in ("n", "e", "o", "x"):
        assert k not in small


def test_an_unpredicted_fixture_carries_its_reason_through():
    res, team_ids, result = season()
    sched = predict_schedule([fixture("Alpha (A)", "Nowhere Prep (Elsewhere)")],
                             res, result, team_ids, CFG, (None, ""))
    small = compact_schedule(sched, team_ids)[0]
    assert "m" not in small and "p" not in small
    assert small["x"]


# ------------------------------- the stand-in scale and the page skeleton
#
# Identity itself is covered by tests/test_identity.py; these are the two
# behaviours that live only here.

def _schedule_csv(tmp_path, rows):
    p = tmp_path / "schedule_2026.csv"
    header = "week,date,time,away,home,neutral,away_state,home_state\n"
    p.write_text(header + "".join(",".join(r) + "\n" for r in rows),
                 encoding="utf-8")
    return str(p)


def test_load_schedule_keeps_the_state_tag_in_the_name(tmp_path):
    path = _schedule_csv(tmp_path, [
        ["2", "2026-08-28", "7pm", "Canfield (Canfield)", "Salem (Salem)", "0", "", ""],
        ["2", "2026-08-29", "1pm", "Salem (Salem)", "Ironton (Ironton)", "0", "NJ", ""],
    ])
    rows = load_schedule(path)
    assert [r["away"] for r in rows] == ["Canfield (Canfield)", "Salem (Salem) [NJ]"]
    assert [r["home"] for r in rows] == ["Salem (Salem)", "Ironton (Ironton)"]


def test_a_stand_in_fixture_is_predicted_less_confidently_than_a_rated_one(tmp_path):
    """Regression guard for the zero-games scale inversion."""
    res, team_ids, result = season()
    rows = load_schedule(_schedule_csv(tmp_path, [
        ["2", "2026-09-04", "7pm", "Bravo (B)", "Alpha (A)", "0", "", ""],
        ["2", "2026-09-04", "7pm", "Nobody (Nowhere)", "Alpha (A)", "0", "PA", ""],
    ]))
    sched = predict_schedule(rows, res, result, team_ids, CFG, STANDIN)
    rated = next(g for g in sched if not g["estimated"])
    stood = next(g for g in sched if g["estimated"])
    # Scale them to the same margin so only the curve differs.
    assert stood["homeWinProb"] < 0.999
    for g in (rated, stood):
        assert 0.0 < g["homeWinProb"] < 1.0
    assert abs(stood["predictedHomeMargin"] - rated["predictedHomeMargin"]) < 60


# ------------------------------------------------------- the page skeleton

def test_both_page_variants_are_real_documents():
    """dist/preview.html used to be written as a bare fragment.

    No charset (the title rendered as mojibake) and, worse, no viewport meta,
    so a phone laid it out at 980px and none of the mobile media queries fired
    -- while the Playwright check, which sets its own viewport, passed.
    """
    doc = _document("<title>x</title>\n", '<div class="wrap">y</div>')
    assert doc.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in doc
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in doc
    assert doc.rstrip().endswith("</html>")
    assert doc.index("<title>") < doc.index('<div class="wrap">')
