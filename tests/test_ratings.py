"""Properties of the rating model that must hold regardless of the data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from ratings import RatingConfig, rate, squash  # noqa: E402


CFG = RatingConfig()


def _game(home, away, hs, as_, week=1, neutral=False):
    return {"home": home, "away": away, "home_score": hs, "away_score": as_,
            "week": week, "neutral": neutral}


# ---------------------------------------------------------------- squash

def test_squash_is_a_half_at_zero():
    assert squash(np.array([0.0]), CFG.squash_scale, CFG.margin_cap)[0] == pytest.approx(0.5)


@pytest.mark.parametrize("m", [1, 3, 7, 14, 21, 35, 60, 200])
def test_squash_is_symmetric(m):
    s = CFG.squash_scale, CFG.margin_cap
    assert (squash(np.array([float(m)]), *s)[0]
            + squash(np.array([float(-m)]), *s)[0]) == pytest.approx(1.0)


def test_squash_is_monotonic():
    v = squash(np.arange(0, 60, dtype=float), CFG.squash_scale, CFG.margin_cap)
    assert np.all(np.diff(v) >= 0)


def test_squash_clips_beyond_the_cap():
    a = squash(np.array([CFG.margin_cap]), CFG.squash_scale, CFG.margin_cap)[0]
    b = squash(np.array([CFG.margin_cap + 500]), CFG.squash_scale, CFG.margin_cap)[0]
    assert a == b, "margins past the cap must not keep adding credit"


def test_blowout_is_worth_less_than_proportional():
    """The whole point of the squash: 45 points is not 15x as good as 3."""
    s = CFG.squash_scale, CFG.margin_cap
    small = squash(np.array([3.0]), *s)[0] - 0.5
    big = squash(np.array([45.0]), *s)[0] - 0.5
    assert big > small
    assert big / small < 15


# ---------------------------------------------------------------- ratings

def test_beating_someone_ranks_you_above_them():
    ids = ["A", "B"]
    r = rate(ids, [_game("A", "B", 28, 7)], CFG)
    assert r.bt_margin[0] > r.bt_margin[1]


def test_ratings_are_centred_and_finite():
    ids = list("ABCDEF")
    games = [_game("A", "B", 21, 14), _game("C", "D", 35, 0),
             _game("E", "F", 7, 6), _game("A", "C", 14, 10, week=2)]
    r = rate(ids, games, CFG)
    assert np.all(np.isfinite(r.bt_margin))
    assert r.bt_margin.mean() == pytest.approx(0.0, abs=1e-6)


def test_a_disconnected_graph_still_solves():
    """Week 1 is hundreds of two-team islands; the fit must not diverge."""
    ids = [f"T{i}" for i in range(20)]
    games = [_game(f"T{i}", f"T{i+1}", 21, 0) for i in range(0, 20, 2)]
    r = rate(ids, games, CFG)
    assert r.converged
    assert np.all(np.isfinite(r.bt_margin))


def test_neutral_site_games_get_no_home_advantage():
    """A team playing only on neutral fields must not be credited with HFA."""
    ids = list("ABCD")
    home = rate(ids, [_game("A", "B", 24, 21)] * 1, CFG)
    neutral = rate(ids, [_game("A", "B", 24, 21, neutral=True)], CFG)
    # With the home edge removed, the same scoreline implies a stronger winner.
    assert neutral.bt_margin[0] >= home.bt_margin[0]


def test_home_advantage_is_positive_when_home_teams_win():
    ids = [f"T{i}" for i in range(40)]
    games = [_game(f"T{i}", f"T{i+1}", 24, 17, week=1) for i in range(0, 40, 2)]
    r = rate(ids, games, CFG)
    assert r.hfa_margin > 0


def test_binary_model_ignores_margin():
    """Same winners, wildly different margins -> identical W/L ratings."""
    ids = list("ABCD")
    close = [_game("A", "B", 21, 20), _game("C", "D", 22, 21)]
    blowout = [_game("A", "B", 63, 0), _game("C", "D", 55, 3)]
    assert rate(ids, close, CFG).bt_binary == pytest.approx(
        rate(ids, blowout, CFG).bt_binary)


def test_margin_model_does_not_ignore_margin():
    ids = list("ABCD")
    close = rate(ids, [_game("A", "B", 21, 20), _game("C", "D", 22, 21)], CFG)
    blow = rate(ids, [_game("A", "B", 63, 0), _game("C", "D", 22, 21)], CFG)
    assert blow.bt_margin[0] > close.bt_margin[0]


# ---------------------------------------------------------------- priors

def test_prior_moves_a_team_it_names():
    ids = list("ABCD")
    games = [_game("A", "B", 21, 14), _game("C", "D", 21, 14)]
    base = rate(ids, games, CFG)
    lifted = rate(ids, games, CFG, priors={"A": 10.0})
    assert lifted.bt_margin[0] > base.bt_margin[0]


def test_prior_influence_shrinks_as_games_accumulate():
    ids = [f"T{i}" for i in range(30)]

    def spread(weeks):
        games = []
        for w in range(weeks):
            for i in range(0, 30, 2):
                games.append(_game(f"T{(i + w) % 30}", f"T{(i + w + 1) % 30}",
                                   21, 14, week=w + 1))
        a = rate(ids, games, CFG, priors={t: 0.0 for t in ids})
        b = rate(ids, games, CFG, priors={"T0": 10.0})
        return b.bt_margin[0] - a.bt_margin[0]

    assert spread(8) < spread(1), "the prior must fade as evidence arrives"


def test_priors_are_recentred_not_taken_literally():
    """A prior that shifts everyone equally shifts nobody's ranking."""
    ids = list("ABCD")
    games = [_game("A", "B", 21, 14), _game("C", "D", 28, 7)]
    flat = rate(ids, games, CFG, priors={t: 5.0 for t in ids})
    none = rate(ids, games, CFG, priors={t: 0.0 for t in ids})
    assert flat.bt_margin == pytest.approx(none.bt_margin, abs=1e-6)


# ---------------------------------------------------------------- records

def test_records_match_the_games_supplied():
    ids = list("ABC")
    games = [_game("A", "B", 21, 0), _game("A", "C", 7, 14, week=2)]
    r = rate(ids, games, CFG)
    i = {t: n for n, t in enumerate(ids)}
    assert (r.wins[i["A"]], r.losses[i["A"]], r.games[i["A"]]) == (1, 1, 2)
    assert (r.wins[i["B"]], r.losses[i["B"]]) == (0, 1)
    assert (r.wins[i["C"]], r.losses[i["C"]]) == (1, 0)


def test_a_tie_counts_as_a_tie():
    r = rate(list("AB"), [_game("A", "B", 14, 14)], CFG)
    assert r.ties[0] == 1 and r.wins[0] == 0 and r.losses[0] == 0
