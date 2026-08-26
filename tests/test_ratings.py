"""Properties of the rating model that must hold regardless of the data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from ratings import (  # noqa: E402
    expected_margin,
    RatingConfig, prob_scale, rate, squash, win_probability,
)


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


# ------------------------------------------------- margin -> win probability

def test_prob_scale_falls_back_to_squash_scale_before_any_games():
    """Week 1 predictions come from the prior, not a fit. Flat scale there."""
    assert prob_scale(0, CFG) == pytest.approx(CFG.squash_scale)


def test_prob_scale_steepens_as_teams_play():
    """More games -> less rating error -> a sharper probability curve."""
    scales = [prob_scale(g, CFG) for g in range(1, 11)]
    assert all(b < a for a, b in zip(scales, scales[1:])), scales


def test_prob_scale_stays_inside_its_bounds():
    for g in (1, 2, 5, 10, 50, 10_000):
        assert CFG.prob_scale_min <= prob_scale(g, CFG) <= CFG.prob_scale_max


def test_prob_scale_uses_the_fitted_curve_not_the_squash_scale():
    """Regression guard: the two must not silently collapse back together."""
    assert abs(prob_scale(10, CFG) - CFG.squash_scale) > 1.0


def test_win_probability_is_a_half_at_zero_margin():
    for g in (0, 1, 5, 10):
        assert win_probability(0.0, g, CFG) == pytest.approx(0.5)


@pytest.mark.parametrize("g", [0, 1, 4, 10])
@pytest.mark.parametrize("m", [1.0, 3.5, 7.0, 21.0, 45.0])
def test_win_probability_is_symmetric(g, m):
    assert win_probability(m, g, CFG) + win_probability(-m, g, CFG) == pytest.approx(1.0)


@pytest.mark.parametrize("m", [1.0, 3.5, 7.0, 21.0])
def test_favourites_get_more_confident_later_in_the_season(m):
    """The same predicted margin means more in week 10 than in week 2."""
    assert win_probability(m, 9, CFG) > win_probability(m, 1, CFG)


def test_win_probability_is_monotone_in_margin():
    ps = [win_probability(m, 5, CFG) for m in range(-40, 41, 5)]
    assert all(b > a for a, b in zip(ps, ps[1:]))


def test_prob_scale_survives_a_degenerate_config():
    """A tuned.json full of nonsense must not produce a NaN probability."""
    bad = RatingConfig(prob_scale_a=-5.0, prob_scale_b=-5.0)
    for g in (0, 1, 10):
        p = win_probability(7.0, g, bad)
        assert 0.0 < p < 1.0


def test_a_stand_in_opponent_is_never_more_confident_than_a_rated_one():
    """The inversion this flag exists to fix.

    Both an unrated stand-in and a week-1 team arrive at zero games played.
    The flat squash_scale (9.0) is *steeper* than the fitted curve at one game
    (10.8), so sharing that path made a prediction against an opponent nobody
    has rated more confident than one against a team with a game behind it.
    """
    for m in (7.0, 14.0, 21.0):
        stand_in = win_probability(m, 0, CFG, stand_in=True)
        for g in range(0, 11):
            assert stand_in <= win_probability(m, g, CFG), (m, g)


def test_the_stand_in_scale_is_the_flattest_the_curve_may_reach():
    assert prob_scale(0, CFG, stand_in=True) == pytest.approx(CFG.prob_scale_max)
    # and it does not depend on the games count it is handed
    assert prob_scale(9, CFG, stand_in=True) == prob_scale(0, CFG, stand_in=True)


def test_the_stand_in_flag_does_not_disturb_ordinary_predictions():
    for g in (0, 1, 5, 10):
        assert win_probability(12.0, g, CFG) == win_probability(12.0, g, CFG,
                                                                stand_in=False)


def test_a_stand_in_probability_is_still_a_probability():
    for m in (-60.0, 0.0, 60.0):
        p = win_probability(m, 0, CFG, stand_in=True)
        assert 0.0 < p < 1.0
    assert win_probability(0.0, 0, CFG, stand_in=True) == pytest.approx(0.5)


# --------------------------------------------- rating difference -> margin

def test_the_rating_scale_is_not_an_expected_margin_by_default():
    """margin_scale defaults to 1.0, i.e. uncalibrated, so an old tuned.json
    keeps the previous behaviour rather than silently changing every number."""
    assert RatingConfig().margin_scale == 1.0
    assert expected_margin(14.0, RatingConfig()) == 14.0


def test_calibration_scales_the_margin():
    cfg = RatingConfig(margin_scale=1.4708)
    assert expected_margin(14.0, cfg) == pytest.approx(20.59, abs=0.01)
    assert expected_margin(-14.0, cfg) == pytest.approx(-20.59, abs=0.01)


def test_calibration_preserves_sign_and_order():
    cfg = RatingConfig(margin_scale=1.4708)
    assert expected_margin(0.0, cfg) == 0.0
    vals = [expected_margin(m, cfg) for m in range(-30, 31, 5)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_the_margin_scale_must_not_touch_the_probability():
    """The trap this separation exists to prevent.

    prob_scale was fitted against the RAW rating difference and is calibrated
    there. Feeding it the calibrated margin instead would shift every
    probability, and both quantities are called "the margin", so the mistake
    is easy and invisible.
    """
    raw = 14.0
    plain = RatingConfig()
    calibrated = RatingConfig(margin_scale=1.4708)
    # Same raw input, same probability, whatever margin_scale says.
    for g in (0, 1, 5, 10):
        assert win_probability(raw, g, plain) == win_probability(raw, g, calibrated)
    # And the calibrated margin is emphatically NOT the right input.
    assert (win_probability(expected_margin(raw, calibrated), 5, calibrated)
            != win_probability(raw, 5, calibrated))
