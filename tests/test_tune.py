"""
Constant selection.

The rule being tested is the one that decides which configurations count as
statistically tied with the best. Getting it wrong is invisible: it does not
crash, it does not produce a NaN, it just quietly hands the conservatism
tie-break a pool of the wrong size and the published constants drift.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from tune import tied_with_best  # noqa: E402

RNG = np.random.default_rng(20260825)
N = 4000

# Per-game log loss is dominated by the game, not the configuration: a
# coin-flip upset scores badly under all of them. That shared component is
# exactly what pairing removes.
GAME_NOISE = RNG.gamma(shape=2.0, scale=0.25, size=N)


def test_the_best_always_ties_with_itself():
    keep, ses = tied_with_best([GAME_NOISE, GAME_NOISE + 0.05])
    assert 0 in keep
    assert ses[keep.index(0)] == 0.0


def test_an_identical_configuration_ties():
    keep, _ = tied_with_best([GAME_NOISE, GAME_NOISE.copy()])
    assert keep == [0, 1]


def test_a_configuration_that_differs_only_by_noise_ties():
    """Same expected loss, disagreeing game by game. Must not be split off."""
    jitter = RNG.normal(0.0, 0.02, size=N)
    other = GAME_NOISE + jitter - jitter.mean()
    keep, _ = tied_with_best([GAME_NOISE, other])
    assert 1 in keep


def test_a_consistently_worse_configuration_is_rejected():
    """Small penalty, but the same sign on every game. That is real."""
    keep, _ = tied_with_best([GAME_NOISE, GAME_NOISE + 0.004])
    assert keep == [0]


def test_pairing_rejects_what_the_marginal_rule_would_admit():
    """The whole point of the change, stated as a test.

    A configuration 0.004 nats/game worse on every single game is far inside
    the marginal standard error of the winning score -- which is set by how
    much football games vary -- and far outside the paired one.
    """
    worse = GAME_NOISE + 0.004
    marginal_se = GAME_NOISE.std(ddof=1) / np.sqrt(N)
    penalty = worse.mean() - GAME_NOISE.mean()
    assert penalty < marginal_se, "the old rule would have called this a tie"
    assert tied_with_best([GAME_NOISE, worse])[0] == [0]


def test_the_paired_error_is_smaller_than_the_marginal_one():
    jitter = RNG.normal(0.0, 0.02, size=N)
    _, ses = tied_with_best([GAME_NOISE, GAME_NOISE + jitter - jitter.mean()])
    marginal_se = GAME_NOISE.std(ddof=1) / np.sqrt(N)
    assert ses[-1] < marginal_se


@pytest.mark.parametrize("n", [2, 5])
def test_a_degenerate_evaluation_set_does_not_explode(n):
    vecs = [np.full(n, 0.5), np.full(n, 0.5)]
    keep, ses = tied_with_best(vecs)
    assert keep == [0, 1]
    assert all(np.isfinite(s) for s in ses)
