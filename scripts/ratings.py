"""
Rating models for Ohio high school football.

Three models are fit over the same game graph:

  bt_margin  -- the headline number. A Bradley-Terry model in which each game
                contributes a *fractional* win derived from a squashed point
                margin, so a 3-point win and a 56-point win are both informative
                but the blowout is not worth 18x the nailbiter.

  bt_binary  -- textbook Bradley-Terry on wins and losses only. Ignores margin
                entirely. Immune to score-running, blind to how close it was.

  massey     -- ridge-regularized least squares on raw point margin. Uses all
                the margin information and none of the diminishing returns.

All three are regularized toward zero with a fixed "prior games" strength. This
matters more than it sounds: it is what makes the models solvable at all in
Week 1, when the game graph is ~500 disconnected components and the unpenalized
maximum-likelihood estimate does not exist (every team is 1-0 or 0-1, which is
textbook perfect separation, and the estimate runs off to +/- infinity).

Because the prior is a fixed number of pseudo-games rather than a hand-tuned
per-week schedule, shrinkage decays on its own as real games accumulate. In
Week 1 the prior is roughly as strong as the evidence. By Week 9 it is noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RatingConfig:
    # Logistic squash scale, in points. Controls how fast margin saturates into
    # a fractional win. At scale=9: 3pts -> .58, 7 -> .68, 14 -> .83,
    # 21 -> .91, 35 -> .98, 56 -> .998.
    #
    # This is the single most consequential knob in the model. Too small and
    # every win becomes a 1.0 and you have thrown away margin; too large and
    # you are back to rewarding 63-0.
    squash_scale: float = 9.0

    # Hard ceiling on margin before squashing. The squash already saturates, so
    # this is belt-and-braces against typos and running clock oddities.
    margin_cap: float = 49.0

    # Strength of the shrinkage prior, expressed in pseudo-games against a
    # league-average opponent. Fixed, not scheduled: real games outgrow it.
    prior_games: float = 1.5

    # How much of the *measured* division ladder to apply to a team's starting
    # point. 1.0 uses it in full, 0.0 ignores divisions entirely. This is a
    # dial on a number estimated from past cross-division results, not on an
    # assumption about enrollment -- and tune.py fits it like any other.
    division_weight: float = 1.0

    # Home field advantage is fitted, not assumed.
    fit_hfa: bool = True

    # L-BFGS convergence
    tol: float = 1e-9
    max_iter: int = 800


# ---------------------------------------------------------------------------
# Squash
# ---------------------------------------------------------------------------

def squash(margin: np.ndarray, scale: float, cap: float) -> np.ndarray:
    """Map a point margin to a fractional win in (0, 1).

    Symmetric about zero: squash(-m) == 1 - squash(m), and squash(0) == 0.5.
    """
    m = np.clip(np.asarray(margin, dtype=float), -cap, cap)
    return 1.0 / (1.0 + np.exp(-m / scale))


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def _design(n_teams: int, home_idx, away_idx, neutral) -> sparse.csr_matrix:
    """Sparse design matrix, one row per game.

    Column j is +1 for the home team, -1 for the away team. The final column is
    the home-field indicator (0 on neutral fields).
    """
    n = len(home_idx)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2] = home_idx
    cols[1::2] = away_idx
    vals = np.empty(2 * n)
    vals[0::2] = 1.0
    vals[1::2] = -1.0

    X = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n_teams + 1)).tocsr()

    hfa_col = sparse.coo_matrix(
        (np.where(np.asarray(neutral, dtype=bool), 0.0, 1.0), (np.arange(n), np.full(n, n_teams))),
        shape=(n, n_teams + 1),
    ).tocsr()
    return (X + hfa_col).tocsr()


# ---------------------------------------------------------------------------
# Bradley-Terry (works for both binary and fractional targets)
# ---------------------------------------------------------------------------

def fit_bradley_terry(
    n_teams: int,
    home_idx,
    away_idx,
    y,
    neutral,
    cfg: RatingConfig,
    weights=None,
    prior=None,
):
    """Fit ratings by penalized logistic regression with continuous targets.

    Minimizes cross-entropy + ridge. Convex, so the ridge term guarantees a
    unique finite solution even when the game graph is disconnected -- which in
    Week 1 it emphatically is.

    Returns (ratings_in_logits, hfa_in_logits).
    """
    y = np.asarray(y, dtype=float)
    n_games = len(y)
    w = np.ones(n_games) if weights is None else np.asarray(weights, dtype=float)

    X = _design(n_teams, home_idx, away_idx, neutral)

    # Ridge penalty. Each game contributes ~0.25 to the Hessian per team on the
    # logit scale, so prior_games pseudo-games is 0.25 * prior_games. The HFA
    # parameter is deliberately left unpenalized.
    lam = 0.25 * cfg.prior_games
    penalty = np.full(n_teams + 1, lam)
    penalty[-1] = 0.0
    if not cfg.fit_hfa:
        penalty[-1] = 1e9  # effectively pins HFA at zero

    # Shrink toward the prior rather than toward zero. With no prior supplied
    # this is identical to before -- every team starts at league average, which
    # is exactly the assumption that makes Week 1 rank teams by margin alone.
    centre = np.zeros(n_teams + 1)
    if prior is not None:
        centre[:n_teams] = prior

    def objective(theta):
        z = X @ theta
        # log(1 + exp(z)) computed stably
        logsig = -np.logaddexp(0.0, -z)
        log1msig = -np.logaddexp(0.0, z)
        nll = -np.sum(w * (y * logsig + (1.0 - y) * log1msig))
        delta = theta - centre
        nll += 0.5 * np.sum(penalty * delta * delta)

        p = np.exp(logsig)
        grad = X.T @ (w * (p - y)) + penalty * delta
        return nll, grad

    theta0 = centre.copy()
    res = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": cfg.max_iter, "ftol": cfg.tol, "gtol": 1e-8},
    )

    theta = res.x
    ratings = theta[:n_teams]
    hfa = theta[n_teams]

    # Ridge already prefers the centered solution, but pin it exactly so the
    # scale is interpretable and stable week to week.
    ratings = ratings - ratings.mean()
    return ratings, hfa, res


# ---------------------------------------------------------------------------
# Massey
# ---------------------------------------------------------------------------

def fit_massey(n_teams: int, home_idx, away_idx, margin, neutral, cfg: RatingConfig,
               prior=None):
    """Ridge-regularized least squares on point margin. Ratings are in points."""
    X = _design(n_teams, home_idx, away_idx, neutral)
    m = np.clip(np.asarray(margin, dtype=float), -cfg.margin_cap, cfg.margin_cap)

    lam = cfg.prior_games
    penalty = np.full(n_teams + 1, lam)
    penalty[-1] = 0.0
    if not cfg.fit_hfa:
        penalty[-1] = 1e9

    centre = np.zeros(n_teams + 1)
    if prior is not None:
        centre[:n_teams] = prior

    # With no home games at all, the home-field column is entirely zero and
    # the normal equations are singular. Pin the term rather than solving a
    # degenerate system.
    if not np.any(~np.asarray(neutral, dtype=bool)):
        penalty[-1] = 1e9

    A = (X.T @ X).tocsc() + sparse.diags(penalty).tocsc()
    b = X.T @ m + penalty * centre
    theta = spsolve(A, b)

    ratings = np.asarray(theta[:n_teams])
    hfa = float(theta[n_teams])
    ratings = ratings - ratings.mean()
    return ratings, hfa


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class RatingResult:
    team_ids: list
    bt_margin: np.ndarray      # points scale
    bt_binary: np.ndarray      # points scale
    massey: np.ndarray         # points scale
    hfa_margin: float          # points
    hfa_binary: float          # points
    hfa_massey: float          # points
    wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    games: np.ndarray
    sos: np.ndarray            # mean opponent bt_margin rating
    point_diff: np.ndarray
    converged: bool = True
    notes: list = field(default_factory=list)


def rate(team_ids, games, cfg: RatingConfig | None = None, priors=None) -> RatingResult:
    """Fit all three models.

    `games` is a sequence of dicts with keys:
        home, away        -- team ids present in team_ids
        home_score, away_score
        neutral           -- optional bool, default False
    """
    cfg = cfg or RatingConfig()
    index = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)

    home_idx, away_idx, hs, as_, neutral = [], [], [], [], []
    for g in games:
        home_idx.append(index[g["home"]])
        away_idx.append(index[g["away"]])
        hs.append(float(g["home_score"]))
        as_.append(float(g["away_score"]))
        neutral.append(bool(g.get("neutral", False)))

    home_idx = np.array(home_idx)
    away_idx = np.array(away_idx)
    hs = np.array(hs)
    as_ = np.array(as_)
    neutral = np.array(neutral)
    margin = hs - as_

    # Priors arrive in points (the scale people read); the logit models work on
    # the squashed scale, so divide through.
    prior_pts = np.zeros(n)
    if priors:
        for i, t in enumerate(team_ids):
            prior_pts[i] = float(priors.get(t, 0.0))
        prior_pts -= prior_pts.mean()
    prior_logit = prior_pts / cfg.squash_scale

    # --- headline model: fractional wins from squashed margin
    y_frac = squash(margin, cfg.squash_scale, cfg.margin_cap)
    r_margin, hfa_margin, res = fit_bradley_terry(
        n, home_idx, away_idx, y_frac, neutral, cfg, prior=prior_logit
    )

    # --- plain Bradley-Terry on W/L
    y_bin = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))
    r_binary, hfa_binary, _ = fit_bradley_terry(
        n, home_idx, away_idx, y_bin, neutral, cfg, prior=prior_logit
    )

    # --- Massey
    r_massey, hfa_massey = fit_massey(n, home_idx, away_idx, margin, neutral, cfg,
                                      prior=prior_pts)

    # Convert logit ratings to points. A rating difference of d points is the
    # model's expected neutral-field margin, which is what makes the headline
    # number readable to someone who has never heard of Bradley-Terry.
    bt_margin_pts = r_margin * cfg.squash_scale
    bt_binary_pts = r_binary * cfg.squash_scale

    # --- records and derived quantities
    wins = np.zeros(n)
    losses = np.zeros(n)
    ties = np.zeros(n)
    pdiff = np.zeros(n)
    opp_sum = np.zeros(n)
    gcount = np.zeros(n)

    for k in range(len(margin)):
        h, a, m = home_idx[k], away_idx[k], margin[k]
        gcount[h] += 1
        gcount[a] += 1
        pdiff[h] += m
        pdiff[a] -= m
        opp_sum[h] += bt_margin_pts[a]
        opp_sum[a] += bt_margin_pts[h]
        if m > 0:
            wins[h] += 1
            losses[a] += 1
        elif m < 0:
            wins[a] += 1
            losses[h] += 1
        else:
            ties[h] += 1
            ties[a] += 1

    sos = np.divide(opp_sum, gcount, out=np.zeros_like(opp_sum), where=gcount > 0)

    return RatingResult(
        team_ids=list(team_ids),
        bt_margin=bt_margin_pts,
        bt_binary=bt_binary_pts,
        massey=r_massey,
        hfa_margin=hfa_margin * cfg.squash_scale,
        hfa_binary=hfa_binary * cfg.squash_scale,
        hfa_massey=hfa_massey,
        wins=wins,
        losses=losses,
        ties=ties,
        games=gcount,
        sos=sos,
        point_diff=pdiff,
        converged=bool(res.success),
    )


def predict_margin(result: RatingResult, home: str, away: str, neutral: bool = False) -> float:
    """Expected margin (home perspective) using the headline model."""
    idx = {t: i for i, t in enumerate(result.team_ids)}
    d = result.bt_margin[idx[home]] - result.bt_margin[idx[away]]
    return d + (0.0 if neutral else result.hfa_margin)
