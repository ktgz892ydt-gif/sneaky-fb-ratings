"""
Monte Carlo over the rest of the regular season.

What this produces, and why it needs simulating at all
-----------------------------------------------------
The question is "what are the odds this team makes the playoffs". Harbin decides
that, and Harbin cannot answer it: it is a backward-looking reward with no
opinion about who wins on Friday. So the odds are not computable from Harbin. A
forecast is needed, and that is what the board's own rating supplies.

The division of labour is deliberate and it is the whole design:

    the RULE is Harbin, unaltered, so the answer is defensible to anyone who
    follows Ohio football;

    the FORECAST is Alex Points, because nothing in the official system can
    forecast anything.

A closed form will not do, for a reason specific to Harbin: Level 2 points come
from *your opponents' wins*. One Friday result moves the qualifier for dozens of
teams at once, and the teams competing for a regional spot are exactly the teams
whose fates are entangled that way. Simulating the season keeps those
correlations; multiplying independent probabilities destroys them.

Cost
----
Vectorised over simulations. Every per-simulation quantity is a sparse matrix
applied to a (games x sims) or (teams x sims) dense block, so 10,000 seasons is
a handful of matrix products rather than 10,000 passes over the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from harbin import DIVISION_POINTS, LAST_REGULAR_WEEK, OUT_OF_STATE_POINTS


@dataclass
class SeasonSim:
    """The finished simulation, kept whole so it can be asked new questions.

    `summary` is the per-team distribution the page reads. The arrays behind it
    are retained because every what-if is already contained in them:

        P(playoffs | we beat Canfield)
          = of the simulated seasons in which we beat Canfield,
            the share in which we also qualified

    That is the exact conditional under the model, and it costs a boolean mask
    -- no re-simulation. Re-running instead would mean two fresh simulations
    per team per remaining fixture, about 12,600 runs, which at 0.9s each is
    over three hours to answer a question the existing sample already answers.
    """
    summary: dict
    team_ids: list
    won: np.ndarray          # (remaining games x sims) did the HOME team win
    qualified: np.ndarray    # (teams x sims) did this team make the playoffs
    home: np.ndarray         # team index of the home side, per remaining game
    away: np.ndarray
    n_sims: int

    def index(self):
        return {t: i for i, t in enumerate(self.team_ids)}


def _division_values(team_ids, teams, out_of_state=OUT_OF_STATE_POINTS):
    return np.array([DIVISION_POINTS.get(teams[t].division, out_of_state)
                     for t in team_ids], dtype=np.float64)


def _scatter(rows, n_teams, n_cols):
    """Sparse (n_teams x n_cols) with a 1 at (rows[k], k). Sums by team."""
    return sparse.csr_matrix(
        (np.ones(n_cols), (rows, np.arange(n_cols))), shape=(n_teams, n_cols))


def simulate_season(team_ids, teams, played, remaining, probs, per_region,
                    n_sims=10000, seed=20260826, last_week=LAST_REGULAR_WEEK):
    """Run `n_sims` seasons and report what happened across them.

    played     -- completed games, resolved, with scores
    remaining  -- unplayed regular-season fixtures as (home_idx, away_idx)
    probs      -- P(home wins) for each remaining fixture, from the rating model
    per_region -- how many qualify (OHSAA: 12 since 2025, 16 before)

    The seed is fixed so a build reproduces. Two runs of build.py must produce
    byte-identical output or the deterministic-build check is worthless.
    """
    n = len(team_ids)
    idx = {t: i for i, t in enumerate(team_ids)}
    value = _division_values(team_ids, teams)
    rng = np.random.default_rng(seed)

    # ---- what is already settled ------------------------------------------
    fixed_l1 = np.zeros(n)
    played_count = np.zeros(n)
    beat_rows, beat_cols = [], []          # winner, loser  (for level 2)
    fixed_wins = np.zeros(n)
    for g in played:
        if g.get("week", 1) > last_week:
            continue
        h, a = idx[g["home"]], idx[g["away"]]
        played_count[h] += 1
        played_count[a] += 1
        m = g["home_score"] - g["away_score"]
        if m == 0:
            continue
        w, l = (h, a) if m > 0 else (a, h)
        fixed_l1[w] += value[l]
        fixed_wins[w] += 1
        beat_rows.append(w)
        beat_cols.append(l)
    # Who-beat-whom among completed games, as a matrix: (S_fixed @ L1)[t] is the
    # level 2 contribution t earns from opponents it has already beaten. L1 is
    # per-simulation, so this cannot be folded into fixed_l1.
    s_fixed = sparse.csr_matrix(
        (np.ones(len(beat_rows)), (beat_rows, beat_cols)), shape=(n, n))

    if len(remaining) == 0:
        home = away = np.zeros(0, dtype=int)
        p = np.zeros(0)
    else:
        home = np.array([idx[h] for h, _ in remaining])
        away = np.array([idx[a] for _, a in remaining])
        p = np.asarray(probs, dtype=float)
    g_rem = len(home)
    for i in (home, away):
        np.add.at(played_count, i, 1)

    s_home = _scatter(home, n, g_rem)
    s_away = _scatter(away, n, g_rem)

    # ---- simulate ----------------------------------------------------------
    # Home wins, as (games x sims). Everything below is a product against this.
    won = rng.random((g_rem, n_sims)) < p[:, None] if g_rem else np.zeros((0, n_sims), bool)
    won = won.astype(np.float64)
    lost = 1.0 - won

    # Level 1: fixed part, plus the value of whoever you beat this week.
    l1 = fixed_l1[:, None] + s_home @ (value[away][:, None] * won) \
                           + s_away @ (value[home][:, None] * lost)

    # Level 2: for every team you beat, their level 1. Split the same way.
    l2 = s_fixed @ l1
    if g_rem:
        l2 = l2 + s_home @ (l1[away] * won) + s_away @ (l1[home] * lost)

    divisor = np.where(played_count > 0, played_count, 1.0)[:, None]
    harbin = (l1 + l2) / divisor

    wins = fixed_wins[:, None] + s_home @ won + s_away @ lost
    summary, qualified = _summarise(team_ids, teams, harbin, wins, per_region,
                                    n_sims, rng)
    return SeasonSim(summary=summary, team_ids=list(team_ids),
                     won=won.astype(bool), qualified=qualified,
                     home=home, away=away, n_sims=n_sims)


# Harbin values are sums of halves divided by a game count of at most 16, so
# two teams that are not exactly level differ by at least 0.5/16**2, about
# 0.002. A jitter six orders of magnitude below that can never reorder teams
# who genuinely differ, and among teams who are exactly level it is a fair coin
# -- which is what OHSAA does at a tied cut line, and what the alternative was
# not: ranking by array position handed every tie to whichever team sorted
# first by id, so an alphabetically early team qualified from every tie in
# every simulation. On four identical teams that skewed the odds from 50% to a
# 40%-61% spread.
TIEBREAK_JITTER = 1e-9


def _summarise(team_ids, teams, harbin, wins, per_region, n_sims,
               rng=None):
    """Turn the raw (teams x sims) blocks into per-team distributions."""
    n = len(team_ids)
    seed_rank = np.full((n, n_sims), 0, dtype=np.int16)

    by_region = {}
    for i, t in enumerate(team_ids):
        tm = teams[t]
        if tm.in_ohio and tm.region is not None:
            by_region.setdefault(tm.region, []).append(i)

    for members in by_region.values():
        rows = np.array(members)
        block = harbin[rows]                      # (members x sims)
        # Rank 1 = highest Harbin, ties broken by coin flip -- see the note on
        # TIEBREAK_JITTER. The jitter is drawn from the run's seeded generator,
        # so this stays reproducible.
        block = block + rng.random(block.shape) * TIEBREAK_JITTER
        order = np.argsort(-block, axis=0, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order,
                          np.arange(1, len(rows) + 1)[:, None].repeat(n_sims, 1),
                          axis=0)
        seed_rank[rows] = ranks

    qualified = np.zeros((n, n_sims), dtype=bool)
    out = {}
    for i, t in enumerate(team_ids):
        tm = teams[t]
        if not (tm.in_ohio and tm.region is not None):
            continue
        r = seed_rank[i]
        qualified[i] = r <= per_region
        w = wins[i]
        made = float((r <= per_region).mean())
        rec = np.bincount(np.rint(w).astype(int), minlength=17)[:17]
        out[t] = {
            "playoffOdds": round(made, DP),
            "byeOdds": round(float((r <= 4).mean()), DP),
            "topSeedOdds": round(float((r == 1).mean()), DP),
            "meanSeed": round(float(r.mean()), 2),
            "medianSeed": int(np.median(r)),
            "meanHarbin": round(float(harbin[i].mean()), 3),
            "meanWins": round(float(w.mean()), 2),
            # P(finishing with exactly k wins), k = 0..10, as whole percents.
            # Trimmed to the range that ever happens so the payload stays small.
            "winDist": _trim(rec / n_sims),
        }
    return out, qualified


def _trim(dist):
    nz = np.nonzero(dist > 0.0005)[0]
    if not len(nz):
        return {}
    return {int(k): round(float(dist[k]), DP) for k in range(nz[0], nz[-1] + 1)}


# ---------------------------------------------------------------------------
# What-ifs
# ---------------------------------------------------------------------------
#
# Two questions, and the second is the one nothing else on the internet answers
# for Ohio high school football:
#
#   1. What do MY remaining games do to my odds?
#   2. Which games that I am NOT playing in do the most to my odds?
#
# (2) exists because of how Harbin is built. Level 2 pays you for your
# opponents' wins, so a team you already beat winning again lifts you; and the
# twelve places in a region are contested, so a rival losing lifts you too.
# Those two channels pull in opposite directions and their net effect is not
# something anyone can work out in their head. The simulation already knows.
#
# Both are read off the finished sample rather than re-simulated. Conditioning
# is exact, not an approximation of re-running: the seasons in which we beat
# Canfield ARE a fair sample of the seasons in which we beat Canfield.

# Below this many simulated seasons on the thinner side of a split, the
# conditional is too noisy to publish. At 10,000 sims a 300-season branch gives
# a standard error near 3 points, which is about the resolution the page claims
# anyway; anything thinner is mostly sampling noise and is dropped rather than
# shown with false precision.
MIN_BRANCH = 300

# Simulation output is rounded to this many decimals. Not a size trick: with
# 10,000 seasons the standard error on a probability near a half is
# sqrt(0.25/10000) = 0.005, so the fourth decimal is noise and publishing it
# claims fifty times the precision the method has. Three decimals is still
# finer than the page displays.
DP = 3


def _conditional(qualified_row, mask, n_sims):
    """P(qualify | mask) and P(qualify | not mask), or None if a branch is thin."""
    n_true = int(mask.sum())
    n_false = n_sims - n_true
    if n_true < MIN_BRANCH or n_false < MIN_BRANCH:
        return None
    return (float((qualified_row & mask).sum()) / n_true,
            float((qualified_row & ~mask).sum()) / n_false,
            n_true)


def own_game_swings(sim, team_ids=None):
    """-> {team_id: [{gameIndex, oddsIfWin, oddsIfLose, swing}]}.

    One entry per remaining fixture the team plays in. `won` records whether the
    HOME side won, so for a team playing away the mask is inverted -- getting
    that backwards would silently report every road game upside-down.
    """
    wanted = set(team_ids) if team_ids is not None else set(sim.team_ids)
    by_team = {}
    for k in range(len(sim.home)):
        for side, team_i in (("home", sim.home[k]), ("away", sim.away[k])):
            t = sim.team_ids[team_i]
            if t not in wanted:
                continue
            mask = sim.won[k] if side == "home" else ~sim.won[k]
            got = _conditional(sim.qualified[team_i], mask, sim.n_sims)
            if got is None:
                continue
            win, lose, n = got
            by_team.setdefault(t, []).append({
                "g": k,
                "oddsIfWin": round(win, DP),
                "oddsIfLose": round(lose, DP),
                "swing": round(win - lose, DP),
            })
    for t in by_team:
        by_team[t].sort(key=lambda r: -abs(r["swing"]))
    return by_team


def scoreboard_watch(sim, teams, top_n=3):
    """-> {team_id: [{gameIndex, rooting, swing}]} for games the team is not in.

    Restricted to fixtures involving a team from the same region, because those
    are the ones that can plausibly move a regional cut line, and because
    scoring every team against every fixture would be 700 x 3,000 conditionals
    for no gain.

    Vectorised per region: one matrix product gives every member's conditional
    against every candidate fixture at once.
    """
    idx = sim.index()
    members = {}
    for t in sim.team_ids:
        tm = teams[t]
        if tm.in_ohio and tm.region is not None:
            members.setdefault(tm.region, []).append(idx[t])

    out = {}
    for region, rows in members.items():
        rows = np.array(rows)
        in_region = np.zeros(len(sim.team_ids), dtype=bool)
        in_region[rows] = True
        # fixtures with at least one team from this region
        cand = np.where(in_region[sim.home] | in_region[sim.away])[0]
        if not len(cand):
            continue

        w = sim.won[cand].astype(np.float32)              # (games x sims)
        n_home = w.sum(axis=1)
        ok = (n_home >= MIN_BRANCH) & ((sim.n_sims - n_home) >= MIN_BRANCH)
        cand, w, n_home = cand[ok], w[ok], n_home[ok]
        if not len(cand):
            continue

        q = sim.qualified[rows].astype(np.float32)        # (members x sims)
        both = q @ w.T                                     # home won AND qualified
        qtot = q.sum(axis=1)[:, None]
        p_home = both / n_home[None, :]
        p_away = (qtot - both) / (sim.n_sims - n_home)[None, :]
        swing = p_home - p_away                            # (members x games)

        for r, team_i in enumerate(rows):
            t = sim.team_ids[team_i]
            # A game this team plays in is not scoreboard watching.
            mine = (sim.home[cand] == team_i) | (sim.away[cand] == team_i)
            s = np.where(mine, 0.0, swing[r])
            order = np.argsort(-np.abs(s))[:top_n]
            picks = [{
                "g": int(cand[j]),
                # Positive swing means the HOME side winning helps this team.
                "rooting": "home" if s[j] > 0 else "away",
                "swing": round(float(abs(s[j])), DP),
            } for j in order if abs(s[j]) >= 0.005]
            if picks:
                out[t] = picks
    return out
