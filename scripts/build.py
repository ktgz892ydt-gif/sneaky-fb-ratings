"""Run resolution + ratings and emit the JSON the site reads."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harbin import (FIRST_ROUND_BYES, LAST_REGULAR_WEEK,  # noqa: E402
                    QUALIFIERS_PER_REGION, harbin_points,
                    leans_on_out_of_state, qualifiers, validate)
from history import (build_snapshot, load as load_history,  # noqa: E402
                     record as record_snapshot, score as score_history,
                     trends)
from ratings import (RatingConfig, expected_margin, rate,  # noqa: E402
                     win_probability)
from scrape import CURRENT_SEASON  # noqa: E402
from rivals import head_to_head, load as load_rivals  # noqa: E402
from resolve import load_games, load_roster, resolve, team_identity  # noqa: E402
from simulate import (own_game_swings, scoreboard_watch,  # noqa: E402
                      simulate_season)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITE = os.path.join(ROOT, "site")


def connectivity(teams, games):
    """Diagnose whether the game graph can support a statewide rating yet.

    Paired-comparison ratings only carry meaning between teams connected by a
    chain of games. Two teams in different components have no comparable
    ratings at all -- whatever number the regularizer assigns them is the
    prior talking, not evidence. In Week 1 essentially every game is its own
    island, and pretending otherwise is the central failure mode of early
    season computer rankings.
    """
    parent = {t: t for t in teams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for g in games:
        union(g["home"], g["away"])

    comps = {}
    for t in teams:
        comps.setdefault(find(t), []).append(t)

    sizes = sorted((len(v) for v in comps.values()), reverse=True)
    n_teams = len(teams)
    largest = sizes[0] if sizes else 0

    gcount = {t: 0 for t in teams}
    for g in games:
        gcount[g["home"]] += 1
        gcount[g["away"]] += 1
    avg_games = sum(gcount.values()) / max(1, n_teams)

    frac = largest / max(1, n_teams)
    if avg_games < 2 or frac < 0.25:
        level, label = "none", "Not yet meaningful"
    elif frac < 0.60:
        level, label = "low", "Fragmented"
    elif frac < 0.90 or avg_games < 4:
        level, label = "medium", "Emerging"
    else:
        level, label = "high", "Usable"

    return {
        "components": len(comps),
        "largestComponent": largest,
        "largestComponentFrac": round(frac, 3),
        "avgGamesPerTeam": round(avg_games, 2),
        "level": level,
        "label": label,
    }


def load_schedule(path):
    """Read not-yet-played fixtures written by scrape.py.

    Deliberately separate from load_games(). These rows have no scores, and
    they must never reach resolve() or rate() -- a fixture treated as a result
    would inject a phantom 0-0 tie between two real teams, and it would do so
    invisibly, because the ratings table would still look entirely normal.
    """
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append({
                    "week": int(row["week"]),
                    "date": (row.get("date") or "").strip(),
                    "time": (row.get("time") or "").strip(),
                    # Same identity rule as completed games, or a fixture
                    # against New Jersey's Salem would be matched to Ohio's.
                    "away": team_identity(row["away"], row.get("away_state")),
                    "home": team_identity(row["home"], row.get("home_state")),
                    "neutral": bool(int(row.get("neutral") or 0)),
                })
            except (ValueError, KeyError):
                continue
    return out


def division_baseline(blob, division, cfg):
    """The measured rating of an average team in `division`.

    Used as the stand-in for an opponent we cannot rate: an out-of-state
    school that has not played anybody yet has no rating at all, and there is
    no evidence from which to invent one. Division III is the middle of the
    seven-division ladder, so it is the least-committal guess available.

    Matches how priors are composed in main(), so the number is on the same
    scale as the ratings themselves rather than merely near them.
    """
    if blob:
        eff = (blob.get("divisionEffects") or {}).get(division)
        if eff is not None:
            return float(eff) * cfg.division_weight, f"division {division} baseline"
    return None, ""


def scoring_profile(res, team_ids, shrink=4.0):
    """A conservative total-points layer for turning margins into scores.

    The rating model says who is better and by how much. A score needs one more
    number: the game's total points. Estimate that from each team's points
    scored and allowed, but shrink hard toward the season average because a
    single high-school football score is a noisy thing.
    """
    idx = {t: i for i, t in enumerate(team_ids)}
    pf = np.zeros(len(team_ids))
    pa = np.zeros(len(team_ids))
    gp = np.zeros(len(team_ids))
    total_points = 0.0

    for g in res.games:
        h, a = idx[g["home"]], idx[g["away"]]
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        pf[h] += hs
        pa[h] += as_
        gp[h] += 1
        pf[a] += as_
        pa[a] += hs
        gp[a] += 1
        total_points += hs + as_

    league_ppg = total_points / max(1.0, 2.0 * len(res.games))

    def adj(points):
        avg = np.divide(points, gp, out=np.full_like(points, league_ppg),
                        where=gp > 0)
        weight = gp / (gp + shrink)
        return weight * (avg - league_ppg)

    return {
        "leagueTotal": 2.0 * league_ppg,
        "offense": adj(pf),
        "defense": adj(pa),
    }


def expected_total_points(home, away, profile, idx):
    """Projected combined score before the margin is applied."""
    total = float(profile["leagueTotal"])
    for tid, kind in ((home, "offense"), (away, "offense"),
                      (home, "defense"), (away, "defense")):
        if tid in idx:
            total += float(profile[kind][idx[tid]])
    return max(10.0, min(100.0, total))


# Scores a football team essentially never finishes on. Measured, not guessed:
# over 37,240 real team-scores from 2023-2026 these are the only values below
# 43 occurring in under 0.4% of games.
#
#     4 -> 0.02%    5 -> 0.03%    11 -> 0.11%    1 -> 0.13%    2 -> 0.20%
#
# Everything else clears 0.4%. Football scoring is spiky -- 0, 7, 14, 6, 21,
# 28, 35 dominate -- and a projection landing between the spikes reads as
# broken to anyone who follows the sport. "Proj 48-1" is not a bold call, it
# is an impossible one.
IMPLAUSIBLE_SCORES = frozenset({1, 2, 4, 5, 11})

# How far the total may be nudged to reach a plausible pair. Six points is
# enough to always succeed across the range expected_total_points can actually
# emit (it clamps to 10-100); the fallback below exists only for inputs that
# function cannot produce.
_TOTAL_SEARCH = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6)


def projected_score(margin, total, implausible=IMPLAUSIBLE_SCORES):
    """Integer projected score from a displayed margin and expected total.

    The margin is the trusted quantity -- it comes from the fitted, calibrated
    model -- while the total is a shrunk estimate off a handful of games. So
    when the naive split lands on a scoreline football does not produce, the
    TOTAL is what moves, and the margin is held.

    Candidates are ranked by margin fidelity first and distance from the
    estimated total second, so a projection never drifts further from the
    model's own margin than it has to. Everything stays inside the tolerance
    check.py asserts.
    """
    margin = float(margin)
    total = max(float(total), abs(margin))

    def split(t):
        h = int(round(max(0.0, (t + margin) / 2.0)))
        a = int(round(max(0.0, (t - margin) / 2.0)))
        return h, a

    best = None
    for delta in _TOTAL_SEARCH:
        h, a = split(total + delta)
        if h in implausible or a in implausible:
            continue
        drift = abs((h - a) - margin)
        if drift > 1.1:                      # the invariant check.py enforces
            continue
        # A stated favourite must not be shown level, and must not be shown
        # losing. Below half a point the margin is not claiming a favourite.
        #
        # Written as two explicit comparisons rather than `(h > a) != (margin >
        # 0)`, which is subtly asymmetric: for a NEGATIVE margin a tie makes
        # both sides False, the test passes, and 22-22 was published against a
        # stated -0.5 favourite. Positive margins were fine, so the bug only
        # ever showed on away favourites.
        if abs(margin) >= 0.5:
            if margin > 0 and h <= a:
                continue
            if margin < 0 and h >= a:
                continue
        cost = (round(drift, 6), abs(delta))
        if best is None or cost < best[0]:
            best = (cost, (h, a))
    if best is not None:
        return best[1]
    # Nothing plausible within reach: keep the honest split rather than invent
    # a scoreline. Rare, and check.py still holds it to the margin.
    return split(total)


def predict_schedule(fixtures, res, result, team_ids, cfg, fallback):
    """Attach a predicted margin and win probability to each fixture.

    Fixtures are matched to rated teams BY NAME, against the table resolve()
    already built from completed games. They are not passed through resolve()
    itself: that function derives identity from results, which a fixture does
    not have, and feeding it one risks changing how real teams are resolved.

    A name shared by several rated teams is left unresolved rather than
    guessed. The resolver's contract is that it declines to guess before it
    merges, and predictions inherit it.
    """
    idx = {t: i for i, t in enumerate(team_ids)}
    scoring = scoring_profile(res, team_ids)
    fb_rating, fb_note = fallback

    by_name = {}
    for tid in team_ids:
        by_name.setdefault(res.teams[tid].name, []).append(tid)

    def rated(tid):
        i = idx[tid]
        return tid, float(result.bt_margin[i]), int(result.games[i]), "rated"

    def side(name):
        """-> (tid or None, rating or None, gamesPlayed, status)

        status is one of: rated, assumed-ohio, stand-in, ambiguous, unknown.
        """
        cands = by_name.get(name, [])
        if len(cands) == 1:
            return rated(cands[0])
        if len(cands) > 1:
            # A name can be shared by an Ohio school and a same-named school
            # from elsewhere that appeared on this scoreboard -- Marietta is
            # the real case. A fixture listed on an Ohio scoreboard against an
            # Ohio-region opponent is the Ohio school; the namesake is here
            # only because one of its games was posted.
            #
            # This picks between two already-separate entities. It does not
            # merge them, and it only fires when exactly one is in Ohio;
            # two Ohio schools sharing a name AND a city stay refused.
            ohio = [t for t in cands if res.teams[t].in_ohio]
            if len(ohio) == 1:
                tid, r, g, _ = rated(ohio[0])
                return tid, r, g, "assumed-ohio"
            return None, None, 0, "ambiguous"
        if fb_rating is None:
            return None, None, 0, "unknown"
        return None, fb_rating, 0, "stand-in"

    out = []
    for f in fixtures:
        ht, hr, hg, hstat = side(f["home"])
        at, ar, ag, astat = side(f["away"])
        row = {
            "week": f["week"],
            "date": f["date"],
            "time": f["time"],
            "home": ht,
            "homeName": f["home"],
            "away": at,
            "awayName": f["away"],
            "neutral": f["neutral"],
        }
        if hr is None or ar is None:
            # Said plainly rather than filled in with a plausible number.
            bad = [s for s in (hstat, astat) if s in ("ambiguous", "unknown")]
            row.update(predicted=False, reason=(
                "opponent's name is shared by several schools"
                if "ambiguous" in bad else "opponent has no rating yet"))
            out.append(row)
            continue

        margin = hr - ar + (0.0 if f["neutral"] else result.hfa_margin)
        # The less established of the two decides how flat the probability
        # curve should be: a rating difference is only as trustworthy as the
        # thinner of the two records behind it.
        established = min(hg, ag)
        # A stood-in opponent is not merely early-season, it is unrated. Both
        # arrive here at zero games and they get different scales -- see
        # prob_scale(). Without this a stand-in reads as more confident than a
        # real team with one game behind it.
        stand_in = "stand-in" in (hstat, astat)
        # The probability is computed from the RAW difference, which is the
        # scale prob_scale was fitted against and is well calibrated on.
        p = win_probability(margin, established, cfg, stand_in=stand_in)
        # The published margin is the calibrated one, because a rating
        # difference measurably is not an expected margin -- see margin_scale.
        # Round FIRST, then derive the score from the same number that gets
        # published. Feeding the unrounded margin here let a raw 0.45 become a
        # published 0.5, so the payload named a favourite while the projected
        # score showed a tie -- the two must be consistent by construction, not
        # by luck.
        shown = round(float(expected_margin(margin, cfg)), 1)
        # Scores are a display layer on top of that margin. The total comes from
        # shrunk scoring/allowing tendencies; the winner comes from the rating.
        total = expected_total_points(ht, at, scoring, idx)
        proj_home, proj_away = projected_score(shown, total)
        row.update(
            predicted=True,
            predictedHomeMargin=shown,
            projectedHomeScore=proj_home,
            projectedAwayScore=proj_away,
            homeWinProb=round(float(p), 3),
            favoriteName=f["home"] if margin >= 0 else f["away"],
            spread=round(abs(shown), 1),
            gamesBehind=established,
            estimated=stand_in,
            estimatedNote=(fb_note if stand_in else ""),
            assumedOhio=("assumed-ohio" in (hstat, astat)),
        )
        out.append(row)
    return out


def compact_schedule(schedule, team_ids):
    """Shrink the fixture list for transport.

    A full season is roughly 3,500 fixtures. Written out in full that is over
    a megabyte of JSON for a page that is often opened on a phone, and it is
    all redundant: a team's name, division and record are already in `teams`,
    so a fixture only needs to point at a row.

    Keys are short but named, never positional -- a positional row breaks
    silently the first time someone inserts a column.

        w  week            h  home  (index into teams, or a name string
        d  date               a  away   when the team could not be resolved)
        t  kickoff time    n  neutral site
        m  predicted margin, home perspective
        p  home win probability
        ph projected home score
        pa projected away score
        e  1 if either side used the stand-in rating
        o  1 if a shared name was read as the Ohio school
        x  reason the game could not be predicted
    """
    pos = {t: i for i, t in enumerate(team_ids)}
    out = []
    for g in schedule:
        row = {"w": g["week"],
               "h": pos.get(g["home"], g["homeName"]),
               "a": pos.get(g["away"], g["awayName"])}
        if g.get("date"):
            row["d"] = g["date"]
        if g.get("time"):
            row["t"] = g["time"]
        if g.get("neutral"):
            row["n"] = 1
        if g.get("predicted"):
            row["m"] = g["predictedHomeMargin"]
            row["p"] = g["homeWinProb"]
            row["ph"] = g["projectedHomeScore"]
            row["pa"] = g["projectedAwayScore"]
            if g.get("estimated"):
                row["e"] = 1
            if g.get("assumedOhio"):
                row["o"] = 1
        else:
            row["x"] = g.get("reason", "not predicted")
        out.append(row)
    return out


def project_records(schedule, team_ids, result):
    """Expected final record = games banked + the sum of win probabilities.

    This is an expectation, not a most-likely record: a team with three
    coin-flips left projects 1.5 wins, which is not a record anyone can
    finish with. Phase 2's Monte Carlo gives the distribution; this gives its
    mean, which is the honest one-number summary.
    """
    idx = {t: i for i, t in enumerate(team_ids)}
    exp = {t: 0.0 for t in team_ids}
    remaining = {t: 0 for t in team_ids}
    for g in schedule:
        if not g.get("predicted"):
            continue
        p = g["homeWinProb"]
        for tid, pw in ((g["home"], p), (g["away"], 1.0 - p)):
            if tid in exp:
                exp[tid] += pw
                remaining[tid] += 1
    out = {}
    for t in team_ids:
        i = idx[t]
        w, l = float(result.wins[i]), float(result.losses[i])
        # Ties are excluded from both halves rather than folded into losses.
        decided = w + l + remaining[t]
        pw = round(w + exp[t], 1)
        # Derived from the rounded wins, not rounded independently: otherwise
        # "3.5-6.5" can add up to one more game than the team will play.
        out[t] = {"remaining": remaining[t],
                  "projWins": pw,
                  "projLosses": round(decided - pw, 1)}
    return out


def main(games_path=None, roster_path=None, out_path=None, generated_at=None,
         prior_path=None, use_prior=True, schedule_path=None,
         history_path=None, record_history=True):
    # Prefer real scraped data when it exists; fall back to the checked-in
    # Week 1 fixture so the pipeline is runnable without network access.
    if games_path is None:
        scraped = os.path.join(DATA, f"games_{CURRENT_SEASON}.csv")
        games_path = scraped if os.path.exists(scraped) else os.path.join(
            DATA, f"fixture_week1_{CURRENT_SEASON}.psv"
        )
    roster_path = roster_path or os.path.join(DATA, f"roster_{CURRENT_SEASON}.csv")
    out_path = out_path or os.path.join(SITE, "ratings.json")

    # The season is whichever one these files are for -- inferred from the
    # filename so that building a past season does not mislabel its output.
    m = re.search(r"(\d{4})", os.path.basename(games_path))
    season = int(m.group(1)) if m else CURRENT_SEASON

    roster = load_roster(roster_path)
    raw_games = load_games(games_path)
    res = resolve(roster, raw_games)
    diag = connectivity(res.teams, res.games)

    team_ids = sorted(res.teams)

    # Constants fitted against past seasons by scripts/tune.py, if that has
    # been run. Otherwise the defaults, which are judgement rather than
    # measurement -- see the module docstring in tune.py.
    cfg = RatingConfig()
    tuned_meta = None
    tpath = os.path.join(DATA, "tuned.json")
    if os.path.exists(tpath):
        with open(tpath, encoding="utf-8") as fh:
            tb = json.load(fh)
        best = tb.get("best") or {}
        # The margin-to-probability curve, if this tuned.json is new enough to
        # carry one. An older file simply keeps the dataclass defaults.
        ps = tb.get("probScale") or {}
        cfg = RatingConfig(
            squash_scale=float(best.get("squash_scale", cfg.squash_scale)),
            prior_games=float(best.get("prior_games", cfg.prior_games)),
            division_weight=float(best.get("division_weight", cfg.division_weight)),
            prob_scale_a=float(ps.get("a", cfg.prob_scale_a)),
            prob_scale_b=float(ps.get("b", cfg.prob_scale_b)),
            margin_scale=float(ps.get("marginScale") or cfg.margin_scale),
        )
        tuned_meta = {k: best.get(k) for k in
                      ("squash_scale", "prior_games", "carry", "division_weight",
                       "accuracy", "logloss", "mae_margin", "n")}
        tuned_meta["tunedOn"] = tb.get("tunedOn")
        if ps:
            tuned_meta["probScaleLogloss"] = (ps.get("crossValidated") or {}).get(
                "meanLoglossFitted")

    # A preseason prior from last season, if one has been built. Teams are
    # matched on the school ID published on the ranking pages, which is stable
    # across years.
    priors, prior_meta = None, None
    ppath = prior_path or os.path.join(DATA, "prior.json")
    blob = None
    if use_prior and os.path.exists(ppath):
        try:
            with open(ppath, encoding="utf-8") as fh:
                blob = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            # An unreadable prior is a reason to warn and carry on, not to
            # abandon the week's ratings.
            print(f"warning: could not read prior at {ppath} ({exc}); "
                  f"continuing without one", file=sys.stderr)
            blob = None
    if blob:
        pmap = blob.get("prior", {})
        deffect = blob.get("divisionEffects", {}) or {}
        dw = cfg.division_weight
        priors, matched = {}, 0
        for t in team_ids:
            tm = res.teams[t]
            if not tm.in_ohio:
                continue
            key = (tm.school_id or "").strip()
            dev = pmap.get(key) if key and key in pmap else pmap.get(tm.name)
            if dev is not None:
                matched += 1
            # A team starts at its division's measured baseline, plus whatever
            # it personally earned above or below that division last season.
            # A team new to the data starts at its division's baseline alone.
            base = deffect.get(tm.division, 0.0) * dw
            priors[t] = base + (dev or 0.0)
        prior_meta = dict(blob.get("meta", {}), matched=matched,
                          divisionWeight=dw)
        if not priors:
            priors = None

    result = rate(team_ids, res.games, cfg, priors=priors)

    idx = {t: i for i, t in enumerate(team_ids)}

    # ---- the remaining schedule -------------------------------------------
    # Loaded after the fit, never before. Nothing below this line may touch
    # res.games or the rating itself.
    if schedule_path is None:
        cand = os.path.join(DATA, f"schedule_{season}.csv")
        schedule_path = cand if os.path.exists(cand) else None
    fixtures = load_schedule(schedule_path)

    # The source is an Ohio scoreboard, but it also carries border-state games
    # between two schools that are both from elsewhere -- Kentucky at Kentucky,
    # Michigan at Michigan. About a fifth of the fixture list.
    #
    # A completed one of those is worth keeping: it rates an out-of-state team
    # that some Ohio school will later play. An unplayed one carries no
    # information at all -- it is a prediction about two teams nobody here
    # follows, made from two stand-in ratings. Dropping them removes a fifth of
    # the payload and most of the "estimated" flags.
    in_ohio_names = {res.teams[t].name for t in team_ids if res.teams[t].in_ohio}
    before = len(fixtures)
    fixtures = [f for f in fixtures
                if f["home"] in in_ohio_names or f["away"] in in_ohio_names]
    dropped_foreign = before - len(fixtures)

    fallback = division_baseline(blob, "III", cfg)
    if fixtures and fallback[0] is None:
        # No prior to read a division ladder from -- use the middle of the
        # teams we did rate, which is always available and means the same
        # thing. Better a stated approximation than a silent hole.
        rated = [float(result.bt_margin[idx[t]]) for t in team_ids
                 if res.teams[t].in_ohio and result.games[idx[t]] > 0]
        if rated:
            fallback = (round(float(np.median(rated)), 2),
                        "median of rated Ohio teams")

    schedule = predict_schedule(fixtures, res, result, team_ids, cfg, fallback)
    projections = project_records(schedule, team_ids, result)

    # ---- the playoff picture ----------------------------------------------
    #
    # Two different things, kept apart on purpose. `harbin` is the OHSAA
    # qualifier computed from results that have actually happened -- the
    # standings as they stand. `sim` is 10,000 simulated finishes to the
    # regular season, each scored by that same rule, which is where the odds
    # come from. The rule is theirs; only the forecast is ours.
    harbin_now = harbin_points(res.teams, res.games)
    seeds_now = qualifiers(res.teams, harbin_now, QUALIFIERS_PER_REGION)

    published_harbin = {t: res.teams[t].harbin for t in team_ids
                        if res.teams[t].harbin is not None}
    harbin_check = validate(res.teams, res.games, published_harbin)
    harbin_approx = leans_on_out_of_state(res.teams, res.games)

    # Only unplayed REGULAR-season games decide qualification. Fixtures in
    # week 11 and beyond are playoff or out-of-state dates and must not be
    # simulated into the standings that produce the playoffs.
    sim_rem, sim_p = [], []
    for g in schedule:
        if not g.get("predicted") or g["week"] > LAST_REGULAR_WEEK:
            continue
        if g["home"] is None or g["away"] is None:
            continue
        sim_rem.append((g["home"], g["away"]))
        sim_p.append(g["homeWinProb"])
    # `sim_rows` keeps the schedule row each simulated fixture came from, so a
    # what-if can name the opponent and the week rather than an array index.
    sim_rows = []
    for g in schedule:
        if not g.get("predicted") or g["week"] > LAST_REGULAR_WEEK:
            continue
        if g["home"] is None or g["away"] is None:
            continue
        sim_rows.append(g)

    # NB: `season` in this function is the YEAR. Do not shadow it here -- doing
    # so put the simulation object into the payload's "season" field, which
    # json.dump then refused. Loud, but only because a dataclass is not
    # serialisable; an int-like object would have shipped a wrong year.
    # A finished season has nothing to simulate. Running it anyway is not
    # merely wasted work: it writes playoff odds, seed distributions and win
    # distributions for all 700 teams into a file that exists ONLY to derive
    # next season's prior, of which season_prior.py reads six fields. The
    # workflow rebuilds last season on every run, so this shipped a
    # 16,691-line diff and a 2.2 MB artifact the first time the simulator ran
    # in CI.
    if sim_rem:
        sim_season = simulate_season(team_ids, res.teams, res.games, sim_rem, sim_p,
        QUALIFIERS_PER_REGION)
        sim = sim_season.summary
    else:
        sim_season, sim = None, {}
        print("simulation     : nothing left to play; skipped", file=sys.stderr)

    # ---- what-ifs ---------------------------------------------------------
    # Read off the finished simulation by conditioning, never re-simulated:
    # two fresh runs per team per fixture would be ~12,600 simulations.
    pos_of_team = {t: i for i, t in enumerate(team_ids)}
    # Only teams that can actually qualify. An out-of-state team is never in a
    # region, so its conditional odds are trivially zero either way -- rows that
    # say nothing, for a third of the payload.
    can_qualify = [t for t in team_ids
                   if res.teams[t].in_ohio and res.teams[t].region is not None]
    swings = own_game_swings(sim_season, can_qualify) if sim_season else {}
    watches = scoreboard_watch(sim_season, res.teams) if sim_season else {}

    def _row(k):
        return sim_rows[k]

    what_if = {}
    for t, entries in swings.items():
        rows = []
        for e in entries:
            g = _row(e["g"])
            opp = g["away"] if g["home"] == t else g["home"]
            rows.append({"w": g["week"],
                         "o": pos_of_team.get(opp, opp),
                         "h": 1 if g["home"] == t else 0,
                         "win": e["oddsIfWin"], "lose": e["oddsIfLose"]})
        rows.sort(key=lambda r: r["w"])
        if rows:
            what_if[t] = rows

    watch_out = {}
    for t, picks in watches.items():
        rows = []
        for pick in picks:
            g = _row(pick["g"])
            rows.append({"w": g["week"],
                         "h": pos_of_team.get(g["home"], g["homeName"]),
                         "a": pos_of_team.get(g["away"], g["awayName"]),
                         "for": pick["rooting"], "sw": pick["swing"]})
        if rows:
            watch_out[t] = rows

    # Rank Ohio teams only, and only those that have actually played.
    #
    # A team with no result this season still gets a rating -- the prior and
    # the regularizer see to that -- but it has not earned a place in the
    # standings the way a team that took the field has. Ranking them anyway
    # buries real teams beneath placeholders. They stay in the table, carry
    # their prior-based rating, and are labelled.
    ohio = [t for t in team_ids if res.teams[t].in_ohio]
    played = [t for t in ohio if result.games[idx[t]] > 0]
    ohio_sorted = sorted(played, key=lambda t: -result.bt_margin[idx[t]])
    rank_of = {t: i + 1 for i, t in enumerate(ohio_sorted)}
    unplayed = [t for t in ohio if result.games[idx[t]] == 0]

    # Division and region ranks
    div_rank, reg_rank = {}, {}
    for key, getter in (("division", "division"), ("region", "region")):
        buckets = {}
        for t in ohio_sorted:
            v = getattr(res.teams[t], getter)
            if v is None:
                continue
            buckets.setdefault(v, []).append(t)
        target = div_rank if key == "division" else reg_rank
        for v, lst in buckets.items():
            for i, t in enumerate(lst):
                target[t] = i + 1

    pos_of = {t: i for i, t in enumerate(team_ids)}

    rows = []
    for t in team_ids:
        i = idx[t]
        tm = res.teams[t]
        rows.append(
            {
                "id": t,
                "name": tm.name,
                "schoolId": tm.school_id,
                "city": tm.city,
                "division": tm.division,
                "region": tm.region,
                "inOhio": tm.in_ohio,
                "ambiguous": tm.ambiguous,
                "note": tm.note,
                "rank": rank_of.get(t),
                "unplayed": bool(tm.in_ohio and result.games[i] == 0),
                "divRank": div_rank.get(t),
                "regRank": reg_rank.get(t),
                "rating": round(float(result.bt_margin[i]), 2),
                "btBinary": round(float(result.bt_binary[i]), 2),
                "massey": round(float(result.massey[i]), 2),
                "harbin": tm.harbin,
                "w": int(result.wins[i]),
                "l": int(result.losses[i]),
                "t": int(result.ties[i]),
                "games": int(result.games[i]),
                "sos": round(float(result.sos[i]), 2),
                "pd": int(result.point_diff[i]),
                # The qualifier as it stands, and the seed it currently earns.
                "harbinNow": round(float(harbin_now.get(t, 0.0)), 3),
                "harbinApprox": bool(harbin_approx.get(t, False)),
                "seedNow": seeds_now.get(t),
                **projections.get(t, {}),
                **sim.get(t, {}),
                # What each remaining game is worth, and which games elsewhere
                # to watch. See whatIfCols in the payload.
                **({"whatIf": what_if[t]} if t in what_if else {}),
                **({"watch": watch_out[t]} if t in watch_out else {}),
            }
        )

    # ---- the track record ------------------------------------------------
    #
    # Capture what the board is claiming BEFORE the games are played, then
    # grade every earlier capture against what has since happened. The log is
    # append-only and idempotent per (season, week): the first look at a week
    # is the prediction, and a later build has seen more of the season, so
    # re-recording would quietly swap a forecast for hindsight.
    hpath = history_path if history_path is not None else os.path.join(DATA, "history.jsonl")
    weeks_loaded = sorted({g["week"] for g in res.games})
    through_week = max(weeks_loaded) if weeks_loaded else 0

    scorecard, trend, headtohead = None, {}, None
    if hpath:
        snaps = load_history(hpath)
        results_by_season = {season: {
            (g.get("week", 1), res.teams[g["home"]].name, res.teams[g["away"]].name):
                g["home_score"] - g["away_score"] for g in res.games}}
        # Past seasons are scored from their own committed scores.
        for other in sorted({s.get("season") for s in snaps} - {season}):
            gp = os.path.join(DATA, f"games_{other}.csv")
            rp = os.path.join(DATA, f"roster_{other}.csv")
            if not (os.path.exists(gp) and os.path.exists(rp)):
                continue
            o = resolve(load_roster(rp), load_games(gp))
            results_by_season[other] = {
                (g.get("week", 1), o.teams[g["home"]].name, o.teams[g["away"]].name):
                    g["home_score"] - g["away_score"] for g in o.games}
        scorecard = score_history(snaps, results_by_season)
        # Another public forecaster, scored on the games we both predicted.
        # Only the intersection, and the difference tested paired -- see the
        # module docstring in rivals.py for why headline figures cannot be
        # compared directly.
        rpath = os.path.join(DATA, "rivals.jsonl")
        headtohead = head_to_head(load_rivals(rpath), snaps, results_by_season)
        trend = trends(snaps, season, [t["name"] for t in rows if t.get("inOhio")])

    payload = {
        "generatedAt": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "weeksLoaded": sorted({g["week"] for g in res.games}),
        "gameCount": len(res.games),
        "teamCount": len(team_ids),
        "ohioTeamCount": len(ohio),
        "rankedTeamCount": len(ohio_sorted),
        "unplayedTeamCount": len(unplayed),
        "config": {
            "squashScale": cfg.squash_scale,
            "marginCap": cfg.margin_cap,
            "priorGames": cfg.prior_games,
            # Everything a consumer needs to turn a rating difference into a
            # win probability without re-deriving it:
            #     scale = sqrt(a + b / gamesPlayed), floored/capped, and
            #     flatScale when gamesPlayed < 1
            #     standInScale whenever the fixture is flagged `e`
            #     p(home) = 1 / (1 + exp(-margin / scale))
            # gamesPlayed is w+l+t of the *less* established of the two teams.
            # Display-only. predictedHomeMargin is already multiplied by this;
            # a consumer recomputing a probability must use the RAW difference,
            # i.e. divide it back out first.
            "marginScale": cfg.margin_scale,
            "probScale": {
                "a": round(cfg.prob_scale_a, 4),
                "b": round(cfg.prob_scale_b, 4),
                "flatScale": cfg.squash_scale,
                "standInScale": cfg.prob_scale_max,
                "min": cfg.prob_scale_min,
                "max": cfg.prob_scale_max,
            },
        },
        "hfa": {
            "rating": round(result.hfa_margin, 2),
            "btBinary": round(result.hfa_binary, 2),
            "massey": round(result.hfa_massey, 2),
        },
        "converged": result.converged,
        "prior": prior_meta,
        "tuned": tuned_meta,
        "connectivity": diag,
        # ---- the playoff model, and what it is entitled to claim -----------
        #
        # `harbinAgreement` is this build scoring its own Harbin implementation
        # against the source's published column. It is published rather than
        # merely asserted because the whole feature rests on the rule being the
        # real one, and a reader is owed the evidence rather than the promise.
        "playoffs": {
            "qualifiersPerRegion": QUALIFIERS_PER_REGION,
            "firstRoundByes": FIRST_ROUND_BYES,
            "lastRegularWeek": LAST_REGULAR_WEEK,
            "simulations": 10000,
            # Whether the Monte Carlo actually ran this build. Once the regular
            # season is over there is nothing left to decide, so it is skipped
            # and no team carries playoff odds -- the legitimate state of every
            # build from about November 1 to the state finals.
            #
            # check.py needs to be told, rather than inferring it from the
            # absence of odds, because "no odds because the season is finished"
            # and "no odds because the remaining schedule was lost" look
            # identical in the payload and mean opposite things. The count is
            # published alongside so the reason is legible.
            "simulated": bool(sim_rem),
            "remainingRegularFixtures": len(sim_rem),
            "method": (
                "Each remaining regular-season game is decided by the board's "
                "own win probability, the OHSAA Harbin qualifier is computed on "
                "the finished season, and the top "
                f"{QUALIFIERS_PER_REGION} of each region qualify. Repeated "
                "10,000 times. The rule is OHSAA's and is not modified; the "
                "forecast is this board's, because Harbin cannot forecast."
            ),
            "harbinAgreement": harbin_check,
            "whatIfCols": (
                "whatIf: w=week o=opponent (index into teams) h=1 when at home "
                "win/lose=playoff odds conditional on that result. "
                "watch: w=week h/a=the two teams for=which side to root for "
                "sw=how much their result moves this team's odds. "
                "Both are read off the same 10,000 seasons by conditioning, so "
                "playoffOdds always lies between win and lose."
            ),
            "approxTeams": sum(1 for t in team_ids
                               if res.teams[t].in_ohio and harbin_approx.get(t)),
            "approxNote": (
                "An out-of-state opponent carries no OHSAA division, so one is "
                "stood in. Teams whose two-level Harbin tree touches one are "
                "flagged harbinApprox: measured on 2025 the stand-in overstates "
                "by about 0.3 points for the teams most exposed to it."
            ),
        },
        "conflicts": res.conflicts,
        "warnings": res.warnings,
        "teams": rows,
        # Completed games, same compact shape. Together with `schedule` this
        # is a team's whole season -- what happened, then what is left. The two
        # lists shrink and grow past each other, so the payload stays about the
        # same size all season.
        #   w week · d date · h home · a away · hs/as scores · n neutral
        #
        # `d` is omitted for a season scraped before the date column existed,
        # which is why check.py treats it as optional rather than asserting it.
        # It costs little after gzip -- a season holds ~14 distinct dates -- and
        # it is what lets a missed game be told apart from an unplayed one.
        "results": [
            {"w": g.get("week", 1), "h": pos_of[g["home"]], "a": pos_of[g["away"]],
             "hs": g["home_score"], "as": g["away_score"],
             **({"d": g["date"]} if g.get("date") else {}),
             **({"n": 1} if g.get("neutral") else {})}
            for g in res.games
        ],
        "resultCols": "w=week d=date h=home a=away hs=homeScore as=awayScore "
                      "n=neutral; h/a are indexes into teams",
        "scheduleCols": "w=week d=date t=time h=home a=away n=neutral "
                        "m=predictedHomeMargin p=homeWinProb "
                        "ph=projectedHomeScore pa=projectedAwayScore "
                        "e=usedStandIn "
                        "o=assumedOhio x=whyNotPredicted; h/a are indexes into "
                        "teams, or a name when unresolved",
        "scheduleGameCount": len(schedule),
        "scheduleForeignDropped": dropped_foreign,
        "schedulePredictedCount": sum(1 for g in schedule if g.get("predicted")),
        "scheduleEstimatedCount": sum(1 for g in schedule if g.get("estimated")),
        # How an unrateable opponent was stood in for, so the site can say so
        # rather than presenting a guess as a measurement.
        "fallbackRating": ({"value": fallback[0], "basis": fallback[1]}
                           if fallback[0] is not None else None),
        "schedule": compact_schedule(schedule, team_ids),
        # How the board has actually done. `live` weeks were captured before
        # the games; `backtest` weeks were replayed afterwards from committed
        # scores and are a weaker claim -- they are never pooled together.
        "scorecard": scorecard,
        # Attribution is the term under which this data is used, not a
        # courtesy: the source grants reuse "provided that they credit the
        # source". sourceName/sourceUrl travel with the numbers so the page
        # cannot render them uncredited.
        "headToHead": headtohead,
        "trendCols": "w=throughWeek rating=Alex Points odds=playoff odds",
    }
    for r in rows:
        t = trend.get(r["name"])
        if t and len(t["w"]) >= 2:
            r["trend"] = t

    if hpath and record_history:
        snap = build_snapshot(season, through_week, payload["generatedAt"],
                              tuned_meta, rows, schedule)
        # A snapshot with nothing to forecast is not a prediction, and must not
        # enter a log whose whole value is that its entries were written before
        # the games. The workflow builds LAST season every run to derive the
        # prior; that build has no remaining fixtures, and without this guard it
        # appended a "live" line for 2025 week 16 -- a finished season, recorded
        # as though it had been foreseen. It would have done so on every run.
        if not snap["pred"]:
            print(f"history        : {season} week {through_week} has nothing "
                  f"left to predict; not recorded", file=sys.stderr)
        else:
            # `through_week` is the highest week holding ANY result, so it
            # turns over on the first Thursday-night game -- while ~350 of
            # that week's fixtures are still to come. Passing the results in
            # lets a capture be improved by a later build for as long as every
            # game it forecasts is still unplayed, and freezes it the moment
            # one kicks off. Without this the first build of the week won,
            # even a mid-week manual one, and the far better Saturday-morning
            # forecast was refused in silence.
            played = results_by_season.get(season) or {}
            what = record_snapshot(hpath, snap, played=played)
            if what != "kept":
                print(f"history        : {what} {season} week {through_week} "
                      f"({len(snap['pred'])} predictions)", file=sys.stderr)
            else:
                print(f"history        : {season} week {through_week} already "
                      f"recorded and its games have started; left alone",
                      file=sys.stderr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    emit_html(payload)
    return payload, result, res


# app.html is a fragment: a bare <head> run followed by the page body. These
# are the tags that make it a document, and BOTH variants get them. The preview
# used to be written as the raw fragment, which cost it the charset (the title
# rendered as "Alexâ€™s Awesome Aggregator") and, more seriously, the viewport
# meta -- so a real phone opening it laid the page out at 980px and none of the
# mobile media queries fired. The Playwright check in the handoff sets its
# viewport explicitly, so it passed while testing a page that was not the one
# being shipped.
HEAD_META = (
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<meta name=\"description\" content=\"Bradley-Terry and Massey ratings for "
    "Ohio high school varsity football, rebuilt weekly.\">\n"
)


def _document(head, body):
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + HEAD_META + head
            + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def emit_html(payload):
    """Write the two page variants from one source of truth.

    site/index.html   -- for GitHub Pages; fetches ratings.json alongside it
    dist/preview.html -- fully self-contained, data inlined, for sharing
    """
    app = os.path.join(SITE, "app.html")
    if not os.path.exists(app):
        return
    with open(app, encoding="utf-8") as fh:
        content = fh.read()

    marker = '<div class="wrap">'
    split = content.index(marker)
    head, body = content[:split], content[split:]

    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_document(head, body))

    inline = (
        "<script>window.__RATINGS__ = "
        + json.dumps(payload, separators=(",", ":"))
        + ";</script>\n"
    )
    # The <title> must stay near the top of the file -- publishers only scan the
    # first few KB for it, and the inlined dataset is a few hundred KB. So the
    # data goes in immediately after it, not at the top of the head.
    tend = head.index("</title>") + len("</title>")
    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    with open(os.path.join(dist, "preview.html"), "w", encoding="utf-8") as fh:
        fh.write(_document(head[:tend] + "\n" + inline + head[tend:], body))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--games")
    ap.add_argument("--roster")
    ap.add_argument("--out")
    ap.add_argument("--prior", help="path to prior.json; omit to auto-detect")
    ap.add_argument("--schedule",
                    help="path to schedule_{season}.csv; omit to auto-detect")
    ap.add_argument("--no-schedule", action="store_true",
                    help="ignore any remaining schedule (use for past seasons)")
    ap.add_argument("--no-prior", action="store_true",
                    help="ignore any prior (use when building a past season)")
    ap.add_argument("--no-history", action="store_true",
                    help="do not append this build to data/history.jsonl. Use "
                         "for reproducibility checks: the log is append-only "
                         "and a check should not write to it.")
    ap.add_argument("--no-site", action="store_true",
                    help="write ratings JSON only, do not rebuild the pages")
    ap.add_argument("--generated-at",
                    help="pin the generatedAt timestamp (e.g. "
                         "2026-08-25T00:00:00+00:00). Use this to verify a "
                         "build reproduces byte-for-byte; without it the "
                         "timestamp refreshes and the diff is never empty.")
    a = ap.parse_args()

    if a.no_site:
        globals()["emit_html"] = lambda payload: None
        if not a.out:
            print("note: --no-site without --out still rewrites "
                  "site/ratings.json, and generatedAt refreshes each run. "
                  "Pass --out, or --generated-at, to verify without a diff.",
                  file=sys.stderr)

    payload, result, res = main(
        games_path=a.games,
        roster_path=a.roster,
        out_path=a.out,
        generated_at=a.generated_at,
        prior_path=a.prior,
        use_prior=not a.no_prior,
        schedule_path=(False if a.no_schedule else a.schedule),
        record_history=not a.no_history,
    )
    print(f"games          : {payload['gameCount']}")
    print(f"teams          : {payload['teamCount']}  (Ohio: {payload['ohioTeamCount']})")
    print(f"converged      : {payload['converged']}")
    print(f"home field adv : {payload['hfa']['rating']:.2f} pts (headline model)")
    print(f"                 {payload['hfa']['massey']:.2f} pts (Massey)")
    print(f"schedule       : {payload['scheduleGameCount']} fixtures, "
          f"{payload['schedulePredictedCount']} predicted, "
          f"{payload['scheduleEstimatedCount']} using a stand-in opponent "
          f"({payload['scheduleForeignDropped']} dropped: no Ohio team)")
    if payload["fallbackRating"]:
        print(f"stand-in rating: {payload['fallbackRating']['value']:+.2f} "
              f"({payload['fallbackRating']['basis']})")
    print(f"conflicts      : {len(payload['conflicts'])}")
    for c in payload["conflicts"]:
        print(f"   - {c['name']}: {c['detail']}")
    print(f"warnings       : {len(payload['warnings'])}")
    for w in payload["warnings"][:20]:
        print(f"   ! {w}")
