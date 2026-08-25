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

from ratings import RatingConfig, rate, win_probability  # noqa: E402
from resolve import load_games, load_roster, resolve, team_identity  # noqa: E402

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
        p = win_probability(margin, established, cfg)
        row.update(
            predicted=True,
            predictedHomeMargin=round(float(margin), 1),
            homeWinProb=round(float(p), 3),
            favoriteName=f["home"] if margin >= 0 else f["away"],
            spread=round(abs(float(margin)), 1),
            gamesBehind=established,
            estimated=("stand-in" in (hstat, astat)),
            estimatedNote=(fb_note if "stand-in" in (hstat, astat) else ""),
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
         prior_path=None, use_prior=True, schedule_path=None):
    # Prefer real scraped data when it exists; fall back to the checked-in
    # Week 1 fixture so the pipeline is runnable without network access.
    if games_path is None:
        scraped = os.path.join(DATA, "games_2026.csv")
        games_path = scraped if os.path.exists(scraped) else os.path.join(
            DATA, "fixture_week1_2026.psv"
        )
    roster_path = roster_path or os.path.join(DATA, "roster_2026.csv")
    out_path = out_path or os.path.join(SITE, "ratings.json")

    # The season is whichever one these files are for -- inferred from the
    # filename so that building a past season does not mislabel its output.
    m = re.search(r"(\d{4})", os.path.basename(games_path))
    season = int(m.group(1)) if m else 2026

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
                **projections.get(t, {}),
            }
        )

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
            #     p(home) = 1 / (1 + exp(-margin / scale))
            # gamesPlayed is w+l+t of the *less* established of the two teams.
            "probScale": {
                "a": round(cfg.prob_scale_a, 4),
                "b": round(cfg.prob_scale_b, 4),
                "flatScale": cfg.squash_scale,
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
        "conflicts": res.conflicts,
        "warnings": res.warnings,
        "teams": rows,
        # Completed games, same compact shape. Together with `schedule` this
        # is a team's whole season -- what happened, then what is left. The two
        # lists shrink and grow past each other, so the payload stays about the
        # same size all season.
        #   w week · h home · a away · hs/as scores · n neutral
        "results": [
            {"w": g.get("week", 1), "h": pos_of[g["home"]], "a": pos_of[g["away"]],
             "hs": g["home_score"], "as": g["away_score"],
             **({"n": 1} if g.get("neutral") else {})}
            for g in res.games
        ],
        "scheduleCols": "w=week d=date t=time h=home a=away n=neutral "
                        "m=predictedHomeMargin p=homeWinProb e=usedStandIn "
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
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    emit_html(payload)
    return payload, result, res


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
        fh.write(
            "<!doctype html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<meta name=\"description\" content=\"Bradley-Terry and Massey ratings for "
            "Ohio high school varsity football, rebuilt weekly.\">\n"
            + head +
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
        )

    inline = (
        "<script>window.__RATINGS__ = "
        + json.dumps(payload, separators=(",", ":"))
        + ";</script>\n"
    )
    # The <title> must stay near the top of the file -- publishers only scan the
    # first few KB for it, and the inlined dataset is a few hundred KB.
    tend = content.index("</title>") + len("</title>")
    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    with open(os.path.join(dist, "preview.html"), "w", encoding="utf-8") as fh:
        fh.write(content[:tend] + "\n" + inline + content[tend:])


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
