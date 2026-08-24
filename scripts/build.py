"""Run resolution + ratings and emit the JSON the site reads."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ratings import RatingConfig, rate  # noqa: E402
from resolve import load_games, load_roster, resolve  # noqa: E402

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


def main(games_path=None, roster_path=None, out_path=None, generated_at=None,
         prior_path=None, use_prior=True):
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
        cfg = RatingConfig(
            squash_scale=float(best.get("squash_scale", cfg.squash_scale)),
            prior_games=float(best.get("prior_games", cfg.prior_games)),
        )
        tuned_meta = {k: best.get(k) for k in
                      ("squash_scale", "prior_games", "carry",
                       "accuracy", "logloss", "mae_margin", "n")}
        tuned_meta["tunedOn"] = tb.get("tunedOn")

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
        priors = {}
        for t in team_ids:
            tm = res.teams[t]
            key = (tm.school_id or "").strip()
            if key and key in pmap:
                priors[t] = pmap[key]
            elif tm.name in pmap:
                priors[t] = pmap[tm.name]
        prior_meta = dict(blob.get("meta", {}), matched=len(priors))
        if not priors:
            priors = None

    result = rate(team_ids, res.games, cfg, priors=priors)

    idx = {t: i for i, t in enumerate(team_ids)}

    # Rank Ohio teams only; out-of-state teams are rated but not ranked.
    ohio = [t for t in team_ids if res.teams[t].in_ohio]
    ohio_sorted = sorted(ohio, key=lambda t: -result.bt_margin[idx[t]])
    rank_of = {t: i + 1 for i, t in enumerate(ohio_sorted)}

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
            }
        )

    payload = {
        "generatedAt": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "weeksLoaded": sorted({g["week"] for g in res.games}),
        "gameCount": len(res.games),
        "teamCount": len(team_ids),
        "ohioTeamCount": len(ohio),
        "config": {
            "squashScale": cfg.squash_scale,
            "marginCap": cfg.margin_cap,
            "priorGames": cfg.prior_games,
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
    ap.add_argument("--no-prior", action="store_true",
                    help="ignore any prior (use when building a past season)")
    ap.add_argument("--no-site", action="store_true",
                    help="write ratings JSON only, do not rebuild the pages")
    a = ap.parse_args()

    if a.no_site:
        globals()["emit_html"] = lambda payload: None

    payload, result, res = main(
        games_path=a.games,
        roster_path=a.roster,
        out_path=a.out,
        prior_path=a.prior,
        use_prior=not a.no_prior,
    )
    print(f"games          : {payload['gameCount']}")
    print(f"teams          : {payload['teamCount']}  (Ohio: {payload['ohioTeamCount']})")
    print(f"converged      : {payload['converged']}")
    print(f"home field adv : {payload['hfa']['rating']:.2f} pts (headline model)")
    print(f"                 {payload['hfa']['massey']:.2f} pts (Massey)")
    print(f"conflicts      : {len(payload['conflicts'])}")
    for c in payload["conflicts"]:
        print(f"   - {c['name']}: {c['detail']}")
    print(f"warnings       : {len(payload['warnings'])}")
    for w in payload["warnings"][:20]:
        print(f"   ! {w}")
