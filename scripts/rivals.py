"""
Scoring this board against another public forecaster, on the same games.

The other model here is Drew Pasteur's Ohio Fantastic 50 (fantastic50.net), a
long-running Ohio high school football rating and prediction site. It publishes
a favourite, a predicted margin and a win probability for each upcoming game --
the same three quantities this board produces -- which is what makes an
apples-to-apples comparison possible at all.

Permission and etiquette
------------------------
The site states: "Media outlets (print, broadcast, or online) are welcome to
use any content from this site, provided that they credit the source." That is
explicit permission conditional on attribution, so the credit is not optional
politeness -- it is the term under which this data is used, and it must appear
wherever the comparison appears.

Its robots.txt asks for a 10-second crawl delay, six times what joeeitel.com
asks. One request per week is made, and the delay is honoured anyway. This
module does not mirror or republish his predictions: it records what is needed
to score them and reports aggregates.

Why the comparison must be PAIRED
---------------------------------
Both sites publish their own accuracy. Those figures are not comparable: he
predicted 345 games in week 1 of 2026 where this board's scrape found 400
completed, so the two are scoring different sets, of different difficulty.
Comparing published headline numbers would be measuring the schedules, not the
models.

So only games BOTH models predicted are scored, and the difference is tested
paired -- the same argument as the one-standard-error rule in tune.py. Most of
the variance in whether a game is called correctly is the game itself; both
models miss the same upsets. Differencing removes that shared noise.

Matching
--------
His team names are Joe Eitel's school names -- he credits the same source -- but
they are bare ("Deer Park"), sometimes city-prefixed ("Dayton Stivers") and
sometimes abbreviated ("Cuyahoga Val. Christian"). Matching on names alone
resolves 95% and leaves Ohio's duplicate names (Perry, Jackson, Madison)
genuinely ambiguous.

So the match is on the GAME, not the team: a pair of names against the fixture
list this board already holds for that week. That is far more identifying, and
it took coverage to 97% with zero ambiguous matches before abbreviation
handling. Where a pick still matches more than one fixture it is DROPPED, not
guessed -- the same contract resolve.py keeps.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

SOURCE = "fantastic50"
SOURCE_NAME = "Drew Pasteur's Ohio Fantastic 50"
SOURCE_URL = "https://www.fantastic50.net/"
PICKS_URL = "https://www.fantastic50.net/picks.html"
CRAWL_DELAY = 10.0          # robots.txt asks for this; we make one request a week

_NAME = r"[A-Za-z][A-Za-z0-9'&.\-/ ]{0,38}?"
_DAY = r"(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day"

# "#45 Perkins (1-0) by 49 (99%) at Scott (0-1)"
#
# The first team named is the FAVOURITE, not the away side, and "at" means the
# favourite is away. That is the opposite convention to the scoreboard, where
# the first team named is always the visitor. Reading it the other way would
# invert the home/away of every pick and still look entirely plausible.
PICK_RE = re.compile(
    rf"(?:#\d+\s+)?(?P<fav>{_NAME})\s*\((?P<favrec>\d{{1,2}}-\d{{1,2}})\)\s+"
    rf"by\s+(?P<margin>\d{{1,3}})\s+\((?P<prob>\d{{1,3}})%\)\s+"
    rf"(?P<sep>at|vs)\s+(?:#\d+\s+)?(?P<dog>{_NAME})\s*\((?P<dogrec>\d{{1,2}}-\d{{1,2}})\)"
)
WEEK_RE = re.compile(r"[Pp]icks for week\s*#?\s*(\d{1,2})")


def _strip_day(name):
    """A day heading runs into the next team name once the page is flattened."""
    return re.sub(rf"^.*?{_DAY}\s+", "", name.strip()).strip(" .")


def parse_picks(flat):
    """-> (week, [pick dicts]) from the flattened picks page."""
    m = WEEK_RE.search(flat)
    week = int(m.group(1)) if m else None
    out = []
    for mm in PICK_RE.finditer(flat):
        d = mm.groupdict()
        fav, dog = _strip_day(d["fav"]), _strip_day(d["dog"])
        if not fav or not dog or fav.lower() == dog.lower():
            continue
        prob = int(d["prob"]) / 100.0
        out.append({
            "fav": fav,
            "dog": dog,
            # "at" => the favourite is the visitor.
            "favHome": d["sep"].lower() == "vs",
            "margin": int(d["margin"]),
            "favProb": min(max(prob, 0.01), 0.99),
        })
    return week, out


# ---------------------------------------------------------------- matching

def _norm(s):
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


def _tokens_agree(his, ours):
    """Every token of his name prefixes the matching token of ours, in order.

    Handles the abbreviations: "Cuyahoga Val. Christian" against "Cuyahoga
    Valley Christian Academy", "Notre Dame-Cath. Latin" against "Notre
    Dame-Cathedral Latin". Ours may carry extra trailing words; his may not.
    """
    a, b = _norm(his).split(), _norm(ours).split()
    if not a or len(a) > len(b):
        return False
    return all(y.startswith(x) for x, y in zip(a, b))


def _compatible(his, our_full):
    """Could his bare name refer to our 'School (City)' team?"""
    school = our_full.rsplit(" (", 1)[0]
    city = our_full.rsplit(" (", 1)[1][:-1] if " (" in our_full else ""
    if _tokens_agree(his, school):
        return True
    # "Dayton Stivers" for "Stivers (Dayton)" -- the city moved to the front.
    if city and (_tokens_agree(his, f"{city} {school}")
                 or _tokens_agree(his, f"{school} {city}")):
        return True
    # "Woodward (Cincy)" -- city kept, but abbreviated.
    m = re.match(r"^(.*?)\s*\(([^)]+)\)$", his.strip())
    if m and city:
        return _tokens_agree(m.group(1), school) and _tokens_agree(m.group(2), city)
    return False


def match_picks(picks, fixtures):
    """Join his picks onto our fixtures for the same week.

    `fixtures` maps (homeName, awayName) -> our own prediction dict. Returns
    (matched, report). A pick matching several fixtures is dropped and counted,
    never assigned -- the resolver's contract, applied to someone else's data.
    """
    matched, ambiguous, unmatched = [], [], []
    for p in picks:
        home = p["dog"] if not p["favHome"] else p["fav"]
        away = p["fav"] if not p["favHome"] else p["dog"]
        cands = [k for k in fixtures
                 if _compatible(home, k[0]) and _compatible(away, k[1])]
        if len(cands) == 1:
            key = cands[0]
            # Restate his call from the HOME team's perspective, which is how
            # this board states every prediction.
            home_is_fav = p["favHome"]
            matched.append({
                "home": key[0], "away": key[1],
                "homeMargin": p["margin"] if home_is_fav else -p["margin"],
                "homeProb": round(p["favProb"] if home_is_fav
                                  else 1 - p["favProb"], 3),
            })
        elif len(cands) > 1:
            ambiguous.append([p["fav"], p["dog"]])
        else:
            unmatched.append([p["fav"], p["dog"]])
    return matched, {
        "picks": len(picks),
        "matched": len(matched),
        "ambiguous": len(ambiguous),
        "unmatched": len(unmatched),
        "coverage": round(len(matched) / len(picks), 4) if picks else None,
        "unmatchedSample": unmatched[:8],
    }


# ---------------------------------------------------------------- the log

def append_if_new(path, record):
    """One record per (source, season, week). Same discipline as history.jsonl:
    the first capture is the prediction, a later one has seen more."""
    for existing in load(path):
        if (existing.get("source") == record["source"]
                and existing.get("season") == record["season"]
                and existing.get("week") == record["week"]):
            return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return True


def load(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fetch(session=None, url=PICKS_URL, delay=CRAWL_DELAY):
    """One polite request. Returns the flattened page text."""
    import requests
    from scrape import UA, flat_text
    s = session or requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "text/html"})
    resp = s.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)
    return flat_text(resp.text)


# ------------------------------------------------------- head to head

def head_to_head(rival_records, our_snapshots, results_by_season, source=SOURCE):
    """Score both models on the games BOTH predicted, and test the difference.

    Only the intersection is scored. Each side's own published accuracy is over
    its own game set -- he predicted 345 games in week 1 of 2026 where this
    board's scrape found 400 completed -- so comparing headline figures would
    measure the schedules rather than the models.

    The test is PAIRED, for the same reason the one-standard-error rule in
    tune.py is paired: most of the variance in whether a game is called right
    is the game, and both models miss the same upsets. On accuracy the paired
    statistic is McNemar's -- only the games the two disagreed about carry any
    information about which is better.
    """
    ours_by = {}
    for snap in our_snapshots:
        for home, away, week, margin, prob in snap.get("pred", []):
            ours_by[(snap.get("season"), week, home, away)] = (margin, prob)

    n = 0
    we_right = they_right = 0
    both_wrong = both_right = 0
    we_only = they_only = 0          # McNemar's discordant pairs
    our_ll = their_ll = 0.0
    ll_diffs = []
    our_abs = their_abs = 0.0

    for rec in rival_records:
        if rec.get("source") != source:
            continue
        season, week = rec.get("season"), rec.get("week")
        got = results_by_season.get(season) or {}
        for p in rec.get("picks", []):
            key = (season, week, p["home"], p["away"])
            mine = ours_by.get(key)
            actual = got.get((week, p["home"], p["away"]))
            if mine is None or actual is None or actual == 0:
                continue
            home_won = actual > 0
            n += 1

            ours_ok = (mine[0] > 0) == home_won
            theirs_ok = (p["homeMargin"] > 0) == home_won
            we_right += ours_ok
            they_right += theirs_ok
            if ours_ok and theirs_ok:
                both_right += 1
            elif ours_ok:
                we_only += 1
            elif theirs_ok:
                they_only += 1
            else:
                both_wrong += 1

            op = min(max(mine[1], 1e-6), 1 - 1e-6)
            tp = min(max(p["homeProb"], 1e-6), 1 - 1e-6)
            oll = -(math.log(op) if home_won else math.log(1 - op))
            tll = -(math.log(tp) if home_won else math.log(1 - tp))
            our_ll += oll
            their_ll += tll
            ll_diffs.append(oll - tll)
            our_abs += abs(mine[0] - actual)
            their_abs += abs(p["homeMargin"] - actual)

    if not n:
        return None

    # McNemar: only discordant games say anything about which model is better.
    disc = we_only + they_only
    acc_gap = (we_only - they_only) / n
    acc_se = (disc ** 0.5) / n if disc else 0.0

    mean_ll_gap = sum(ll_diffs) / n
    if n > 1:
        var = sum((d - mean_ll_gap) ** 2 for d in ll_diffs) / (n - 1)
        ll_se = (var / n) ** 0.5
    else:
        ll_se = 0.0

    # McNemar, done EXACTLY rather than by the normal approximation.
    #
    # Under the null the two models are equally likely to win each game they
    # disagreed about, so the discordant pairs are coin flips and the p-value
    # is a plain two-sided binomial. The approximation sqrt(b+c)/n breaks at
    # small counts in a way that matters here: with a single disagreement the
    # gap and its standard error are ALWAYS exactly equal (1/n against
    # sqrt(1)/n), so one lucky call would be graded as evidence. It is not --
    # the exact p-value there is 1.0.
    def _p_value(b, c):
        k = b + c
        if k == 0:
            return 1.0
        hi = max(b, c)
        tail = sum(math.comb(k, i) for i in range(hi, k + 1)) / (2 ** k)
        return min(1.0, 2 * tail)

    def _verdict(p):
        if p > 0.10:
            return "indistinguishable"
        if p > 0.01:
            return "leaning"
        return "clear"

    def _z_verdict(gap, se):
        """For log loss, where n is large and the quantity is continuous."""
        if se <= 0:
            return "indistinguishable"
        z = abs(gap) / se
        return "indistinguishable" if z < 1.64 else ("leaning" if z < 2.58 else "clear")

    return {
        "source": source,
        "sourceName": SOURCE_NAME,
        "sourceUrl": SOURCE_URL,
        "sharedGames": n,
        "ours": {"accuracy": round(we_right / n, 4),
                 "logloss": round(our_ll / n, 4),
                 "meanMarginError": round(our_abs / n, 2)},
        "theirs": {"accuracy": round(they_right / n, 4),
                   "logloss": round(their_ll / n, 4),
                   "meanMarginError": round(their_abs / n, 2)},
        # Only these two counts carry information about which model is better.
        "disagreements": {"weWereRight": we_only, "theyWereRight": they_only,
                          "bothRight": both_right, "bothWrong": both_wrong},
        "accuracyGap": round(acc_gap, 4),
        "accuracyGapSE": round(acc_se, 4),
        "accuracyPValue": round(_p_value(we_only, they_only), 4),
        "accuracyVerdict": _verdict(_p_value(we_only, they_only)),
        "loglossGap": round(mean_ll_gap, 4),
        "loglossGapSE": round(ll_se, 4),
        "loglossVerdict": _z_verdict(mean_ll_gap, ll_se),
    }
