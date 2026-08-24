"""
Scraper for joeeitel.com.

Two page families are read:

  /hsfoot/scoreboard/{season}/week-{n}   every game played that week
  /hsfoot/rankings/{season}/region-{n}   the OHSAA roster, 28 regions

Politeness: this is a one-person site that has been posting Ohio scores since
2000. A full season refresh is about 44 requests. We identify ourselves, we
rate limit, and we cache. Do not remove any of that.

Design note: every week is re-scraped on every run, not appended. Scores get
corrected days later -- a forfeit is recorded, a typo is fixed -- and an
append-only store would carry the original error forever.

The parser is deliberately loud. If a page's structure changes and the row
pattern stops matching, it raises rather than silently writing a short file,
because a scraper that quietly returns 30 games instead of 500 will produce a
plausible-looking ratings table built on almost nothing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://joeeitel.com"
UA = (
    "sneaky-fb-ratings/1.0 (+https://github.com/ktgz892ydt-gif/sneaky-fb-ratings; "
    "weekly ratings project; contact via repo issues)"
)
DELAY = 1.5  # seconds between requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, ".cache")

DIVISION_OF_REGION = {}
for _r in range(1, 29):
    DIVISION_OF_REGION[_r] = ["I", "II", "III", "IV", "V", "VI", "VII"][(_r - 1) // 4]

# Parsing strategy
# ----------------
# These pages are read as text, and the exact shape of a row varies with how
# the site marks it up: extra columns, a leading rank number, a trailing link.
# So none of these patterns anchor to end-of-line -- anchoring there was the
# original bug, and it fails *silently to the eye* by matching nothing at all.
#
# Patterns are tried in order, strictest first. Anything looser carries its own
# guard so a stray line of prose can't masquerade as a game.

NAME = r"[A-Za-z][A-Za-z0-9'&.\-/ ]*?(?:\([^)]*\))?"

GAME_PATTERNS = [
    # "Antwerp 14 at Montpelier 21"  ("vs" implies a neutral site)
    re.compile(
        rf"^\s*(?:\d+\.?\s+)?(?P<away>{NAME})\s+(?P<ascore>\d{{1,3}})\s+"
        rf"(?P<sep>at|vs\.?|@)\s+(?P<home>{NAME})\s+(?P<hscore>\d{{1,3}})(?!\d)",
        re.IGNORECASE,
    ),
    # "Antwerp at Montpelier 14 21" -- names first, then both scores
    re.compile(
        rf"^\s*(?:\d+\.?\s+)?(?P<away>{NAME})\s+(?P<sep>at|vs\.?|@)\s+"
        rf"(?P<home>{NAME})\s+(?P<ascore>\d{{1,3}})\s+(?P<hscore>\d{{1,3}})(?!\d)",
        re.IGNORECASE,
    ),
    # Bare table row with no joining word: "Antwerp 14 Montpelier 21"
    re.compile(
        rf"^\s*(?:\d+\.?\s+)?(?P<away>{NAME})\s+(?P<ascore>\d{{1,3}})\s+"
        rf"(?P<home>{NAME})\s+(?P<hscore>\d{{1,3}})(?!\d)"
    ),
]

# Ranking rows are columnar, and crucially the record comes BEFORE the name:
#
#   Current Rank | Rated W-L | ID # | Mailing City | School | Current Average | ...
#   1t           | 1-0       | 268  | Brunswick    | Brunswick | 6.5000       | ...
#   8            | 1-0       | 764  | Massillon    | Jackson   | 5.5000       | ...
#
# The Mailing City is what finally separates the three schools called Perry,
# and the ID # is a stable key that never collides at all. Both are worth more
# than any name-matching heuristic.
#
# Because "Mailing City" and "School" are both free text with spaces, a flat
# regex cannot reliably find the boundary between them ("Cuyahoga Falls" +
# "Cuyahoga Valley Christian Academy"). So the columns are read positionally
# from the table cells, and the text pattern below exists only as a fallback.

RANK_TEXT_RE = re.compile(
    r"^\s*(?P<rank>\d{1,3}t?)\s+"
    r"(?P<w>\d{1,2})-(?P<l>\d{1,2})(?:-(?P<t>\d{1,2}))?\s+"
    r"(?P<sid>\d{1,6})\s+"
    r"(?P<middle>.+?)\s+"
    r"(?P<harbin>\d+\.\d{3,4})(?!\d)"
)

ROSTER_HEADERS = {
    "record": ("rated w-l", "rated wl", "w-l", "record"),
    "sid": ("id #", "id#", "id"),
    "city": ("mailing city", "city"),
    "school": ("school", "team"),
    "harbin": ("current average", "average", "harbin"),
}


def _clean_name(s):
    s = re.sub(r"\s+", " ", s).strip(" .-–—")
    return s


def _plausible_team(s):
    """Reject prose and navigation chrome that happens to sit near numbers."""
    if not s or len(s) < 2 or len(s) > 48:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    if len(s.split()) > 6:
        return False
    lowered = s.lower()
    junk = ("copyright", "rankings", "scoreboard", "click", "week ", "http",
            "all rights", "powered by", "search", "select", "printable")
    return not any(j in lowered for j in junk)


# Scoreboard rows carry when the game kicked off, e.g.
#   "2026-08-20     7pm Antwerp (Antwerp) 14 at Montpelier (Montpelier) 21"
# That prefix has to come off before the row will parse. Times appear as
# "7pm", "7:00", "7:00 PM", "noon" or "TBA".
LEAD_RE = re.compile(
    r"^\s*"
    r"(?:\d{4}-\d{2}-\d{2}\s+)?"
    r"(?:(?:Mon|Tues?|Wed(?:nes)?|Thur?s?|Fri|Sat(?:ur)?|Sun)(?:day)?\.?,?\s+)?"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s*\d{4})?\s+)?"
    r"(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s+)?"
    r"(?:(?:\d{1,2}(?::\d{2})?\s*[APap]\.?[Mm]\.?|\d{1,2}:\d{2}|noon|TBA|TBD)\s+)?",
    re.IGNORECASE,
)


def _candidates(line):
    """The line as-is, then again with any leading date/time removed."""
    yield line
    stripped = LEAD_RE.sub("", line, count=1)
    if stripped != line and stripped:
        yield stripped


def _match_game(line):
    raw = _match_game_one(line)
    stripped = None
    cands = list(_candidates(line))
    if len(cands) > 1:
        stripped = _match_game_one(cands[1])

    if raw and stripped:
        # Both parsed. Prefer the de-prefixed one only when its team name is
        # strictly the tail of the raw one -- i.e. the prefix really was junk
        # glued to the front, not part of a school called Mayfield or Sunbury.
        if raw[0].lower().endswith(stripped[0].lower()) and len(stripped[0]) < len(raw[0]):
            return stripped
        return raw
    return raw or stripped


def _match_game_one(line):
    for pat in GAME_PATTERNS:
        m = pat.match(line)
        if not m:
            continue
        away = _clean_name(m.group("away"))
        home = _clean_name(m.group("home"))
        if not (_plausible_team(away) and _plausible_team(home)):
            continue
        if away.lower() == home.lower():
            continue
        a, h = int(m.group("ascore")), int(m.group("hscore"))
        if a > 120 or h > 120:
            continue
        sep = (m.groupdict().get("sep") or "").lower()
        return away, a, home, h, 1 if sep.startswith("vs") else 0
    return None


def team_key(school, city):
    """The identity used everywhere: 'Jackson (Massillon)'.

    The scoreboard already writes teams this way, so keying on it makes the
    join between scores and roster exact instead of a guess.
    """
    school = _clean_name(school)
    city = _clean_name(city)
    return f"{school} ({city})" if city else school


def page_rows(html):
    """Table rows as lists of cell strings, preserving column boundaries."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    rows = []
    for tr in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", td.get_text(" ", strip=True))
                 for td in tr.find_all(["td", "th"])]
        if any(c for c in cells):
            rows.append(cells)
    return rows


def _header_map(cells):
    """Map our field names onto this table's column indices, or None."""
    lowered = [c.strip().lower().rstrip(":") for c in cells]
    found = {}
    for field, aliases in ROSTER_HEADERS.items():
        for i, h in enumerate(lowered):
            if h in aliases:
                found[field] = i
                break
    # A usable header needs at least the record, the school and the average.
    if all(k in found for k in ("record", "school", "harbin")):
        return found
    return None


REC_RE = re.compile(r"^(\d{1,2})-(\d{1,2})(?:-(\d{1,2}))?$")


def _rows_from_table(rows):
    """Yield (key, school, city, sid, w, l, t, harbin) using header columns."""
    cols = None
    for cells in rows:
        if cols is None:
            cols = _header_map(cells)
            continue
        if max(cols.values()) >= len(cells):
            continue
        rec = REC_RE.match(cells[cols["record"]].strip())
        if not rec:
            continue
        school = cells[cols["school"]]
        city = cells[cols["city"]] if "city" in cols and cols["city"] < len(cells) else ""
        if not _plausible_team(school):
            continue
        try:
            harbin = float(cells[cols["harbin"]])
        except ValueError:
            continue
        sid = ""
        if "sid" in cols and cols["sid"] < len(cells):
            sid = cells[cols["sid"]].strip()
        w, l = int(rec.group(1)), int(rec.group(2))
        t = int(rec.group(3) or 0)
        if w > 16 or l > 16:
            continue
        yield team_key(school, city), _clean_name(school), _clean_name(city), sid, w, l, t, harbin


def _rows_from_text(lines, known_pairs):
    """Fallback when the page isn't a real table.

    'Massillon Jackson' has to be split into city and school, and only the
    scoreboard knows where the boundary is -- so we test each split against
    the (school, city) pairs already seen in the scores.
    """
    for ln in lines:
        m = RANK_TEXT_RE.match(ln)
        if not m:
            continue
        middle = _clean_name(m.group("middle"))
        words = middle.split()
        school = city = None
        for cut in range(1, len(words)):
            c, s = " ".join(words[:cut]), " ".join(words[cut:])
            if (s.lower(), c.lower()) in known_pairs:
                city, school = c, s
                break
        if school is None:
            # Nothing to check against (a team on a bye, say). The site lists
            # city then school, and single-word cities are overwhelmingly the
            # common case, so take one word as the city.
            if len(words) >= 2:
                city, school = words[0], " ".join(words[1:])
            else:
                city, school = "", middle
        if not _plausible_team(school):
            continue
        w, l = int(m.group("w")), int(m.group("l"))
        t = int(m.groupdict().get("t") or 0)
        yield (team_key(school, city), school, city, m.group("sid"),
               w, l, t, float(m.group("harbin")))


def _diagnose(label, lines, limit=45):
    """Print what we actually saw, so a failed run is one round trip to fix."""
    print(f"\n  !! {label}: nothing matched. First {limit} candidate lines:",
          file=sys.stderr)
    shown = 0
    for ln in lines:
        if len(ln) < 3 or len(ln) > 200:
            continue
        print(f"     | {ln}", file=sys.stderr)
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("     (the page produced no usable text at all -- it may be "
              "rendered by JavaScript, which needs a different approach)",
              file=sys.stderr)
    print("", file=sys.stderr)


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "text/html"})
    return s


def fetch(sess, path, use_cache=True):
    os.makedirs(CACHE, exist_ok=True)
    key = path.strip("/").replace("/", "_") + ".html"
    cached = os.path.join(CACHE, key)
    if use_cache and os.path.exists(cached):
        age = time.time() - os.path.getmtime(cached)
        if age < 60 * 30:  # 30 minutes
            with open(cached, encoding="utf-8") as fh:
                return fh.read()

    url = BASE + path
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(DELAY)
    with open(cached, "w", encoding="utf-8") as fh:
        fh.write(resp.text)
    return resp.text


def page_lines(html):
    """Every plausible one-row-per-line rendering of the page.

    The same row may appear more than once here (once from its table markup,
    once from the flat text). That is deliberate: whichever rendering the
    patterns can read is the one that wins, and duplicates are removed later
    by the caller.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "head"]):
        tag.decompose()

    lines = []

    # Table rows, cells joined -- handles data split across <td>s.
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" ".join(cells))

    # Block elements that often hold one row each.
    for el in soup.find_all(["li", "p", "div"]):
        if el.find(["tr", "li", "p", "div", "table"]):
            continue  # only leaf-ish blocks, else we get the whole page as one line
        txt = el.get_text(" ", strip=True)
        if txt:
            lines.append(txt)

    # Flat text as a final fallback.
    lines.extend(soup.get_text("\n", strip=True).split("\n"))

    out, seen = [], set()
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out


def scrape_week(sess, season, week, use_cache=True, diagnose=False):
    html = fetch(sess, f"/hsfoot/scoreboard/{season}/week-{week}", use_cache)
    lines = page_lines(html)
    games, seen = [], set()
    for ln in lines:
        hit = _match_game(ln)
        if not hit:
            continue
        away, ascore, home, hscore, neutral = hit
        key = (away.lower(), home.lower())
        if key in seen:
            continue
        seen.add(key)
        games.append(
            {
                "week": week,
                "away": away,
                "away_score": ascore,
                "home": home,
                "home_score": hscore,
                "neutral": neutral,
            }
        )
    if not games and diagnose:
        _diagnose(f"week {week} scoreboard", lines)
    return games


def scrape_roster(sess, season, use_cache=True, known_pairs=frozenset()):
    rows = []
    empty_regions = []
    for region in range(1, 29):
        html = fetch(sess, f"/hsfoot/rankings/{season}/region-{region}", use_cache)

        parsed = list(_rows_from_table(page_rows(html)))
        source = "table"
        if not parsed:
            parsed = list(_rows_from_text(page_lines(html), known_pairs))
            source = "text"

        found, seen = 0, set()
        for key, school, city, sid, w, l, t, harbin in parsed:
            if key.lower() in seen:
                continue
            seen.add(key.lower())
            rows.append(
                {
                    "name": key,
                    "school": school,
                    "city": city,
                    "school_id": sid,
                    "division": DIVISION_OF_REGION[region],
                    "region": region,
                    "record": f"{w}-{l}" + (f"-{t}" if t else ""),
                    "harbin": harbin,
                }
            )
            found += 1

        print(f"  region {region:>2}: {found} teams ({source})", file=sys.stderr)
        if found == 0:
            empty_regions.append(region)
            if len(empty_regions) == 1:
                _diagnose(f"region {region} rankings", page_lines(html))

    if empty_regions:
        raise SystemExit(
            f"\nRegions with no teams parsed: {empty_regions}\n"
            f"The sample lines above show what the page actually looks like. "
            f"Fix the column mapping in ROSTER_HEADERS, or RANK_TEXT_RE, to "
            f"match -- rather than shipping a roster with missing regions."
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through-week", type=int, default=16)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    sess = _session()
    use_cache = not args.no_cache
    os.makedirs(DATA, exist_ok=True)

    # Scores are read first: they establish the (school, city) pairs that the
    # roster's text fallback needs to split its city and school columns.
    all_games = []
    print("games:", file=sys.stderr)
    for wk in range(1, args.through_week + 1):
        try:
            # Diagnose on week 1 only -- if that one is empty the pattern is
            # wrong, and dumping every week would bury the useful output.
            g = scrape_week(sess, args.season, wk, use_cache, diagnose=(wk == 1))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break  # season hasn't reached this week
            raise
        if not g:
            break  # week not played yet
        print(f"  week {wk:>2}: {len(g)} games", file=sys.stderr)
        all_games.extend(g)

    if not all_games:
        raise SystemExit(
            "No games parsed at all. The sample lines above show what the "
            "scoreboard page actually looks like -- fix GAME_PATTERNS to match. "
            "Refusing to overwrite good data."
        )

    gpath = os.path.join(DATA, f"games_{args.season}.csv")
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as fh:
            prior = sum(1 for _ in fh) - 1
        # Guard against a partial scrape silently truncating the season record.
        if prior > 50 and len(all_games) < prior * 0.8:
            raise SystemExit(
                f"Parsed {len(all_games)} games but {gpath} already holds "
                f"{prior}. That is a suspicious drop -- not overwriting. "
                f"Inspect the source pages before re-running."
            )

    with open(gpath, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(
            fh, fieldnames=["week", "away", "away_score", "home", "home_score", "neutral"]
        )
        wtr.writeheader()
        wtr.writerows(all_games)
    print(f"games: {len(all_games)} -> {gpath}", file=sys.stderr)

    # Every (school, city) pair the scores mention, for the roster fallback.
    known_pairs = set()
    for g in all_games:
        for side in ("away", "home"):
            m = re.match(r"^(.*?)\s*\(([^)]*)\)$", g[side])
            if m:
                known_pairs.add((m.group(1).strip().lower(), m.group(2).strip().lower()))

    print("roster:", file=sys.stderr)
    roster = scrape_roster(sess, args.season, use_cache, known_pairs)
    rpath = os.path.join(DATA, f"roster_{args.season}.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(
            fh,
            fieldnames=["name", "school", "city", "school_id",
                        "division", "region", "record", "harbin"],
        )
        wtr.writeheader()
        wtr.writerows(roster)
    print(f"roster: {len(roster)} teams -> {rpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
