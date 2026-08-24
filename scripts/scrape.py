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
    "sneaky-fb-ratings/1.0 (+https://github.com/YOURNAME/sneaky-fb-ratings; "
    "weekly ratings project; contact via repo issues)"
)
DELAY = 1.5  # seconds between requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, ".cache")

DIVISION_OF_REGION = {}
for _r in range(1, 29):
    DIVISION_OF_REGION[_r] = ["I", "II", "III", "IV", "V", "VI", "VII"][(_r - 1) // 4]

# "Away 14 at Home 21" / "Away 14 vs Home 21" (vs implies a neutral site).
# Team names may carry a parenthetical qualifier, e.g. "Jackson (Massillon)".
GAME_RE = re.compile(
    r"^\s*(?P<away>[A-Za-z0-9'&.\-/ ]+?(?:\([^)]*\))?)\s+"
    r"(?P<ascore>\d{1,3})\s+"
    r"(?P<sep>at|vs\.?)\s+"
    r"(?P<home>[A-Za-z0-9'&.\-/ ]+?(?:\([^)]*\))?)\s+"
    r"(?P<hscore>\d{1,3})\s*$",
    re.IGNORECASE,
)

RANK_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(?P<name>[A-Za-z0-9'&.\-/ ]+?(?:\([^)]*\))?)\s+"
    r"(?P<w>\d{1,2})-(?P<l>\d{1,2})(?:-(?P<t>\d{1,2}))?\s+"
    r"(?P<harbin>\d+(?:\.\d+)?)\s*$"
)


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
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    # Tables first: if the data is in rows, join cells so the regex sees one
    # game per line regardless of how the cells are split.
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" ".join(cells))

    text = soup.get_text("\n", strip=True)
    lines.extend(ln.strip() for ln in text.split("\n") if ln.strip())
    return lines


def scrape_week(sess, season, week, use_cache=True):
    html = fetch(sess, f"/hsfoot/scoreboard/{season}/week-{week}", use_cache)
    games, seen = [], set()
    for ln in page_lines(html):
        m = GAME_RE.match(ln)
        if not m:
            continue
        away = m.group("away").strip()
        home = m.group("home").strip()
        if not away or not home or away.lower() == home.lower():
            continue
        key = (away, home, m.group("ascore"), m.group("hscore"))
        if key in seen:
            continue
        seen.add(key)
        games.append(
            {
                "week": week,
                "away": away,
                "away_score": int(m.group("ascore")),
                "home": home,
                "home_score": int(m.group("hscore")),
                "neutral": 1 if m.group("sep").lower().startswith("vs") else 0,
            }
        )
    return games


def scrape_roster(sess, season, use_cache=True):
    rows = []
    for region in range(1, 29):
        html = fetch(sess, f"/hsfoot/rankings/{season}/region-{region}", use_cache)
        found = 0
        for ln in page_lines(html):
            m = RANK_RE.match(ln)
            if not m:
                continue
            w, l = int(m.group("w")), int(m.group("l"))
            t = int(m.group("t") or 0)
            rows.append(
                {
                    "name": m.group("name").strip(),
                    "division": DIVISION_OF_REGION[region],
                    "region": region,
                    "record": f"{w}-{l}" + (f"-{t}" if t else ""),
                    "harbin": float(m.group("harbin")),
                }
            )
            found += 1
        if found == 0:
            raise SystemExit(
                f"Region {region} parsed 0 teams. The page structure has "
                f"probably changed -- fix the parser rather than shipping a "
                f"roster with a missing region."
            )
        print(f"  region {region:>2}: {found} teams", file=sys.stderr)
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

    print("roster:", file=sys.stderr)
    roster = scrape_roster(sess, args.season, use_cache)
    rpath = os.path.join(DATA, f"roster_{args.season}.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["name", "division", "region", "record", "harbin"])
        wtr.writeheader()
        wtr.writerows(roster)
    print(f"roster: {len(roster)} teams -> {rpath}", file=sys.stderr)

    all_games = []
    for wk in range(1, args.through_week + 1):
        try:
            g = scrape_week(sess, args.season, wk, use_cache)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break  # season hasn't reached this week
            raise
        if not g:
            break  # week not played yet
        print(f"  week {wk:>2}: {len(g)} games", file=sys.stderr)
        all_games.extend(g)

    if not all_games:
        raise SystemExit("No games parsed at all. Refusing to overwrite good data.")

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


if __name__ == "__main__":
    main()
