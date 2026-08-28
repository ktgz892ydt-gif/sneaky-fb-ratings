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
import json
import os
import re
import sys
import time
from typing import NamedTuple

import requests
from bs4 import BeautifulSoup

# Bump this whenever a change to the parsing above could produce a DIFFERENT
# set of games from the same pages -- a new pattern, a relaxed filter, a
# changed identity rule. It is written into data/parser_versions.json beside
# each season it scrapes, and the workflow refuses to re-fit model constants
# against a season recorded under an older version.
#
#   1  original patterns
#   2  nested parentheses in school names, empty mailing cities on completed
#      games, TBA/TBD/called-off records, non-varsity placeholders, and the
#      relaxed name-length bound
PARSER_VERSION = 2

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

# Every team on the scoreboard is written "School (City)", with an optional
# state tag for non-Ohio opponents: "West Orange (Winter Garden) [FL]".
# Requiring the parenthesised city makes this pattern precise enough to scan
# the whole page at once.
#
# The school name may ITSELF contain a parenthetical, and the original pattern
# could not match those at all -- it read one "(...)" and stopped:
#
#     St Xavier (Louisville) (Louisville) [KY]   disambiguated in the name
#     Trinity (Louisville) (Louisville) [KY]     (Cincinnati has its own St X)
#     Landmark Eagles (club) (Cincinnati)        club sides are marked
#     Valley (Wetzel) (Pine Grove) [WV]          county in the name
#     University Prep (USO co-op) (Pittsburgh) [PA]
#     Football North (via Clarkson SS) (Mississauga) [ON]
#
# The rule that resolves it: the LAST parenthetical is always the mailing city,
# and anything before it belongs to the school. Because the stem is non-greedy
# the engine tries the shortest name first and extends only when the rest of
# the record won't match, which lands on exactly that split. The two branches
# are disjoint (a word character can never be "("), so there is one way to
# consume each position and no backtracking blowup.
#
# The city may be EMPTY -- "Flint Beecher () [MI]". That used to be allowed for
# fixtures but not for completed games, which silently skipped any played game
# involving one. Both now use this single pattern.
#
# The bracketed tag is usually a two-letter state or province, but not always:
# "TBD" marks an opponent the site has not settled, and the Department of
# Defense schools abroad carry a country -- "Humphreys (Pyeongtaek) [South
# Korea]", "American School in Japan (Chofu, Tokyo) [Japan]". Accepting only
# two letters left those as UNRECOGNISED forever, which matters less for the
# games (none of them will ever play an Ohio school) than for the alarm: that
# count is how a change in the page format announces itself, and a number that
# is never zero cannot announce anything.
#
# TBD parses here and is then discarded by name in the scrape loop, for the
# same reason -- a placeholder is a real record, just not a real game.
_TEAM_WORD = r"[A-Za-z0-9'&.,\-/ ]"
_TEAM_INNER = r"\([^()]{1,30}\)"     # a parenthetical that is part of the NAME
_TEAM_CITY = r"\([^()]{0,40}\)"      # the mailing city, always last, may be empty
_TEAM_STATE = r"(?:\s*\[[A-Za-z][A-Za-z .'\-]{1,19}\])?"
# Two guards on the stem, and both are load-bearing.
#
# The old stem could not cross a "(" so it always halted within a few
# characters. This one may consume whole parentheticals, and that freedom is
# dangerous in two different ways:
#
#   Cost.   Unbounded, it runs from every start position to the end of the page
#           looking for a terminator that is not there -- quadratic on a 450
#           record scoreboard, and it took the probe from 0.02s to 6.3s. Hence
#           the {0,60} repetition cap.
#
#   Truth.  The cap alone is not enough, because one repetition can swallow a
#           whole "(...)" group. A record with no scores -- a cancelled game --
#           gives the pattern nothing to stop on, so it ran straight through
#           into the NEXT record and ate a real game:
#
#             2023-08-25 7pm Trimble (Glouster) at River (Hannibal) cancel
#             2023-08-25 7pm Tri-Village (New Madison) 48 at Preble Shawnee (Camden) 14
#             ^ away matched all of this, ascore matched 48, and Tri-Village's
#               game silently disappeared. Four real Ohio games went this way.
#
#           The ISO date is what separates one record from the next, so the
#           stem is forbidden from containing one. A team name cannot cross a
#           record boundary, whatever else it contains.
_NOT_A_NEW_RECORD = r"(?!\d{4}-\d{2}-\d{2})"
SB_TEAM = (rf"[A-Za-z](?:{_NOT_A_NEW_RECORD}(?:{_TEAM_WORD}|{_TEAM_INNER}))"
           rf"{{0,60}}?{_TEAM_CITY}{_TEAM_STATE}")
SB_TIME = r"(?:\d{1,2}(?::\d{2})?\s*[apAP]\.?[mM]\.?|\d{1,2}:\d{2}|[Nn]oon|TBA|TBD)"

# A whole game, found anywhere in the page's flattened text. The ISO date is
# the anchor -- it starts every record and appears nowhere else.
#
# This replaces line-based parsing because the site splits a single game
# across multiple elements, putting the away team, its score, and the home
# team in different nodes. Any approach that reads one line at a time sees
# fragments; flattening first and anchoring on the date sees whole games.
SB_GAME_RE = re.compile(
    rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s+(?:{SB_TIME}\s+)?"
    rf"(?P<away>{SB_TEAM})\s+(?P<ascore>\d{{1,3}})\s+"
    rf"(?P<sep>at|vs\.?)\s+"
    rf"(?P<home>{SB_TEAM})\s+(?P<hscore>\d{{1,3}})(?!\d)",
    re.IGNORECASE,
)

# A scheduled, not-yet-played game. Confirmed from a workflow-log probe of the
# 2026 week 2 scoreboard: 466 records, every one of them shaped
#
#   2026-08-27 6:30pm Deer Park (Cincinnati) *** at Shroder (Cincinnati) ***
#   2026-08-27 7pm Weir (Weirton) [WV] *** at Oak Glen (New Cumberland) [WV] ***
#   2026-08-28 Lewis County (Vanceburg) [KY] *** at Morgan County (West Liberty) [KY] ***
#
# Both scores are replaced by '***' and the kickoff time is sometimes absent.
#
# No neutral-site marker ('vs.') appeared anywhere in the 466. 'vs.' is still
# accepted here because completed games do use it, and a neutral fixture
# appearing later must not be silently dropped.
#
# Fixtures and completed games share SB_TEAM now. They used to differ only in
# whether an empty city was allowed, and the stricter half was a bug: a *played*
# game involving "Bath County () [KY]" was skipped without trace.
SB_TEAM_EMPTY = SB_TEAM
SB_FUTURE_RE = re.compile(
    rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s+(?:(?P<time>{SB_TIME})\s+)?"
    rf"(?P<away>{SB_TEAM_EMPTY})\s+\*\*\*\s+"
    rf"(?P<sep>at|vs\.?)\s+"
    rf"(?P<home>{SB_TEAM_EMPTY})\s+\*\*\*",
    re.IGNORECASE,
)

EMPTY_CITY_RE = re.compile(r"\s*\(\s*\)\s*$")

# "St Xavier (Louisville) (Louisville)" -- the site appends the mailing city
# even when the school name already ends with it. Collapsing the repeat keeps
# the identity unique against Cincinnati's St Xavier while staying readable.
# Deliberately narrow: it fires only when the last two parentheticals are
# identical, so "Landmark Eagles (club) (Cincinnati)" is left alone.
REPEATED_CITY_RE = re.compile(r"\(([^()]{1,40})\)\s*\(\1\)\s*$")

# An opponent the site has not settled yet: "TBD () [TBD]". A real record, but
# not a real game -- there is no team to rate or predict against. Dropped by
# name and counted separately, so it neither invents a team called TBD nor
# inflates the UNRECOGNISED count.
# "Non-varsity opponent" is the site's way of recording that a varsity side
# played someone who is not a varsity programme. It is a label, not a school,
# and left alone it becomes ONE rated entity that a dozen unrelated teams have
# results against -- which quietly distorts the strength of schedule of every
# one of them. Dropped for the same reason as TBD: the record is real, the
# opponent is not.
PLACEHOLDER_RE = re.compile(
    r"^(?:TBA|TBD|OPEN|BYE|Non[- ]varsity(?:\s+opponent)?)$", re.IGNORECASE)

# A game called off. The site writes the outcome where the scores would go:
#
#   2026-08-20 6pm Expression Prep Academy (Huntington) [WV] at Foxfire (Zanesville) cancel
#
# It has no result to rate and is no longer a fixture to predict, so it is
# recognised and dropped -- deliberately, and counted with the placeholders
# rather than left to look like a parser failure.
CALLED_OFF = r"(?:cancel(?:l?ed)?|ppd|postponed|forfeit(?:ed)?|no\s+contest)"
SB_CALLED_OFF_RE = re.compile(
    rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})\s+(?:{SB_TIME}\s+)?"
    rf"(?P<away>{SB_TEAM})\s+(?:\*\*\*\s+)?(?:at|vs\.?)\s+"
    rf"(?P<home>{SB_TEAM})\s+(?:\*\*\*\s+)?{CALLED_OFF}",
    re.IGNORECASE,
)


def _is_placeholder(name):
    return bool(PLACEHOLDER_RE.match((name or "").strip()))

# Same idea for the ranking pages. The "Current Average" always carries four
# decimals, which terminates the free-text city+school run reliably.
RANK_FLAT_RE = re.compile(
    r"(?P<rank>\d{1,3}t?)\s+"
    r"(?P<w>\d{1,2})-(?P<l>\d{1,2})(?:-(?P<t>\d{1,2}))?\s+"
    r"(?P<sid>\d{1,6})\s+"
    r"(?P<middle>[A-Za-z][A-Za-z0-9'&.,\-/ ]{1,70}?)\s+"
    r"(?P<harbin>\d+\.\d{4})(?!\d)"
)

STATE_TAG_RE = re.compile(r"\s*\[([A-Za-z][A-Za-z .'\-]{1,19})\]\s*$")


def flat_text(html):
    """The entire page as one whitespace-normalised string.

    Deliberately structure-blind: the markup splits records across elements,
    so any structure we try to honour is structure we get wrong.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

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


# Bounds on a team name, not a guess at one.
#
# The old limits -- 48 characters, 6 words -- were set when this scraper read
# BARE names off the scoreboard ("Antwerp"). Names carry the mailing city now,
# which is far longer, and the limits were never re-cut. The result was a
# filter that rejected real OHSAA schools by construction:
#
#     Cuyahoga Valley Christian Academy (Cuyahoga Falls)   50 chars
#     Brecksville-Broadview Heights (Broadview Heights)    49 chars
#
# Both lost every game of the season, in silence. The longest legitimate name
# on the 2026 roster is 50 characters, so a 48-character ceiling was below the
# real maximum -- the bound was simply wrong, not merely tight.
#
# These are deliberately generous. They exist to reject page furniture, and the
# junk-word check below is what actually does that work; length is a backstop.
# tests/test_schedule.py asserts every name on the committed roster passes, so
# this can never drift under the real data again.
MAX_TEAM_CHARS = 80
MAX_TEAM_WORDS = 12


def _plausible_team(s):
    """Reject prose and navigation chrome that happens to sit near numbers."""
    if not s or len(s) < 2 or len(s) > MAX_TEAM_CHARS:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    if len(s.split()) > MAX_TEAM_WORDS:
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


def _rows_from_flat(flat, known_pairs):
    """Scan the whole flattened ranking page, anchored on rank + record + id."""
    for m in RANK_FLAT_RE.finditer(flat):
        middle = _clean_name(m.group("middle"))
        school, city = _split_city_school(middle, known_pairs)
        if not _plausible_team(school):
            continue
        w, l = int(m.group("w")), int(m.group("l"))
        t = int(m.groupdict().get("t") or 0)
        if w > 16 or l > 16:
            continue
        yield (team_key(school, city), school, city, m.group("sid"),
               w, l, t, float(m.group("harbin")))


def _split_city_school(middle, known_pairs):
    """Split 'Massillon Jackson' into city and school.

    The site lists city first, then school, and both may contain spaces
    ('Cuyahoga Falls' + 'Cuyahoga Valley Christian Academy'), so the boundary
    is genuinely ambiguous. The scores already know every real (school, city)
    pair, so test the splits against those and only fall back to a guess when
    the team hasn't appeared in a game yet.
    """
    words = middle.split()
    for cut in range(1, len(words)):
        c, s = " ".join(words[:cut]), " ".join(words[cut:])
        if (s.lower(), c.lower()) in known_pairs:
            return s, c
    if len(words) >= 2:
        return " ".join(words[1:]), words[0]
    return middle, ""


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


def _diagnose(label, lines, limit=30, flat=None):
    """Print what we actually saw, so a failed run is one round trip to fix.

    The flattened sample matters more than the lines: this site splits single
    records across elements, so per-line output shows fragments while the flat
    text shows the record as the patterns actually see it.
    """
    print(f"\n  !! {label}: nothing matched.", file=sys.stderr)

    if flat is not None:
        print("\n     --- flattened text, first 1200 chars "
              "(this is what the patterns scan) ---", file=sys.stderr)
        print(f"     {flat[:1200]}", file=sys.stderr)

    print(f"\n     --- first {limit} individual lines ---", file=sys.stderr)
    shown = 0
    for ln in lines:
        if not ln or len(ln) > 200:
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


ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _shape(s):
    """Collapse a sample to its format signature.

    'Antwerp (Antwerp) *** at Montpelier (Montpelier)' and the 400 other rows
    shaped exactly like it are one finding, not 400. Letters and digits are
    flattened so only punctuation, keywords and layout survive.
    """
    s = ISO_DATE_RE.sub("\x00", s)          # placeholders survive the letter pass
    # Kickoff time is spelled several ways on one page -- "7pm", "7:30 PM",
    # "noon", "TBA". Left alone, each spelling becomes its own "format" and
    # buries the real distinction we are looking for.
    s = re.sub(SB_TIME, "\x01", s)
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"[A-Za-z]+", "w", s)
    s = re.sub(r"(?:w ?)+", "w ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("\x00", "<DATE>").replace("\x01", "<TIME>")


def probe_unscored(flat, label, samples=12, window=200):
    """Report every date-anchored record that the scored-game pattern missed.

    SB_GAME_RE requires both scores, so an unplayed game is invisible to it --
    which is precisely why we have never seen how one is written. Anchoring on
    the ISO date instead finds every record on the page, and the difference
    between those two sets is the future-game format we need.

    This prints and returns; it never parses. The point is to look before
    writing a pattern, because the last two pattern rewrites were built from a
    reassembled view of the page that does not exist in the HTML.
    """
    played_at = {m.start() for m in SB_GAME_RE.finditer(flat)}
    sched_at = {m.start() for m in SB_FUTURE_RE.finditer(flat)}
    off_at = {m.start() for m in SB_CALLED_OFF_RE.finditer(flat)}
    known = played_at | sched_at | off_at
    dates = list(ISO_DATE_RE.finditer(flat))
    unmatched = [m for m in dates if m.start() not in known]

    print(f"\n  == PROBE: {label} ==", file=sys.stderr)
    print(f"     date-anchored records on page : {len(dates)}", file=sys.stderr)
    print(f"     matched as completed games    : {len(played_at)}", file=sys.stderr)
    print(f"     matched as scheduled games    : {len(sched_at)}", file=sys.stderr)
    print(f"     called off (not a game)       : {len(off_at)}", file=sys.stderr)
    print(f"     UNRECOGNISED                  : {len(unmatched)}", file=sys.stderr)

    for token in ("***", "TBA", "TBD", "vs.", " at ", "()"):
        print(f"     literal {token!r:8} appears        : {flat.count(token)}",
              file=sys.stderr)

    if not unmatched:
        print("     (every record on the page is accounted for)", file=sys.stderr)
        return []

    # One record per sample. Without this the window runs on into the next
    # game and every sample is unique, which defeats the whole point of
    # grouping -- 400 identical rows would print as 400 distinct "formats".
    starts = [m.start() for m in dates]
    by_shape = {}
    for m in unmatched:
        nxt = next((s for s in starts if s > m.start()), len(flat))
        chunk = flat[m.start(): min(nxt, m.start() + window)].strip()
        by_shape.setdefault(_shape(chunk), []).append(chunk)

    print(f"\n     --- {len(by_shape)} distinct format(s), most common first ---",
          file=sys.stderr)
    ordered = sorted(by_shape.items(), key=lambda kv: -len(kv[1]))
    for shape, chunks in ordered[:samples]:
        print(f"\n     [{len(chunks)}x] shape: {shape}", file=sys.stderr)
        for c in chunks[:2]:
            print(f"           | {c}", file=sys.stderr)
    print("", file=sys.stderr)
    return [c for _, chunks in ordered for c in chunks[:2]]


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


def _split_state(name):
    """'West Orange (Winter Garden) [FL]' -> ('West Orange (Winter Garden)', 'FL')."""
    m = STATE_TAG_RE.search(name)
    if m:
        return _clean_name(name[: m.start()]), m.group(1).upper()
    return _clean_name(name), ""


class WeekResult(NamedTuple):
    games: list        # completed, with scores -- these drive the rating fit
    scheduled: list    # fixtures with no result yet -- predictions only
    residual: int      # date-anchored records neither pattern recognised
    placeholders: int  # records parsed, then dropped as "TBD" non-games
    flat: str


def _team_name(raw):
    """Scoreboard text -> the (name, state) a team is known by.

        'Flint Beecher () [MI]'                    -> ('Flint Beecher', 'MI')
        'St Xavier (Louisville) (Louisville) [KY]' -> ('St Xavier (Louisville)', 'KY')
        'Landmark Eagles (club) (Cincinnati)'      -> unchanged, '' 

    Two normalisations, both about identity rather than tidiness. An empty
    mailing city is written as bare parentheses, and left in place a team would
    appear as both 'Flint Beecher ()' and 'Flint Beecher' depending on which
    page named it. A mailing city that merely repeats what the school name
    already ends with is collapsed for the same reason.

    Used for completed games and fixtures alike -- the two must agree exactly
    or a team's results and its remaining schedule land on different entities.
    """
    name, state = _split_state(raw)
    name = REPEATED_CITY_RE.sub(r"(\1)", name)
    return _clean_name(EMPTY_CITY_RE.sub("", name)), state


_sched_name = _team_name   # the old name, kept for callers


def _placeholder_count(flat):
    """How many records parsed cleanly but named no real opponent.

    Reported next to UNRECOGNISED and never folded into it: one means "the page
    format moved and we are losing games", the other means "the site has not
    announced this matchup yet". Conflating them was why UNRECOGNISED could
    never reach zero, and an alarm that always rings is not an alarm.
    """
    n = sum(1 for _ in SB_CALLED_OFF_RE.finditer(flat))
    for rx in (SB_GAME_RE, SB_FUTURE_RE):
        for m in rx.finditer(flat):
            a, _ = _team_name(m.group("away"))
            h, _ = _team_name(m.group("home"))
            if _is_placeholder(a) or _is_placeholder(h):
                n += 1
    return n


def scrape_schedule(flat, week):
    """Not-yet-played fixtures on a scoreboard page.

    Kept strictly separate from completed games: these have no result, must
    never reach the rating fit, and exist only to be predicted.
    """
    out, seen = [], set()
    # A game the site has marked cancelled still carries '***' where its scores
    # would be, so the fixture pattern matches it. It is not a fixture.
    called_off = {m.start() for m in SB_CALLED_OFF_RE.finditer(flat)}
    for m in SB_FUTURE_RE.finditer(flat):
        if m.start() in called_off:
            continue
        away, astate = _team_name(m.group("away"))
        home, hstate = _team_name(m.group("home"))
        # A real record, but not a real game -- the site has not settled the
        # opponent yet. Skipped by name rather than left to fail the pattern.
        if _is_placeholder(away) or _is_placeholder(home):
            continue
        if not (_plausible_team(away) and _plausible_team(home)):
            continue
        if away.lower() == home.lower():
            continue
        key = (away.lower(), home.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "week": week,
            "date": m.group("date"),
            "time": (m.group("time") or "").strip(),
            "away": away,
            "home": home,
            "neutral": 1 if m.group("sep").lower().startswith("vs") else 0,
            "away_state": astate,
            "home_state": hstate,
        })
    return out


def scrape_week(sess, season, week, use_cache=True, diagnose=False, probe=False):
    html = fetch(sess, f"/hsfoot/scoreboard/{season}/week-{week}", use_cache)
    flat = flat_text(html)

    if probe:
        probe_unscored(flat, f"{season} week {week} scoreboard", samples=40)

    games, seen = [], set()
    called_off = {m.start() for m in SB_CALLED_OFF_RE.finditer(flat)}
    for m in SB_GAME_RE.finditer(flat):
        if m.start() in called_off:
            continue
        away, astate = _team_name(m.group("away"))
        home, hstate = _team_name(m.group("home"))
        if _is_placeholder(away) or _is_placeholder(home):
            continue
        if not (_plausible_team(away) and _plausible_team(home)):
            continue
        if away.lower() == home.lower():
            continue
        a, h = int(m.group("ascore")), int(m.group("hscore"))
        if a > 120 or h > 120:
            continue
        key = (away.lower(), home.lower())
        if key in seen:
            continue
        seen.add(key)
        games.append(
            {
                "week": week,
                "away": away,
                "away_score": a,
                "home": home,
                "home_score": h,
                "neutral": 1 if m.group("sep").lower().startswith("vs") else 0,
                "away_state": astate,
                "home_state": hstate,
            }
        )

    scheduled = scrape_schedule(flat, week)
    placeholders = _placeholder_count(flat)

    # Anything date-anchored that neither pattern claimed. Reported per week as
    # a single number so a format change shows up as a rising count in the log
    # rather than as games quietly going missing.
    known = ({m.start() for m in SB_GAME_RE.finditer(flat)}
             | {m.start() for m in SB_FUTURE_RE.finditer(flat)}
             | {m.start() for m in SB_CALLED_OFF_RE.finditer(flat)})
    residual = sum(1 for m in ISO_DATE_RE.finditer(flat) if m.start() not in known)

    if not games and not scheduled and diagnose:
        _diagnose(f"week {week} scoreboard", page_lines(html), flat=flat)
    return WeekResult(games=games, scheduled=scheduled, residual=residual,
                      placeholders=placeholders, flat=flat)


def scrape_roster(sess, season, use_cache=True, known_pairs=frozenset()):
    rows = []
    empty_regions = []
    for region in range(1, 29):
        html = fetch(sess, f"/hsfoot/rankings/{season}/region-{region}", use_cache)

        parsed = list(_rows_from_table(page_rows(html)))
        source = "table"
        if not parsed:
            parsed = list(_rows_from_flat(flat_text(html), known_pairs))
            source = "flat"
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
                _diagnose(f"region {region} rankings", page_lines(html),
                          flat=flat_text(html))

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
    ap.add_argument(
        "--probe-weeks", default="",
        help="Comma-separated weeks to dump unmatched date-anchored records "
             "for, then exit without writing anything. Reconnaissance only.",
    )
    args = ap.parse_args()

    sess = _session()
    use_cache = not args.no_cache
    os.makedirs(DATA, exist_ok=True)

    if args.probe_weeks:
        for wk in [int(w) for w in args.probe_weeks.split(",") if w.strip()]:
            try:
                scrape_week(sess, args.season, wk, use_cache, probe=True)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "?"
                print(f"\n  == PROBE: week {wk} -> HTTP {code} "
                      f"(no such page) ==", file=sys.stderr)
        return

    # Scores are read first: they establish the (school, city) pairs that the
    # roster's text fallback needs to split its city and school columns.
    # The loop no longer stops at the first unplayed week. That week is exactly
    # where the remaining schedule lives, and the season's fixtures are what
    # the simulator predicts. It stops on a 404, or on a page holding no
    # records of either kind -- i.e. genuinely past the end of the season.
    all_games, all_sched = [], []
    probed = 0
    print("games:", file=sys.stderr)
    for wk in range(1, args.through_week + 1):
        try:
            # Diagnose on week 1 only -- if that one is empty the pattern is
            # wrong, and dumping every week would bury the useful output.
            r = scrape_week(sess, args.season, wk, use_cache, diagnose=(wk == 1))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                break  # no such page: past the end of the season
            raise

        if not r.games and not r.scheduled:
            print(f"  week {wk:>2}: nothing on the page -- stopping",
                  file=sys.stderr)
            break

        note = f"  week {wk:>2}: {len(r.games):>3} played, {len(r.scheduled):>3} scheduled"
        if r.placeholders:
            note += f"  ({r.placeholders} TBD)"
        if r.residual:
            note += f"  [{r.residual} UNRECOGNISED]"
        print(note, file=sys.stderr)

        # Detail for the first two weeks that hold anything we cannot read.
        # Capped because mid-season every remaining week would repeat it.
        if r.residual and probed < 2:
            probed += 1
            probe_unscored(r.flat, f"{args.season} week {wk} scoreboard", samples=40)

        all_games.extend(r.games)
        all_sched.extend(r.scheduled)

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
            fh,
            fieldnames=["week", "away", "away_score", "home", "home_score",
                        "neutral", "away_state", "home_state"],
        )
        wtr.writeheader()
        wtr.writerows(all_games)
    print(f"games: {len(all_games)} -> {gpath}", file=sys.stderr)

    # A fixture that has since been played is a completed game, not a fixture.
    # The site rewrites '***' into scores so this should never fire, but a
    # duplicated game would show a team two predicted opponents in one week.
    played_keys = {(g["week"], g["away"].lower(), g["home"].lower())
                   for g in all_games}
    dropped = [s for s in all_sched
               if (s["week"], s["away"].lower(), s["home"].lower()) in played_keys]
    all_sched = [s for s in all_sched
                 if (s["week"], s["away"].lower(), s["home"].lower()) not in played_keys]
    if dropped:
        print(f"  ({len(dropped)} fixtures dropped -- already played)",
              file=sys.stderr)

    spath = os.path.join(DATA, f"schedule_{args.season}.csv")
    with open(spath, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(
            fh,
            fieldnames=["week", "date", "time", "away", "home",
                        "neutral", "away_state", "home_state"],
        )
        wtr.writeheader()
        wtr.writerows(all_sched)
    print(f"schedule: {len(all_sched)} fixtures -> {spath}", file=sys.stderr)

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

    # Record which parser produced this season, so a later re-fit can tell
    # whether the committed history still matches the current code. File
    # mtimes cannot answer that: git does not preserve them, so on a fresh
    # checkout every file carries the same timestamp.
    vpath = os.path.join(DATA, "parser_versions.json")
    try:
        with open(vpath, encoding="utf-8") as fh:
            versions = json.load(fh)
    except (OSError, ValueError):
        versions = {}
    versions[str(args.season)] = PARSER_VERSION
    with open(vpath, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(versions.items())), fh, indent=1)
        fh.write("\n")
    print(f"parser version {PARSER_VERSION} recorded for {args.season}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
