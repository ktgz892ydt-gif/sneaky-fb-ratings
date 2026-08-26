"""
Scheduled (not-yet-played) fixtures.

Every sample below is a real line, copied verbatim out of the workflow-log
probe of the 2026 week 2 scoreboard. That matters: the two previous parser
rewrites were built from a reassembled view of the page that does not exist
in the HTML, and both failed. These strings are what the parser actually
sees.

The property this file protects above all others: a fixture must never be
mistaken for a result. A scheduled game has no score, and if one leaked into
the rating fit it would be a phantom 0-0 tie between two real teams.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scrape import (MAX_TEAM_CHARS, MAX_TEAM_WORDS, SB_GAME_RE,  # noqa: E402
                    _plausible_team, _team_name, scrape_schedule)


# Verbatim from the probe output, one per distinct format it reported.
REAL = {
    "plain": "2026-08-27 6:30pm Deer Park (Cincinnati) *** at Shroder (Cincinnati) ***",
    "no_minutes": "2026-08-27 6pm Dunbar (Dayton) *** at Stivers (Dayton) ***",
    "both_oos": "2026-08-27 7pm Weir (Weirton) [WV] *** at Oak Glen (New Cumberland) [WV] ***",
    "hyphen_home": "2026-08-28 7pm Barberton (Barberton) *** at Stow-Munroe Falls (Stow) ***",
    "hyphen_away": "2026-08-28 7pm Berea-Midpark (Berea) *** at Euclid (Euclid) ***",
    "empty_city": "2026-08-27 7pm Flint Beecher () [MI] *** at Petersburg Summerfield (Petersburg) [MI] ***",
    "home_oos": "2026-08-28 7pm Beaver Local (East Liverpool) *** at Western Beaver (Industry) [PA] ***",
    "away_oos": "2026-08-27 6pm Portage Central (Portage) [MI] *** at Central Catholic (Toledo) ***",
    "no_time": "2026-08-28 Lewis County (Vanceburg) [KY] *** at Morgan County (West Liberty) [KY] ***",
    "no_time_empty_city": "2026-08-28 Newport Central Catholic () [KY] *** at Paintsville (Paintsville) [KY] ***",
}


def test_every_observed_format_parses():
    for label, line in REAL.items():
        got = scrape_schedule(line, week=2)
        assert len(got) == 1, f"{label} did not parse: {line}"


def test_a_page_of_mixed_formats_yields_one_row_each():
    flat = " ".join(REAL.values())
    assert len(scrape_schedule(flat, week=2)) == len(REAL)


def test_teams_and_sides_are_read_correctly():
    g = scrape_schedule(REAL["plain"], week=2)[0]
    assert g["away"] == "Deer Park (Cincinnati)"
    assert g["home"] == "Shroder (Cincinnati)"
    assert g["week"] == 2
    assert g["date"] == "2026-08-27"
    assert g["time"] == "6:30pm"
    assert g["neutral"] == 0


def test_out_of_state_tags_are_split_off_the_name():
    g = scrape_schedule(REAL["both_oos"], week=2)[0]
    assert g["away"] == "Weir (Weirton)", g["away"]
    assert g["home"] == "Oak Glen (New Cumberland)", g["home"]
    assert g["away_state"] == "WV"
    assert g["home_state"] == "WV"


def test_empty_city_parentheses_are_removed_from_identity():
    """'Flint Beecher ()' and 'Flint Beecher' must not become two teams."""
    g = scrape_schedule(REAL["empty_city"], week=2)[0]
    assert g["away"] == "Flint Beecher", g["away"]
    assert g["away_state"] == "MI"
    assert "(" not in g["away"]


def test_a_missing_kickoff_time_is_not_fatal():
    g = scrape_schedule(REAL["no_time"], week=2)[0]
    assert g["time"] == ""
    assert g["away"] == "Lewis County (Vanceburg)"
    assert g["home"] == "Morgan County (West Liberty)"


def test_hyphenated_school_names_survive():
    assert scrape_schedule(REAL["hyphen_home"], week=2)[0]["home"] == "Stow-Munroe Falls (Stow)"
    assert scrape_schedule(REAL["hyphen_away"], week=2)[0]["away"] == "Berea-Midpark (Berea)"


def test_a_neutral_site_fixture_is_flagged():
    """No 'vs.' appeared in the 466 fixtures probed, but completed games use
    it, so a neutral fixture appearing later must not be dropped or
    mislabelled as a home game."""
    line = "2026-08-29 7pm Marion Local (Maria Stein) *** vs. Coldwater (Coldwater) ***"
    g = scrape_schedule(line, week=2)
    assert len(g) == 1
    assert g[0]["neutral"] == 1


# ------------------------------------------- the separation that must hold

def test_a_completed_game_is_never_read_as_a_fixture():
    played = "2026-08-20 7pm Antwerp (Antwerp) 14 at Montpelier (Montpelier) 21"
    assert scrape_schedule(played, week=1) == []


def test_a_fixture_is_never_read_as_a_completed_game():
    """The one that would corrupt the ratings: a phantom 0-0 result."""
    for label, line in REAL.items():
        assert not SB_GAME_RE.search(line), \
            f"{label} was matched as a played game -- this would inject a fake result"


def test_a_mixed_page_separates_cleanly():
    flat = ("2026-08-20 7pm Antwerp (Antwerp) 14 at Montpelier (Montpelier) 21 "
            + REAL["plain"] + " " + REAL["both_oos"])
    fixtures = scrape_schedule(flat, week=2)
    assert len(fixtures) == 2
    assert all("Antwerp" not in f["away"] for f in fixtures)
    assert len(SB_GAME_RE.findall(flat)) == 1


def test_duplicate_fixtures_are_collapsed():
    flat = " ".join([REAL["plain"]] * 3)
    assert len(scrape_schedule(flat, week=2)) == 1


def test_a_team_scheduled_against_itself_is_rejected():
    line = "2026-08-27 7pm Xenia (Xenia) *** at Xenia (Xenia) ***"
    assert scrape_schedule(line, week=2) == []


# ---------------------------------------------------------------------------
# Schools whose NAME contains parentheses
#
# Every string below is verbatim from the 2026 scoreboard. The original pattern
# read one "(...)" and stopped, so none of these matched at all and the games
# vanished -- Elder lost its week 5 fixture this way, and the reason it was
# confusing is that Cincinnati's St Xavier parses fine while Louisville's does
# not. The rule that separates them: the LAST parenthetical is the mailing
# city, everything before it belongs to the school.
# ---------------------------------------------------------------------------

NESTED = {
    "disambiguated_school": (
        "2026-09-18 7pm St Xavier (Louisville) (Louisville) [KY] *** at Elder (Cincinnati) ***",
        "St Xavier (Louisville)", "Elder (Cincinnati)"),
    "disambiguated_home": (
        "2026-09-04 7pm Archbishop Moeller (Cincinnati) *** at Trinity (Louisville) (Louisville) [KY] ***",
        "Archbishop Moeller (Cincinnati)", "Trinity (Louisville)"),
    "club_side_home": (
        "2026-09-05 7pm Adena (Frankfort) *** at Landmark Eagles (club) (Cincinnati) ***",
        "Adena (Frankfort)", "Landmark Eagles (club) (Cincinnati)"),
    "club_side_away": (
        "2026-08-29 4pm Noblesville Lions (club) (Noblesville) [IN] *** at Foxfire (Zanesville) ***",
        "Noblesville Lions (club) (Noblesville)", "Foxfire (Zanesville)"),
    "county_in_name": (
        "2026-09-18 7pm Valley (Wetzel) (Pine Grove) [WV] *** at Frontier (New Matamoras) ***",
        "Valley (Wetzel) (Pine Grove)", "Frontier (New Matamoras)"),
    "co_op_in_name": (
        "2026-09-04 7pm University Prep (USO co-op) (Pittsburgh) [PA] *** at Girard (Girard) ***",
        "University Prep (USO co-op) (Pittsburgh)", "Girard (Girard)"),
    "three_letter_tag": (
        "2026-09-25 7pm Football North (via Clarkson SS) (Mississauga) [ON] *** at Massillon Washington (Massillon) ***",
        "Football North (via Clarkson SS) (Mississauga)", "Massillon Washington (Massillon)"),
}


def test_a_school_whose_name_contains_parentheses_parses():
    for label, (line, away, home) in NESTED.items():
        got = scrape_schedule(line, week=5)
        assert len(got) == 1, f"{label}: parsed {len(got)} fixtures, expected 1"
        assert got[0]["away"] == away, f"{label}: away = {got[0]['away']!r}"
        assert got[0]["home"] == home, f"{label}: home = {got[0]['home']!r}"


def test_a_repeated_mailing_city_is_collapsed():
    """'St Xavier (Louisville) (Louisville)' -> 'St Xavier (Louisville)'.

    Narrow on purpose: only when the last two parentheticals are identical.
    """
    got = scrape_schedule(NESTED["disambiguated_school"][0], week=5)[0]
    assert got["away"] == "St Xavier (Louisville)"
    assert got["away_state"] == "KY"


def test_a_differing_second_parenthetical_is_left_alone():
    """'Landmark Eagles (club) (Cincinnati)' keeps both -- they aren't a repeat."""
    got = scrape_schedule(NESTED["club_side_home"][0], week=5)[0]
    assert got["home"] == "Landmark Eagles (club) (Cincinnati)"


def test_the_two_st_xaviers_stay_distinct():
    lou = scrape_schedule(NESTED["disambiguated_school"][0], week=5)[0]
    cin = scrape_schedule(
        "2026-10-02 7pm St Xavier (Cincinnati) *** at Elder (Cincinnati) ***", week=7)[0]
    assert lou["away"] != cin["away"]
    assert (lou["away"], lou["away_state"]) == ("St Xavier (Louisville)", "KY")
    assert (cin["away"], cin["away_state"]) == ("St Xavier (Cincinnati)", "")


# ---------------------------------------------------------------------------
# Placeholder opponents
# ---------------------------------------------------------------------------

def test_a_tbd_opponent_is_not_a_game():
    """'TBD () [TBD]' parses as a record and is then discarded by name.

    It must not become a team called TBD, and it must not be left to fail the
    pattern -- that inflated the UNRECOGNISED count, which is the alarm for the
    page format moving.
    """
    line = ("2026-08-21 TBD () [TBD] *** at "
            "Cornerstone Christian (San Antonio) [TX] ***")
    assert scrape_schedule(line, week=1) == []


def test_a_real_game_is_still_a_game():
    assert len(scrape_schedule(REAL["plain"], week=2)) == 1


# ---------------------------------------------------------------------------
# The name filter, checked against the real roster rather than a guess
# ---------------------------------------------------------------------------

def test_every_name_on_the_committed_roster_survives_the_filter():
    """The regression that cost two schools their entire season.

    _plausible_team's limits were set for bare names and never re-cut once the
    mailing city was appended, so the ceiling (48 chars) ended up BELOW the
    longest real OHSAA name (50). Cuyahoga Valley Christian Academy and
    Brecksville-Broadview Heights lost all ten games each, silently.

    Asserting against the committed roster means the bound can never drift
    under the real data again -- whatever the source starts publishing, this
    fails the build rather than dropping teams.
    """
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "roster_2026.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]
    assert names, "roster fixture is empty"
    rejected = [n for n in names if not _plausible_team(n)]
    assert not rejected, f"_plausible_team rejects real OHSAA schools: {rejected}"


def test_the_longest_real_name_has_headroom():
    """Not just passing -- passing with room, so the next long name is fine."""
    import csv
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "roster_2026.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]
    assert max(len(n) for n in names) < MAX_TEAM_CHARS - 10
    assert max(len(n.split()) for n in names) < MAX_TEAM_WORDS - 2


# ---------------------------------------------------------------------------
# Completed games: the empty-city case that used to be skipped outright
# ---------------------------------------------------------------------------

def test_a_played_game_with_an_empty_city_is_read():
    """SB_TEAM used to demand 1-40 chars inside the parentheses, so a *played*
    game involving an empty-city team was skipped without trace."""
    m = SB_GAME_RE.search(
        "2026-08-21 7:30pm Fairview (Ashland) [KY] 34 at Bath County () [KY] 13")
    assert m is not None
    assert _team_name(m.group("away")) == ("Fairview (Ashland)", "KY")
    assert _team_name(m.group("home")) == ("Bath County", "KY")


def test_a_played_game_against_a_parenthesised_school_is_read():
    m = SB_GAME_RE.search(
        "2026-08-21 7pm Trinity (Louisville) (Louisville) [KY] 59 at Anderson (Cincinnati) 13")
    assert m is not None
    assert _team_name(m.group("away")) == ("Trinity (Louisville)", "KY")
    assert int(m.group("ascore")) == 59 and int(m.group("hscore")) == 13


def test_a_full_page_of_unparseable_records_stays_fast():
    """A bounded stem, asserted rather than assumed.

    SB_TEAM's repetition may consume whole parentheticals, so an unbounded
    version can run from every start position to the end of the page looking
    for a terminator that is not there -- quadratic on a real scoreboard. This
    took the probe from 0.02s to 6.3s before the bound went in. A real page
    holds ~470 records; 400 here with nothing to match is the worst case.
    """
    import time
    flat = " ".join(f"2026-08-28 7pm Aa{i} (Cc{i}) at Bb{i} (Dd{i})" for i in range(400))
    start = time.perf_counter()
    scrape_schedule(flat, week=2)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"parsing a dead page took {elapsed:.2f}s -- stem unbounded?"


def test_a_called_off_game_is_neither_a_result_nor_a_fixture():
    """The site writes the outcome where the scores go, and '***' can remain,
    so the fixture pattern matches it. It is not a fixture."""
    line = ("2026-08-20 6pm Expression Prep Academy (Huntington) [WV] at "
            "Foxfire (Zanesville) cancel")
    assert scrape_schedule(line, week=1) == []
    assert SB_GAME_RE.search(line) is None


def test_an_overseas_tag_is_read_as_a_place_not_a_failure():
    """DoD schools abroad carry a country, not a two-letter code."""
    line = ("2026-09-04 5am American School in Japan (Chofu, Tokyo) [Japan] *** at "
            "Humphreys (Pyeongtaek) [South Korea] ***")
    got = scrape_schedule(line, week=3)
    assert len(got) == 1
    # Upper-cased like every other tag -- that normalisation is what stops
    # "[ky]" and "[KY]" resolving to two different schools.
    assert got[0]["away_state"] == "JAPAN"
    assert got[0]["home_state"] == "SOUTH KOREA"
    assert got[0]["home"] == "Humphreys (Pyeongtaek)"
