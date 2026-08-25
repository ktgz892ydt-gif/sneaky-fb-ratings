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

from scrape import SB_GAME_RE, scrape_schedule  # noqa: E402


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
