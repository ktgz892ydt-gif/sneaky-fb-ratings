"""
The future-game probe.

This is reconnaissance tooling, not a parser. Its whole job is to show what a
scoreboard page really contains before anyone writes a pattern against it --
because the last two pattern rewrites were built from a reassembled view of
the page that does not exist in the HTML, and both failed.

So these tests cover the *mechanism*: does it separate completed games from
everything else, and does it collapse hundreds of identical rows into one
finding. They deliberately do NOT assert any particular future-game format.
Asserting a format we have not yet observed is the exact mistake this tool
exists to prevent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scrape import _shape, probe_unscored  # noqa: E402


COMPLETED = "2026-08-20 7pm Antwerp (Antwerp) 14 at Montpelier (Montpelier) 21"


def bulk(n, fmt):
    return " ".join(fmt.format(i=i) for i in range(n))


def test_completed_games_are_not_reported_as_futures():
    """A fully played week must probe clean, or the signal is worthless."""
    flat = " ".join([COMPLETED,
                     "2026-08-20 7pm Hoban (Akron) 24 at St Edward (Lakewood) 28"])
    assert probe_unscored(flat, "played week") == []


def test_a_record_without_scores_is_surfaced():
    flat = COMPLETED + " 2026-08-28 7pm Kirtland (Kirtland) at Perry (Perry)"
    found = probe_unscored(flat, "mixed week")
    assert len(found) == 1
    assert "Kirtland" in found[0]
    assert "Antwerp" not in found[0], "the completed game leaked into the sample"


def test_identical_rows_collapse_to_one_finding():
    """400 scheduled games are one format, not 400 formats.

    Regression: an over-wide sample window ran on into the following record,
    making every sample unique and the grouping useless.
    """
    flat = bulk(300, "2026-08-28 7pm A{i} (C{i}) at B{i} (D{i})")
    shapes = {_shape(s) for s in probe_unscored(flat, "bulk")}
    assert len(shapes) == 1, f"expected one shape, got {shapes}"


def test_distinct_formats_stay_distinct():
    flat = (bulk(50, "2026-08-28 7pm A{i} (C{i}) at B{i} (D{i})") + " "
            + bulk(50, "2026-08-29 TBA E{i} (F{i}) vs. G{i} (H{i})"))
    shapes = {_shape(s) for s in probe_unscored(flat, "two formats")}
    assert len(shapes) == 2, f"expected two shapes, got {shapes}"


def test_shape_ignores_names_and_numbers_but_keeps_punctuation():
    a = _shape("2026-08-28 7pm Kirtland (Kirtland) *** at Perry (Perry) ***")
    b = _shape("2026-09-04 7:30 PM Marion Local (Maria Stein) *** at Coldwater (Coldwater) ***")
    assert a == b, "same layout must produce the same signature"
    assert "***" in a, "the marker that distinguishes formats must survive"

    c = _shape("2026-08-28 7pm Kirtland (Kirtland) at Perry (Perry)")
    assert a != c, "presence of a marker must change the signature"


def test_probe_survives_a_page_with_no_dates_at_all():
    assert probe_unscored("Ohio high school football scoreboard", "empty") == []
