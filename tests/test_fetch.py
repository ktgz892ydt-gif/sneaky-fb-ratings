"""
Fetching from a one-person server.

joeeitel.com has run since 2000 off one person's hosting. A single 502 or a
dropped connection used to abort the whole weekly run before anything was
written -- a good week's ratings lost to a blip that would have cleared on the
next request seconds later.

The distinction these tests exist to pin: a 5xx or a timeout is a transient
fault worth retrying, and a 4xx is not. A 404 in particular is not a fault at
all -- it is the sentinel that tells the scraper the season has no further
weeks. Retrying it would triple the requests at the end of every season and
make the normal stop slower.
"""

import os
import sys
import tempfile
import types

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import scrape  # noqa: E402


class Resp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


@pytest.fixture(autouse=True)
def _no_sleeping_or_caching(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape, "DELAY", 0)
    monkeypatch.setattr(scrape, "CACHE", str(tmp_path))


def session(*responses):
    """A stand-in session that hands back `responses` in order, raising any
    that are exceptions. Records how many times it was called."""
    it, calls = iter(responses), []

    def get(url, timeout=None):
        calls.append(url)
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    return types.SimpleNamespace(get=get, calls=calls)


def test_a_transient_502_is_retried_and_succeeds():
    s = session(Resp(502), Resp(200, "<html>ok</html>"))
    assert scrape.fetch(s, "/a", use_cache=False) == "<html>ok</html>"
    assert len(s.calls) == 2


def test_a_timeout_is_retried():
    s = session(requests.Timeout("slow"), Resp(200, "ok"))
    assert scrape.fetch(s, "/b", use_cache=False) == "ok"
    assert len(s.calls) == 2


def test_a_dropped_connection_is_retried():
    s = session(requests.ConnectionError("reset"), Resp(200, "ok"))
    assert scrape.fetch(s, "/c", use_cache=False) == "ok"
    assert len(s.calls) == 2


def test_a_404_is_never_retried():
    """It is the end-of-season sentinel, not a fault. scrape.py stops its week
    loop on it, so retrying would slow every season's normal finish."""
    s = session(Resp(404))
    with pytest.raises(requests.HTTPError):
        scrape.fetch(s, "/d", use_cache=False)
    assert len(s.calls) == 1


def test_a_persistent_fault_still_gives_up():
    """Retrying is not the same as never failing -- a server that is genuinely
    down must still stop the run rather than hammer it."""
    s = session(*[Resp(500)] * scrape.FETCH_ATTEMPTS)
    with pytest.raises(requests.HTTPError):
        scrape.fetch(s, "/e", use_cache=False)
    assert len(s.calls) == scrape.FETCH_ATTEMPTS


def test_a_successful_fetch_is_cached(tmp_path):
    s = session(Resp(200, "body"))
    scrape.fetch(s, "/week1", use_cache=False)
    assert any("week1" in f for f in os.listdir(str(tmp_path)))
