# Ohio HS Football Ratings

Bradley-Terry and Massey ratings for all ~708 OHSAA 11-man varsity football
programs, rebuilt weekly and published as a static page.

## Why not just Bradley-Terry?

After Week 1, plain Bradley-Terry is not weak — it is **undefined**. Every team
is 1-0 or 0-1, which is textbook perfect separation: the maximum likelihood
estimate diverges to ±∞ and never converges. Worse, the comparison graph is
~400 disconnected two-team components, so even a regularized fit only tells you
"winners are above average." Run `scripts/build.py` on Week 1 data and the
`BT (W/L)` column is literally a constant.

Massey degrades more gracefully — point margin is a continuous signal instead of
a single bit — but it is singular for the same reason and its Week 1 ordering is
just margin of victory against an unknown opponent.

So the headline number here is neither. It is a **margin-informed
Bradley-Terry**: each game contributes a *fractional* win taken from a squashed
point margin.

| Margin | Counts as |
|-------:|----------:|
| 3 pts  | 0.59 wins |
| 7 pts  | 0.68 wins |
| 14 pts | 0.83 wins |
| 21 pts | 0.91 wins |
| 35 pts | 0.98 wins |
| 56 pts | 0.998 wins |

One model, one likelihood, no arbitrary blend weights. Margin still matters, but
a 63-0 game is not worth nine times a 7-0 game. Ratings are in points: the gap
between two ratings is the expected neutral-field margin.

Plain Bradley-Terry and plain Massey are still fitted and shown alongside, under
"Compare models." Where they disagree is the interesting part.

## Same-name schools

Ohio fields three schools named Northwest, three named Perry, three named
Crestview, two named Jackson, plus a long tail of North / South / East /
Eastern / Southern. Keying on the bare name would silently merge distinct
programs and corrupt every rating that touches them.

The resolver matches names to schools in two passes:

1. **Record matching.** A team plays at most one game per week, so its running
   W-L is a fingerprint against the roster pages.
2. **Opponent geography.** Where records tie, it scores each candidate
   assignment by how plausible its opponents are, using a region-vs-region
   scheduling distribution learned from the games that already resolved
   unambiguously.

On Week 1 2026 this takes 20 ambiguous names down to 4. **Anything still
unresolved is kept as separate entities and tagged `?` — never merged.**

## Layout

```
scripts/scrape.py    fetch scoreboards + the 28 regional roster pages
scripts/resolve.py   team identity resolution (the hard part)
scripts/ratings.py   the three models
scripts/build.py     orchestrate, emit ratings.json + both page variants
scripts/check.py     verification; the workflow fails if this fails
data/                committed raw scores and roster — versioned, replayable
site/                what GitHub Pages serves
```

## Running it

```bash
pip install -r requirements.txt
python scripts/scrape.py --season 2026   # needs network
python scripts/build.py
python scripts/check.py
```

`build.py` falls back to the checked-in Week 1 fixture when no scraped data is
present, so the pipeline runs offline.

## Automation

`.github/workflows/update.yml` runs Saturday 08:00 and Sunday 13:00 ET. Every
run re-scrapes **all** weeks rather than appending — scores get corrected days
later, and an append-only store would carry the original typo forever. Raw data
is committed each run, so any past week's board can be replayed exactly.

## Configuration

All knobs live in `RatingConfig` in `scripts/ratings.py`:

- `squash_scale` (9.0) — how fast margin saturates. The most consequential knob.
- `margin_cap` (49.0) — hard clip before squashing.
- `prior_games` (1.5) — shrinkage strength in pseudo-games. Fixed, not
  scheduled: real games outgrow it on their own, so no per-week decay table.

To add prior seasons (recommended — it is the single biggest early-season
improvement), scrape another year and pass both game files; the source has
seasons back to 2000 at the same URL pattern.

## Source and etiquette

Scores, OHSAA divisions/regions and Harbin points come from
[Joe Eitel's Ohio HS Football](https://joeeitel.com/hsfoot/), a one-person site
running since 2000. A full season refresh is ~44 requests. The scraper
identifies itself, rate-limits to one request per 1.5s, and caches. Please keep
all of that, and credit the source on any page you publish.

Harbin points are OHSAA's official playoff qualifier and are shown only for
comparison. They ignore margin entirely and award nothing for a loss, however
narrow — they answer "who earned a playoff spot," not "who is best."
