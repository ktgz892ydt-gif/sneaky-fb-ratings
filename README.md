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

## The honest part

Paired-comparison ratings only carry meaning between teams joined by a chain of
games. The site computes the game graph's connected components every run and
labels itself accordingly — in Week 1 it says, in large type, **"These are not
rankings yet."** Expect the statewide board to become genuinely informative
around Week 4–5, once league play has stitched the regions together.

## Same-name schools

Ohio fields three schools named Northwest, three named Perry, three named
Crestview, two named Jackson, plus a long tail of North / South / East /
Eastern / Southern. Keying on the bare name would silently merge distinct
programs and corrupt every rating that touches them.

The source solves most of this for us, and the scraper preserves what it
gives:

1. **City.** Both the scoreboard and the ranking pages write every team as
   `School (City)`, so `Jackson (Massillon)` and `Jackson (Jackson)` are
   simply different strings. This is the primary key and it resolves the
   overwhelming majority of collisions outright.
2. **State.** The scoreboard carries national scores, and an out-of-state team
   is tagged: `Salem (Salem) [NJ]`. Ohio has schools whose `School (City)`
   string is *also* a school somewhere else — six in the 2026 schedule, and
   twenty-six across the 2025 scores, some of them out-of-state on both sides
   (Bloomfield CT/NM, Greenwich CT/NY, Roswell GA/NM). The tag is part of the
   identity, because the per-week rule below cannot see this case at all: two
   schools that never play in the same week look exactly like one school
   playing a season.
3. **School ID.** The ranking pages carry a stable numeric ID per school.
   It survives division and region changes, so it is what matches a team to
   its own record in a previous season.
4. **Opponent geography.** Only for names that are still shared after the
   above — two Ohio schools with the same name *and* city — the resolver
   scores candidate assignments by how plausible each one's opponents are,
   using a region-versus-region scheduling distribution learned from the games
   that already resolved unambiguously.

**Anything still unresolved is kept as separate entities and tagged `?` —
never merged.**

Duplication is counted **per week**, not per season. A team plays at most one
game a week, so two appearances of "Perry" in the same week are two schools,
while ten appearances across ten weeks are one school playing a season.

Two checks enforce this, and they cover different things. An integrity check
**fails the build** if any team ends up in two *results* in one week. A team
holding two *fixtures* in one week is a warning rather than a failure — it
distorts a projection but leaves the ratings beneath it untouched, and
blocking the week's publish over it would trade a working board for none. What
does fail the build is a team whose season adds up to more games than a team
can play; that is the check that catches fixtures from two schools landing on
one team.

## The remaining schedule

Fixtures are predicted, never rated — a game with no result must not reach the
fit. Two filters apply before prediction: a fixture between two schools that
are both from outside Ohio is dropped (it is a prediction about teams this
board does not cover, made from two stand-in ratings), and an opponent that
still has no rating is stood in for at a division baseline and flagged
`Estimated` rather than quietly filled in.

A team whose played-plus-scheduled games exceed sixteen fails the build. That
is the check that catches fixtures from two different schools landing on one
team, which is invisible to every other assertion — the projected record adds
up perfectly well, just over a season that cannot happen.

## Reading the source pages

Every team is written `School (City)`, optionally tagged with a state or
country for non-Ohio opponents. Three details cost real games before they were
handled, and all three are pinned by tests built from verbatim page text:

- **A school name may contain its own parentheses** — `St Xavier (Louisville)
  (Louisville) [KY]`, `Landmark Eagles (club) (Cincinnati)`. The **last**
  parenthetical is the mailing city; anything before it belongs to the school.
- **Not every record is a game.** `TBD () [TBD]` is an opponent the site has
  not settled; `... at Foxfire (Zanesville) cancel` was called off. Both are
  recognised and discarded by name, and counted separately in the scrape log.
- **`UNRECOGNISED` should be zero.** It is the alarm for the page format
  moving, and it is worthless if it never reaches zero — so anything the source
  legitimately publishes must parse, including the Department of Defense
  schools abroad that tag a country rather than a state.

`check.py` fails the build if any OHSAA team ends the run with no game at all,
played or scheduled. That is what a silent parser regression looks like from
the outside, and nothing else in the pipeline notices it.

## Teams with no result yet

A team that has not played this season still receives a rating — the prior and
the regularizer see to that — but it is **not ranked**. Ranking a team on a
preseason estimate alone buries teams that earned their place on the field.
Unplayed teams stay in the table, carry their prior-based rating, and are
tagged `NO RESULT`.

## Layout

```
scripts/scrape.py    fetch scoreboards + the 28 regional roster pages
scripts/resolve.py   team identity resolution (the hard part)
scripts/ratings.py   the three models
scripts/build.py     orchestrate, emit ratings.json + both page variants
scripts/check.py     verification; the workflow fails if this fails
scripts/tune.py      fits the model constants against past seasons
scripts/season_prior.py  turns a finished season into next season's prior
tests/               unit tests for the fragile parts (pytest)
data/                committed raw scores, schedule and roster — replayable
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

There are two layers, and the difference matters.

**Code defaults** live in `RatingConfig` in `scripts/ratings.py`. They are used
only when no tuned file is present:

| | Default | Tuned? |
|---|---|---|
| `squash_scale` | 9.0 | yes |
| `prior_games` | 1.5 | yes |
| `carry` | 0.5 | yes |
| `division_weight` | 1.0 | yes |
| `margin_cap` | 49.0 | **no** — a fixed judgement value, protecting against typos and running-clock oddities rather than fitted to anything |

**Published constants** come from `data/tuned.json` when it exists, and the
board prefers them. As currently committed, fitted on 2024 and 2025:

| | Published |
|---|---|
| `squash_scale` | 9.0 |
| `prior_games` | 0.5 |
| `carry` | 0.5 |
| `division_weight` | 1.0 |
| `margin_cap` | 49.0 (untuned) |

Measured performance on 4,345 held-out games: **76.6% of games called
correctly**, log loss 0.478, mean margin error 17.4 points. The site footer
reports these, and says plainly when defaults are in use instead.

### How those were chosen

`scripts/tune.py` walks forward through past seasons — fit on weeks before a
holdout week, predict that week, score it — and grid-searches the four tunable
constants. Selection is by log loss, not accuracy, so confident wrong calls are
punished.

It then applies the **one-standard-error rule**: among every configuration
statistically indistinguishable from the best, it takes the most conservative
rather than the best-scoring.

The standard error is of the *difference*, not of the score. Every
configuration is scored on the same games in the same order, so comparing two
of them is a paired comparison — and most of the variance in per-game log loss
is the game itself, since a coin-flip upset scores badly under every
configuration alike. The committed fit used the marginal standard error of the
winning score, which treats those as independent samples and admitted 294 of
1,296 configurations as "tied". That was read as evidence the evaluation set
was too small; it was the wrong standard error. Pairing removes the shared game
noise and leaves only the part that distinguishes one configuration from
another. **This takes effect on the next re-fit**, and is expected to change
which constants get selected.

Two consequences worth understanding:

- `data/tuned.json` reports both standard errors: `standardErrorPaired` is the
  one selection uses, `standardError` is the old marginal quantity, kept only
  so the two can be compared.
- `data/tuned.json` reports `selectedConfigAtGridEdge` separately from
  `outrightBestAtGridEdge`. An edge on the *outright best* means the grid
  constrained the search and should be widened. An edge on the *selected*
  config is expected — conservatism pushes toward a boundary by design.
- `carry` is capped at 1.0. Above that the model would amplify last season's
  estimate rather than regress it, asserting this year's team is *more* extreme
  than last year's measurement. Backtests reward it; football does not.

To re-fit: **Actions → Update ratings → Run workflow**, ticking *Re-fit the
model constants*. It is opt-in because it is slow and the answer moves little.

### Reproducing a build

`generatedAt` refreshes on every build, so `site/ratings.json` always shows a
diff. To verify a build reproduces exactly, pin it:

```bash
python scripts/build.py --generated-at 2026-08-25T00:00:00+00:00 --out /tmp/check.json --no-site
```

`--no-site` matters here: without it the run still rewrites `site/index.html`
and `dist/preview.html` as a side effect, so a command whose whole purpose is
to leave the tree alone does not.

### More seasons

Tuning currently uses two evaluation seasons (2024 and 2025, with 2023 as the
prior source). More would stabilise the constants. Add years to the backfill
loop in `.github/workflows/update.yml`; the source has seasons back to 2000 at
the same URL pattern. Note that 2020 is a COVID-shortened season and is worth
either excluding or reporting separately.

## Source and etiquette

Scores, OHSAA divisions/regions and Harbin points come from
[Joe Eitel's Ohio HS Football](https://joeeitel.com/hsfoot/), a one-person site
running since 2000. A full season refresh is ~44 requests. The scraper
identifies itself, rate-limits to one request per 1.5s, and caches. Please keep
all of that, and credit the source on any page you publish.

Harbin points are OHSAA's official playoff qualifier and are shown only for
comparison. They ignore margin entirely and award nothing for a loss, however
narrow — they answer "who earned a playoff spot," not "who is best."
