# Alex's Awesome Aggregator — project handoff

Everything a fresh session needs to be useful on this project without
rediscovering what already cost real debugging time. Written 2026-08-25.

- **Repo:** https://github.com/ktgz892ydt-gif/sneaky-fb-ratings
- **Live site:** https://ktgz892ydt-gif.github.io/sneaky-fb-ratings
- **Site name:** Alex's Awesome Aggregator · **metric name:** Alex Points
- **Local clone:** `~/Documents/GitHub/sneaky-fb-ratings` (macOS)
- **Data source:** [Joe Eitel's Ohio HS Football](https://joeeitel.com/hsfoot/)

Alex is not a software developer. Lead with the action, name the exact button,
and explain *why* only when it changes a decision.

---

## What it is

A weekly-rebuilt statewide rating for all ~700 OHSAA 11-man varsity football
programs. Scrapes scores, fits a rating model, predicts the remaining
schedule, publishes a static page. Runs itself on GitHub Actions; costs
nothing.

**Current state: working and deployed.** Week 1 of 2026 is live. Three past
seasons (2023–2025) are committed. Model constants are fitted, not guessed.
256 unit tests pass in CI before anything touches the network.

---

## First five minutes in a new session

```bash
cd ~/Documents/GitHub/sneaky-fb-ratings
git fetch && git status          # the bot commits here; the remote is often ahead
python -m pytest tests/ -q       # expect 256 passed; pip install -r requirements.txt if not
python scripts/build.py --generated-at 2026-08-25T00:00:00+00:00 --out /tmp/check.json --no-site --no-history
```

That last command is the deterministic build check — pinning `--generated-at`
is what makes two builds comparable, because otherwise the timestamp changes
and every diff is non-empty.

If `gh` is available, `gh run list` and `gh run view --log` read the workflow
logs directly. That is worth setting up: reading logs used to mean asking Alex
to download a zip and upload it back, which cost a full round trip per
debugging cycle.

---

## The model

The headline number is a **margin-informed Bradley-Terry** rating. Each game
contributes a *fractional* win derived from a logistic squash of the point
margin, so a 3-point win is worth ~0.59 wins and a 45-point win ~0.97.

Why not the obvious alternatives:

- **Plain Bradley-Terry** is *undefined* in Week 1 — every team is 1-0 or 0-1,
  which is perfect separation, and the MLE diverges. Verified empirically: the
  `BT (W/L)` column is literally a constant (+5.84 for every 1-0 team).
- **Massey** (least squares on margin) degrades more gracefully but rewards
  running up the score against weak opponents.

Both are still fitted and shown under "Compare models." Where they disagree is
informative.

Ratings are on a **points scale**, but a rating difference is NOT an expected
margin and it was wrong to say so. Measured over 13,756 walk-forward
predictions, regressing actual margin on predicted gives a slope of **1.49**: a
predicted 14-point win is really a 21-point win. The squash discounts blowouts
when fitting, and that discount survives into the rescaled output.

So the board publishes a **calibrated** margin: the rating difference times
`margin_scale`, fitted at 1.4708. That cuts mean margin error from 18.0 to 16.6
points and removes the bias (+0.60 to -0.14). The slope is flat across the
season (1.53, 1.42, 1.50, 1.53, 1.46 through weeks 1-9), so it is one constant
and not a curve.

**The calibration is display-only and must never reach the probability.**
`prob_scale` was fitted against the RAW difference and is well calibrated there.
Both quantities are called "the margin", so feeding the calibrated one into the
probability is an easy and invisible mistake -- `check.py` reproduces every
published probability from `predictedHomeMargin / marginScale` and fails if it
cannot. Injecting the mistake trips it on 3,030 fixtures.

### Measured performance

Fitted on 2024 and 2025, evaluated walk-forward on 5,065 held-out games:

| | |
|---|---|
| Accuracy | **75.8%** |
| Log loss | 0.4846 |
| Mean margin error | 18.2 pts |

Per holdout week — the model earns its keep as the schedule graph connects:

| Week | Log loss | Accuracy |
|---|---|---|
| 1 | 0.579 | 70.3% |
| 2 | 0.567 | 71.8% |
| 4 | 0.513 | 72.9% |
| 7 | 0.414 | 80.2% |
| 10 | 0.373 | 84.4% |

**These are slightly worse than the previous 76.6% / 0.4776 / 17.4, and that is
not a regression.** Those were measured on 4,345 games. The parser fixes
recovered about a fifth more history, and the recovered games — out-of-state
opponents written without a mailing city — are precisely the ones a statewide
Ohio rating finds hardest to call. The test set is harder and more honest. The
two sets of numbers are not comparable and the drop is not a trend.

### Published constants

In `data/tuned.json`, read automatically by `build.py`:

```
squash_scale     8.0
prior_games      0.5
carry            0.6
division_weight  1.0
margin_cap      49.0   (NOT tuned — a fixed guard against typos/running clock)
probScale a/b   19.68 / 82.72
```

Selected by the **one-standard-error rule**, now applied to *paired* per-game
differences. The outright best was scale=9.0 prior=0.25 carry=0.6 div=1.0
(logloss 0.4839); 20 configurations tied with it and the most conservative was
taken. Neither the outright best nor the selected config sits on a grid edge.

**The old "294 configurations tied" figure was an artefact, and it is gone.**
The SE used was the marginal one — the spread of per-game log loss for the
winning config — but every config is scored on the same games in the same
order, so the comparison is *paired*. Most of that spread is the game, not the
config: a coin-flip upset scores badly under all of them alike. Differencing
first removes the shared noise. Measured on the current fit:

| | standard error | configurations "tied" |
|---|---|---|
| marginal (old) | 0.0068 | 294 of 1,296 |
| **paired (now)** | **0.0010** | **20 of 1,296** |

`tied_with_best()` in `tune.py` is the function; `tests/test_tune.py` pins the
behaviour, including the case that separates the two rules — a config worse on
*every single game* by 0.004 nats sits inside the marginal SE and outside the
paired one.

`carry` is capped at 1.0 by design — above that the model would *amplify* last
season's estimate rather than regress it.

**Probabilities are uncertainty-aware.** `prob_scale(g) = sqrt(a + b/g)`, where
`g` is the games played by the *less* established of the two teams. Fewer games
means a flatter curve. This is why an early-season prediction is not stated
with the same confidence as a Week 9 one — the model handles it, not a
disclaimer.

**A stand-in opponent gets its own scale, not week 1's.** Two different things
arrive at `g = 0`: a real team in week 1, and an opponent with no rating at all
standing in at a division baseline. They used to share the flat `squash_scale`
of 9.0 — which is *steeper* than the fitted curve at one game (10.8), so a
prediction against a completely unrated opponent came out more confident than
one against a barely-rated one. It applied to 23% of the week 1 fixture list.
A stand-in now uses `prob_scale_max` (20.0), the flattest the curve may reach.
Week 1 keeps the flat scale, which is measured; the stand-in bound is a stated
floor on confidence, not a fit, because `tune.py` only fits on `g >= 1`.

---

## The season simulator

Phase 1 is built: click a school, see its whole season in week order — results,
then remaining fixtures with a predicted margin, win probability, and a
projected final record.

**The data flow.** `scrape.py` writes two files per season: `games_{yr}.csv`
(played, with scores) and `schedule_{yr}.csv` (fixtures, no scores).
`build.py` fits ratings from the first and predicts the second.

**The separation is the whole design.** A fixture must never reach `resolve()`
or `rate()`. It has no result, so it would enter as a phantom 0-0 tie between
two real teams — and the ratings table would still look completely normal.
`tests/test_predict.py` builds the same season twice, with and without a
schedule loaded, and demands the ratings come out bit-for-bit identical.

Fixtures are matched to rated teams **by name**, against the table `resolve()`
already built from completed games. They are not passed through `resolve()`
itself, which derives identity from results a fixture does not have.

**Transport.** `site/ratings.json` carries `results` and `schedule` as compact
rows pointing at teams by *index*. Written out in full this was 1.7 MB; it is
now ~82 KB gzipped. `scheduleCols` in the payload documents the short keys.
The two lists grow past each other as the season runs, so the page stays about
the same size all year.

**Phases 2 through 6 are built** — Monte Carlo, regional playoff odds and a real
Harbin implementation, delivered together because they are one thing. See "The
playoff model" and "The track record" below.

Monte Carlo runs at build time in numpy, not in the browser: 10,000 seasons
takes 0.9s vectorised and would be painful on a phone.

---

## The playoff model

**The output is a % chance of reaching the playoffs, per team.** It is built on
a division of labour that is the entire design:

> **The rule is Harbin, unaltered. The forecast is Alex Points.**

Harbin decides who qualifies, so a defensible answer has to use it. But Harbin
*cannot forecast* — it is a backward-looking reward with no opinion about who
wins on Friday. So the odds are not computable from Harbin at all. Something
predictive is required, and that is the board's own rating. This is why the
feature is unique rather than a re-skin of OHSAA's standings: no one without a
predictive rating can produce it.

```
repeat 10,000 times:
    decide each remaining regular-season game by the board's win probability
    compute REAL Harbin over the finished season
    top 12 of each region qualify
playoff% = share of simulations in which the team qualifies
```

Simulation is required, not decorative. Harbin's Level 2 pays you for *your
opponents'* wins, so one Friday result moves the qualifier for dozens of teams,
and the teams competing for a regional place are exactly the ones entangled
that way. Multiplying independent probabilities would destroy that.

### Harbin was recovered from data, not from the rulebook

`scripts/harbin.py` was not written from a description of the rules. The
formula was recovered by least squares against the source's own published
Harbin column over 702 teams of a completed season. The fitted per-division win
values came out `6.54 6.10 5.47 5.02 4.48 3.94 3.40` — a monotone ladder in
steps of ~0.5 — and rounding to the clean values reproduces the published
figure **exactly** for 86% of teams with no out-of-state opponent in their
two-level tree, median error 0.0000.

The test that matters is not point agreement but whether it names the real
playoff field. Scored against the teams that actually played in week 11:

| season | qualifiers/region | we identify | ceiling (site's own Harbin) |
|---|---|---|---|
| 2023 | 16 | 99.3% | 99.8% |
| 2024 | 16 | 99.1% | 99.3% |
| 2025 | 12 | 99.4% | 100% |

**The qualifier count changed.** 16 per region through 2024; 12 from 2025, with
the top 4 seeded into a bye. That was read off the brackets — counting distinct
Ohio teams in weeks 11+ gives exactly 12 in all 28 regions for 2025 — not
assumed. It lives in `harbin.QUALIFIERS_PER_REGION`.

### Is it calibrated?

Backtested on 2025 from the end of week 6, with four weeks unplayed:

| predicted band | teams | actually qualified |
|---|---|---|
| 0–10% | 265 | 4.9% |
| 20–30% | 29 | 24.1% |
| 50–60% | 20 | 55.0% |
| 90–100% | 248 | 96.8% |

Brier score **0.0795** against 0.2495 for always predicting the base rate.

### Known approximation

An out-of-state opponent carries no OHSAA division; OHSAA assigns one by
enrollment, which the scoreboard does not publish. A Division II stand-in is
used (least squares put the effective value at 5.98). Measured on 2025 it
*overstates* by ~0.3 for the teams most exposed. Those teams are flagged
`harbinApprox` and the page says so rather than presenting the figure as exact.

### The simulation stops when the regular season does

From about November 1 there is nothing left to simulate: every regular-season
game has been played, `sim_rem` is empty, and no team carries playoff odds.
That is correct, and it is the state of every build through the state finals.

`check.py` used to demand 28 regions carrying odds unconditionally, so **the
first Saturday after week 10 would have gone red and the board would have
stopped updating for the entire playoffs** — with the failure issue pointing at
a check rather than a fault. Reproduced before fixing.

The fix is not to drop the check. "The season is finished" and "the remaining
schedule was lost" produce the *same* empty payload and mean opposite things,
and that check was doing double duty as the schedule-loss detector. So
`build.py` now publishes `playoffs.simulated` and
`playoffs.remainingRegularFixtures`, and a skip is accepted only against
independent evidence — the results themselves having reached
`lastRegularWeek`. A mid-season skip still fails, loudly:

```
FAIL  the playoff simulation was skipped, but results only reach week 2 of 10
      -- the season is not over, so the remaining schedule has been lost
```

Both directions are covered by reproduction, not by argument.

### The invariant that guards it

`check.py` asserts that playoff odds **sum to exactly 12 across each region**,
bye odds to 4, and top-seed odds to 1. Exactly that many teams qualify in every
simulated season, so those sums are conservation laws. They are the strongest
available check on a Monte Carlo — a plausible-looking set of percentages that
does not add up means the ranking inside the simulation is wrong, and nothing
else would reveal it.

### What-ifs (Phase 5) are conditionals, not re-runs

"Beat La Salle and you are at 94%; lose and you are at 68%."

The obvious implementation is to pin a fixture's probability to 1 and simulate
again. **Don't.** That is two fresh runs per team per remaining fixture — about
12,600 simulations, over three hours at 0.9s each — to answer a question the
sample already contains. The seasons in which we beat La Salle *are* a fair
sample of the seasons in which we beat La Salle, so conditioning on the finished
run is exact, not an approximation, and it costs a boolean mask. Both what-if
tables build in about 0.1s.

Two are published per team:

- **`whatIf`** — every remaining game, with playoff odds conditional on winning
  and on losing. The gap between them is what makes one Friday matter more than
  another, and it is often not the game you would guess: Elder's out-of-state
  fixtures swing 2-7 points while its Ohio games swing 22-26, because Harbin
  only pays for beating Ohio teams who then win.
- **`watch`** — the games this team is *not* playing in that move its odds most,
  and which side to root for. This exists because Harbin's Level 2 pays you for
  your opponents' wins while the twelve regional places are contested, so the
  two pulls run in opposite directions and the net is not something anyone can
  work out unaided.

**The invariant that guards them.** These are conditionals, so the law of total
probability applies: a team's own playoff odds must lie *between* its
odds-if-win and odds-if-lose. `check.py` asserts it on every pair. It catches
the one error that is otherwise invisible — `won` records whether the *home*
side won, so a road game read the wrong way round reports "if we win" numbers
that are really "if we lose", and nothing about the resulting page looks wrong.
6,126 of 6,126 pairs currently satisfy it.

**A branch nobody simulated is not evidence.** Below `MIN_BRANCH` (300 seasons)
on the thinner side, nothing is published rather than quoting a figure built on
a handful of samples.

**Precision is capped at three decimals** (`simulate.DP`). Not a size trick: at
10,000 seasons the standard error near a half is 0.005, so a fourth decimal
claims fifty times the precision the method has.

### One trap already hit

Ranking inside the simulation originally broke ties by array position, which is
alphabetical by team id. On four identical teams that skewed playoff odds from
50% into a 40%–61% spread — an alphabetically early team won every tie in every
simulation. Ties are now broken by a coin flip (see `TIEBREAK_JITTER`), which
is what OHSAA does. `tests/test_harbin.py` pins it.

## The track record

**`data/history.jsonl` is the only file in this repo that cannot be
regenerated.** Everything else in `data/` is derived from the source and can be
rebuilt by re-scraping. A prediction is what the board said *at a moment in
time*, before the game was played; recomputing it later from a model that has
since seen the result is not a prediction, it is a retrofit, and it would
flatter the record without anyone noticing.

**If this file is lost, the track record is genuinely gone.** The workflow
commits it via `git add data/`; there is a note there so nobody narrows that
path without realising what it drops.

### Live and backtest are never pooled

**Including the calibration table, which used to be the one exception.**
`history.score()` split its headline figures by kind but accumulated one shared
`bins` dict, so the calibration display pooled replayed weeks with live ones —
the exact laundering the rest of this section exists to prevent. It was
immaterial at three live games and would have stopped being immaterial without
anyone noticing. Bins are now keyed `(kind, bin)`, `calibration` is nested by
kind, and `check.py` fails a flat one. The page picks whichever bucket actually
has bins (an empty bucket is still a truthy object in JS).

**`head_to_head` reads live captures only.** It built its lookup from every
snapshot regardless of kind. Harmless today — rival picks exist only for 2026,
which has no backtests — but a future backfill would have quietly scored a
replay against another forecaster's archived real-time picks and called it a
head-to-head. Verified byte-identical before and after the filter.


Two kinds of line, and the distinction is the whole point:

| | what it is | strength |
|---|---|---|
| `live` | captured by the build running that week, before kickoff | out of sample; the model could not have seen the result |
| `backtest` | replayed afterwards from committed scores, fit on weeks 1..N to predict N+1 | honest walk-forward, but the model's *constants* were tuned on those seasons |

`scripts/backfill_history.py` produced 27 backtest weeks across 2023–2025 so the
page is not blank for the first ten weeks of a season. They are labelled, shown
separately, and `check.py` fails the build if the two are ever merged into one
figure. Pooling would launder the weaker number into the stronger one and the
headline would look entirely reasonable.

### What it currently says

13,751 backtested games: **77.3% called correctly**, log loss 0.4551, Brier
0.1509, mean margin error 16.7 points.

**The backtest must carry the preseason prior.** It originally did not:
`backfill_history.py` called `rate(ids, train, cfg)` while the live build calls
`rate(..., priors=priors)` and `tune.py` fits with a prior. A replay without one
is not measuring the model that ships, and the gap is widest in exactly the
early weeks a backtest covers. Fixing it moved accuracy 75.9% -> 77.3% and log
loss 0.4758 -> 0.4551 -- the bug was understating the board, not flattering it. Calibration, which is the part that
matters:

| board said | favourite actually won | n |
|---|---|---|
| 60% | 59.7% | 2,397 |
| 70% | 67.7% | 2,420 |
| 80% | 78.7% | 2,582 |
| 90% | 89.3% | 2,704 |
| 95%+ | 98.5% | 2,189 |

### Rules that keep it honest

- **A capture is revisable until its games start, then frozen forever.** This
  is the rule, and it is *not* "write once" — that was tried and it was wrong.
  `through_week` is the highest week holding **any** result, so a single
  Thursday-night game turns the week over while ~350 Friday fixtures are still
  to come. Under write-once, whichever build ran first owned the week's slot
  permanently. It bit on 2026 week 2: a manual Friday-morning run recorded
  week 3's forecast off a model that had seen **27 of week 2's 357 games**, and
  Saturday's much better forecast was refused in silence.

  The damage was not a flattered record — it was the opposite — but an
  *incoherent* one: the standard varied with when somebody happened to click
  "Run workflow", and weeks captured at different points cannot be averaged
  together. `history.record()` now replaces a capture as long as **every game
  it forecasts is still unplayed** (revising a call on a game nobody has played
  is not hindsight; there is no result to have seen) and freezes it the moment
  one kicks off. `build.py` passes the same results map it scores against.
- **The freeze is whole-line, not per-game.** One capture is one instant. If
  even one forecast game has been played the whole line stands. In practice
  the freeze is belt-and-braces: the horizon is `through_week + 1`, so a game
  in it being played is the *same event* as `through_week` advancing, which
  opens a new slot anyway. The load-bearing half is the revision — same week,
  better data, better forecast.
- **A caller without results revises nothing.** `append_if_new` is the strict
  wrapper, still used by `backfill_history.py` — a replay is deterministic, so
  a second one has nothing new to say.
- **A replay cannot overwrite a live capture.** Backtests never revise at all,
  and a test pins it both ways.
- **A revision rewrites the file, so it is written atomically** — beside the
  log, then `os.replace`d into place. An interrupted write leaves the previous
  log intact rather than a truncated one. This is the file that cannot be
  regenerated; it does not get a partial write.
- **Only one week past `through_week` is recorded.** "We called this Friday
  right" is the claim worth being held to, and it keeps the log small enough to
  live in the repo for years. The week currently in progress was forecast by
  the *previous* capture, so nothing is missed by the horizon starting above
  `through_week`.
- **An unplayed prediction is never counted as a miss.** Otherwise the record
  would look worse every time the source posts late.
- **A truncated final line is skipped, not fatal** — that is what an interrupted
  commit looks like, and losing the history to one bad append would be a poor
  trade.
- **Backtest lines carry no per-team block.** They are replayable by definition,
  so storing their ratings duplicated something a two-second script regenerates.
  It cost 1.2 MB of the log's first 1.7.

### `build.py` now writes to data/

This is new: the build has a side effect. A rebuild of the same week rewrites
that week's line rather than adding one, and with `--generated-at` pinned the
rewrite is byte-identical, so the deterministic-build check still passes. But
**the reproducibility recipe should use `--no-history`** — a check should not
write to the track record at all, even harmlessly.

### Comparing against another model — built

The comparison model is **Drew Pasteur's Ohio Fantastic 50**
(fantastic50.net), which publishes a favourite, a predicted margin and a win
probability per game: the same three quantities this board produces.

**Permission.** The site states: *"Media outlets (print, broadcast, or online)
are welcome to use any content from this site, provided that they credit the
source."* That is explicit permission conditional on attribution, so the credit
is a licence term, not politeness — `check.py` fails the build if the payload
carries his figures without `sourceName` and `sourceUrl`. His robots.txt asks
for a 10-second crawl delay (six times joeeitel's); we make one request a week
and honour it anyway.

**Only the intersection is scored, and the test is paired.** Both sites publish
their own accuracy and those numbers are NOT comparable: he predicted 345 games
in week 1 of 2026 where this board's scrape found 400 completed. Comparing
headline figures would measure the schedules, not the models. So only games
both predicted are scored, and the difference is tested on the games they
*disagreed* about — the only ones carrying information about which is better.

**The test is an exact binomial (McNemar), not the normal approximation.** With
a single disagreement the gap and its standard error are always exactly equal
(1/n against sqrt(1)/n), so one lucky call would grade as evidence. It is not:
the exact p-value there is 1.0. This bit me while writing the tests.

**Matching his names.** He credits the same source for scores, so his school
names are Joe Eitel's — but bare ("Deer Park"), sometimes city-prefixed
("Dayton Stivers") and sometimes abbreviated ("Cuyahoga Val. Christian").
Matching on names alone reaches 95% and leaves Ohio's duplicates (Perry,
Jackson, Madison) genuinely ambiguous. So the match is on the **game**, not the
team: a pair of names against the fixture list this board already holds for
that week. That took coverage to **98.9% with zero ambiguous**. A pick matching
more than one fixture is dropped, never guessed.

**His convention is the opposite of the scoreboard's.** On his page the first
team named is the FAVOURITE and "at" means the favourite is away; on the
scoreboard the first team is always the visitor. Reading one as the other
inverts the home side of every pick and still looks plausible. There is a test.

**Fails soft, always.** `capture_rivals.py` is the only step depending on a
site we do not control. An unreachable host, a moved page format or a week
already recorded all exit 0 and skip. A board that refuses to publish because
someone else's server is down would be a bad trade.

**What is still not possible:** a retrospective comparison. That needs his
week-by-week calls archived at the time, which are not published. The record
starts from the first week both were captured and accumulates.

## Decisions already litigated (don't relitigate without new evidence)

**Week 1 rankings are not meaningful, and the site says so.** After one week the
game graph is ~382 disconnected components, largest containing 2 teams. The
reliability panel reads "These are not rankings yet" and recalculates weekly.
The team panel carries the same warning over its projected records. Deliberate.

**Division weighting is measured, not assumed.** Alex asked several times for
lower divisions to be weighted down. The implementation shrinks each team toward
*its own division's measured baseline*, computed from real cross-division games
— never from enrollment. Measured 2025 ladder:

```
Div I  +5.87   Div II +2.60   Div III +1.86   Div IV +1.00
Div V  -0.83   Div VI -2.24   Div VII -8.27
```

A strong small school (Kirtland, Marion Local) still floats above its division
baseline because it earned that. Enrollment-based discounting (what Harbin does)
was rejected because it caps such programs permanently.

**An unrateable opponent stands in at the Division III baseline.** An
out-of-state school that has not played anybody has no rating and no evidence
from which to invent one. Division III is the middle of the seven-division
ladder, so it is the least-committal guess. Alex chose this explicitly, on the
condition that it be visible: every prediction leaning on one is flagged
`Estimated` and the basis is published in `fallbackRating`.

**An out-of-state team's state tag is part of its identity.** `Salem (Salem)`
is a school in Ohio *and* a school in New Jersey, and they write identically.
Six such collisions were in the 2026 schedule and twenty-six in the 2025
scores, some out-of-state on both sides (Bloomfield CT/NM, Greenwich CT/NY).

This was a live bug: `load_schedule()` and `load_games()` read the name column
and ignored the `away_state`/`home_state` columns sitting beside it, so New
Jersey's fixtures landed on Ohio's Salem and the site published a nineteen-game
season projected 13.3–5.7. The per-week duplicate guard cannot see this class
at all — two schools that never play in the same week look exactly like one
school playing a season. `resolve.team_identity()` is now the one place the key is
built, and both loaders go through it. Ohio teams carry no tag, so no Ohio
identity moved.

**A fixture between two non-Ohio schools is dropped before prediction.** The
source is an Ohio scoreboard but carries border-state games that are entirely
someone else's — Kentucky at Kentucky, Michigan at Michigan. A *completed* one
is worth keeping: it rates an out-of-state team an Ohio school will later play.
An unplayed one is a prediction about two teams nobody here follows, built from
two stand-in ratings. `build.py` filters them after resolution and reports the
count as `scheduleForeignDropped`. This is why the fixture list and the
"estimated" count both fell sharply.

**A name shared by an Ohio school and an out-of-state namesake resolves to the
Ohio one** — but only when exactly one candidate is in Ohio. This is now a
fallback for the case where the source omits the tag; the tag handles it first.
Two Ohio schools sharing a name are still refused rather than guessed.

**Harbin is shown for comparison only.** It's OHSAA's playoff qualifier: ignores
margin, gives nothing for a loss, and scales by opponent *division*. It answers
"who earned a playoff spot," not "who is best."

**Teams with no result are rated but not ranked**, and tagged `NO RESULT`.

**Team identity is `School (City)`**, from a stable numeric `school_id` on the
ranking pages. This resolves nearly all same-name collisions outright (Ohio has
three Perrys, three Northwests, three Crestviews). Anything unresolved is kept
separate and tagged — **never merged**.

**Schedule problems warn; they do not fail the build.** A double-booked fixture
distorts a projection, but failing `check.py` would stop the week's ratings from
publishing over the newest and least load-bearing part of the page. Structural
problems that can only come from a code bug still fail.

**Every scored game carries its kickoff date, and that is what makes a missed
game visible.** `games_{yr}.csv` gained a `date` column beside `week`.
`SB_GAME_RE` had always captured the date — it anchors the whole pattern — it
was simply thrown away.

The reason it matters: in a fixture list, *"not played yet"* and *"played, and
we failed to read it"* are the same row. A week number cannot tell them apart,
which is how two schools once sat at zero games for a season with every check
passing. With the date, a fixture whose kickoff has passed is by definition a
score the board should hold and does not. `check.py` names them:

```
WARN  1 fixture(s) are past their kickoff date with no score -- either the
      source has not posted them or the parser missed them:
      ['w2 2026-08-27 Dunbar (Dayton) at Stivers (Dayton)']
```

Three rules, deliberately lopsided:

- **A few overdue games warn.** A postponement and a score the source posts
  late look identical, and neither should stop a good week publishing. The
  measured baseline mid-week 2 of 2026 was 1. The games are listed, not just
  counted, so they can be looked up on the source.
- **Half a week overdue fails.** Nobody postpones 180 games; that shape means
  the scoreboard did not parse. The rule is `missing < 20 or missing <= half`,
  so small playoff weeks (5–59 fixtures) cannot trip it and permanently
  postponed games cannot accumulate into a failure — each stays inside its own
  week.
- **Week numbering must not overlap** — if week N holds games dated into week
  N+1, something is filed wrong. Warns, because a postponement replayed later
  does this innocently. Week numbering is the spine of the model: the prior,
  the walk-forward tuning and the track record are all keyed on it.

`today` is the build's own `generatedAt`, never the wall clock, so a pinned
rebuild reproduces the verdict.

**`PARSER_VERSION` was deliberately NOT bumped.** It gates the re-fit on
whether a season's games would parse differently now; adding a column does not
change the set of games, and bumping would have blocked every re-fit until
2023–2025 were re-scraped for nothing. Those three seasons therefore carry no
dates and never will unless re-scraped — `load_games` treats the column as
optional and the checks skip undated records. 2026 picks up dates on its next
scrape, since the workflow re-reads the whole season every run.

**One schedule problem does fail the build: an impossible season length.**
`check.py` refuses a team whose played + scheduled games exceed 16 (ten regular
plus five playoff rounds). This is the assertion that would have caught the
Salem merge above. Every other check passed on it — the projected record was
internally consistent arithmetic, just over a season that cannot happen.

---

## Hard-won gotchas

**0f. Nothing tested `build.main()`, and a certain crash shipped green.**
`record` was imported into `build.py` as `record_history` — which is also the
name of one of `main()`'s own parameters, so the call resolved to the boolean
`True` and every real build died with `'bool' object is not callable`. **232
unit tests passed over it**, because not one of them called `main()`: the
tests import helper functions and build seasons in memory. `py_compile` does
not catch shadowing either, and the workflow does not run on push, so it
reached CI unseen.

`tests/test_history.py` now drives `main()` end to end on a four-school season
written to real CSVs in `tmp_path`. Reintroducing the shadowed name turns three
of them red at the exact line. **Any new orchestration in `build.main()` needs
a test that calls `main()`** — a helper-level test cannot see this class at all.

To run the suite here, there is no numpy/scipy/pytest on this Mac. Make a venv
rather than installing into the system Python:

```bash
python3 -m venv /tmp/fbenv && /tmp/fbenv/bin/pip install -r requirements.txt
/tmp/fbenv/bin/python -m pytest tests/ -q
```

It resolves numpy 2.0.2 / scipy 1.13.1 on Python 3.9 — the gap against CI's
3.12 that gotcha 0e is about.


**0e. CI runs Python 3.12; a dev box here runs 3.9.** `data/requirements.lock`
records what CI resolved -- currently numpy 2.5.2 / scipy 1.18.1 against 2.0.2
/ 1.13.1 locally. That gap is real and has already bitten once: a test built
two identical vectors and then read their variance, which came out 1.2e-32 on
3.9 and exactly 0.0 on 3.12, flipping a verdict and failing the build. **Never
let a test's outcome turn on the last bits of a float.** Construct inputs that
genuinely vary, and assert the standard error is a real quantity before
asserting anything about the verdict derived from it.


**0d. A guard that cannot fire is worse than no guard.** The re-fit step used
to refuse to run when `scripts/scrape.py` was newer than a season's
`games_{yr}.csv`. It never fired once: git does not preserve mtimes, so on a
fresh CI checkout every file carries the same timestamp and `-nt` is always
false. It read as protection and provided none. The check now compares
`scrape.PARSER_VERSION` against `data/parser_versions.json`, which the scraper
writes beside each season it produces. **Bump `PARSER_VERSION` whenever a
parsing change could yield a different set of games from the same pages.**


**0g. A one-person server gets three attempts, but a 404 gets one.**
`fetch()` had no retry: a single 502 or dropped connection aborted the whole
run before anything was written. It now retries 5xx, timeouts and connection
errors three times with backoff. **4xx is never retried, and that is the point
of the split** — a 404 is not a fault here, it is the sentinel that tells the
week loop the season has no further weeks. Retrying it would triple the
requests at the end of every season. `tests/test_fetch.py` pins all five paths.

These cost real debugging time. Don't rediscover them.

**0. A school's name can contain its own parentheses, and the LAST one is the
city.** The site writes `School (City)`, but some schools carry a parenthetical
in the name itself:

```
St Xavier (Louisville) (Louisville) [KY]     disambiguated -- Cincinnati has its own
Landmark Eagles (club) (Cincinnati)          club sides are marked
Valley (Wetzel) (Pine Grove) [WV]            county in the name
Football North (via Clarkson SS) (Mississauga) [ON]
```

The pattern used to read one `(...)` and stop, so these matched nothing and the
games vanished. Elder lost its week 5 fixture this way — and it was confusing
precisely because Cincinnati's St Xavier parses fine while Louisville's does
not. `SB_TEAM`'s stem may now consume earlier parentheticals; being non-greedy,
it lands on the last one as the city. **That stem is bounded (`{0,60}`) and the
bound is load-bearing** — unbounded it runs from every start position to the end
of the page hunting a terminator, which took the probe from 0.02s to 6.3s.

**0b. `_plausible_team`'s limits were below the real maximum.** 48 chars / 6
words, set back when this scraper read bare names ("Antwerp") and never re-cut
once the mailing city was appended. The longest legitimate OHSAA name is **50
characters**, so the filter rejected real schools by construction — Cuyahoga
Valley Christian Academy and Brecksville-Broadview Heights lost *every game of
the season*, silently. Now 80/12, and `tests/test_schedule.py` asserts every
name on the committed roster passes, so it cannot drift under the data again.

**0c. `UNRECOGNISED` is meant to reach zero, and now does.** It sat at 160
permanently, which made it useless as the alarm for the page format moving. Two
kinds of record were being counted as failures when they are simply not games:
`TBD () [TBD]` (opponent not yet settled) and `... at Foxfire (Zanesville)
cancel` (called off). Both now parse and are dropped by name, counted
separately in the scrape log. Overseas DoD schools tag a country rather than a
state — `[Japan]`, `[South Korea]` — so the tag accepts more than two letters.

**1. The scraped pages don't look like their rendered text.**
`WebFetch` reassembles pages for readability and showed a tidy one-line-per-game
view that does not exist in the HTML. The real scoreboard splits a single game
across multiple elements with the scores in separate nodes. Two fixes failed
because of this. The parser now flattens the whole page to one string and scans
with a regex anchored on the ISO date. **Trust a real fetch or the workflow
log's diagnostic dump, never WebFetch, for page structure.**

**2. Fixtures are marked `***`, and the real format has three traps.**
Confirmed by probing the live week 2 page — 466 records, all fixtures:

```
2026-08-27 6:30pm Deer Park (Cincinnati) *** at Shroder (Cincinnati) ***
2026-08-28 Lewis County (Vanceburg) [KY] *** at Morgan County (West Liberty) [KY] ***
2026-08-27 7pm Flint Beecher () [MI] *** at Petersburg Summerfield (Petersburg) [MI] ***
```

Both scores become `***`; the kickoff time is **sometimes absent**;
out-of-state teams **sometimes carry an empty city**, `Flint Beecher ()`, which
would otherwise become part of the team's identity. And **no neutral-site
`vs.` appeared at all** in 466 fixtures — the pattern accepts it anyway so a
later one isn't silently dropped.

Related latent issue: `SB_TEAM` (completed games) requires 1–40 characters
inside the parentheses, so a *played* game involving an empty-city team is
silently skipped. In practice those are out-of-state-vs-out-of-state games that
contribute nothing to Ohio ratings, so this is flagged rather than fixed.
`scrape.py` reports a per-week `UNRECOGNISED` count; a rising number there is
the signal that the page format moved.

**3. Duplicate detection must be per *week*, not per season.**
A team plays at most once a week, so two "Perry"s in one week are two schools —
but ten appearances across ten weeks are one school playing a season. Counting
per season shattered every school into ten phantom teams. Caught only by
testing against multi-week data. `check.py` now enforces this for both results
and fixtures.

**4. `os.devnull` exists.** Using it as a "no file" sentinel meant the loader
found an empty file and tried to parse it as JSON. Use an explicit falsy flag —
`schedule_path=False` is the pattern now used for `--no-schedule`.

**5. Round a projected record as a pair, not as two halves.** Rounding wins and
losses independently let `3.5-6.5` describe an eleven-game season. Losses are
derived from the rounded wins.

**6. GitHub Pages only redeploys when the workflow runs.** The workflow triggers
on schedule and manual dispatch, *not* on push. Pushing site changes alone will
not update the live site.

**7. Workflow files can't be written by remote tooling.** GitHub protects
`.github/workflows/*`. Changes to that file have to be pasted into the GitHub
web editor by hand, or committed and pushed locally.

---

## Layout

```
scripts/scrape.py        fetch scoreboards + 28 regional roster pages;
                         writes games_{yr}.csv AND schedule_{yr}.csv
scripts/resolve.py       team identity resolution (the fragile part)
scripts/ratings.py       the three models + RatingConfig + predict_margin +
                         win_probability
scripts/season_prior.py  finished season -> next season's prior + division ladder
scripts/tune.py          fits constants by walk-forward prediction
scripts/build.py         orchestrates; fits ratings, predicts the schedule,
                         writes ratings.json + both page variants
scripts/check.py         ~40 assertions; the workflow fails if these fail
                         (its date-completeness helpers are pure and tested)
tests/                   256 unit tests (pytest)
data/                    committed raw scores, schedule, rosters, prior, tuned
site/                    ONLY deployable assets: app.html, index.html, ratings.json
.github/workflows/       the weekly automation
```

`site/app.html` is the source of truth for the page; `site/index.html` is
**generated from it** by `build.py`. Edit `app.html`, never `index.html`.

`dist/preview.html` is a self-contained build with the data inlined — useful
for opening in a headless browser to check rendering without a server.

---

## Operating it

**The season lives in ONE place: `CURRENT_SEASON` in `scripts/scrape.py`.**
It used to be typed into five files, so the annual rollover was a hunt and a
missed copy would publish a finished season as the current one. The workflow
now reads `CURRENT_SEASON`, `HISTORY_SEASONS` and `PRIOR_SEASON` from that
module rather than repeating them; there are no hardcoded years left in
`update.yml`.

### Season rollover checklist (once a year, ~August)

1. Edit `scripts/scrape.py`: bump `CURRENT_SEASON`, and add the season that
   just finished to `HISTORY_SEASONS`.
2. Re-scrape the finished season if the parser has moved since
   (`python scripts/scrape.py --season <year>`), then re-run the tune.
3. **Check Actions is still enabled.** The crons only run August–December now,
   and GitHub disables a scheduled workflow after 60 days of repository
   inactivity — the bot's weekly commit is what keeps it alive, and seven quiet
   months will trip it. GitHub emails the owner; re-enabling is one button in
   the Actions tab.
4. Run the workflow manually once *before* week 1, or the season's week-1
   predictions are never captured live. 2026 lost its week 1 that way.

**Weekly runs are automatic** — Saturday 08:00, Saturday 20:00 and Sunday
13:00 ET, **August through December only**. Every run re-scrapes all weeks (scores get corrected days later) and
commits refreshed data back to the repo.

The Saturday evening run catches games added to Friday's list late, and most of
the ~26 fixtures a week that are actually played on Saturday (the season has
3,810 Friday fixtures, 259 Saturday, 131 Thursday). It costs one extra pass of
~44 requests; see "Source etiquette".

**The crons do not run at minute :00**, because every cron on GitHub is
queued at the top of the hour and a run that loses that scramble is delayed or
dropped. They fire at :13, :17 and :23. **There is no `timezone:` key on a
GitHub schedule** — cron is UTC only, and adding one risks the workflow failing
to parse so that *no* scheduled run fires, silently. `AUTOMATION_REVIEW_CONCERNS.md`
recommended adding it; that advice is struck at the top of that file.

**In UTC that middle run is a SUNDAY cron.** Saturday 20:00 EDT is midnight
UTC, so it is `0 0 * * 0`, not `0 0 * * 6` — the latter would fire Friday
evening ET, before the games it exists to catch. Verified against a real
timezone table in both EDT and EST.

**Manual run:** Actions → Update ratings → Run workflow. Safe at any time —
a mid-week run can no longer poison that week's track-record entry, because a
capture stays revisable until the games it forecasts kick off. That was not
true before 2026-08-28; see "Rules that keep it honest".

**Pushing does NOT run the workflow.** It triggers on schedule and manual
dispatch only. So a batch of changes is not validated by CI until someone runs
it — local green is not CI green, and that gap has already produced one failure
(a test whose outcome turned on floating-point residue: fine on Python 3.9
here, broken on CI's 3.12).

**How you find out a run failed.** A failure is otherwise silent: the site keeps
serving the last good build, which is correct but looks exactly like nothing
having happened.

1. The workflow opens a GitHub issue titled "Weekly update run failed", with a
   link to the run. GitHub emails the repo owner. Repeat failures COMMENT on
   the open issue rather than opening new ones, so a fortnight of breakage is
   one thread.
2. The page footer carries "Generated <timestamp>". If it has not moved since
   the last scheduled run, the run failed.
3. The Actions tab, if you go looking.

The deploy is step 15 of 17, so tests, scrape, build and check.py all have to
pass before the site changes. That is deliberate: a failed run publishes
nothing rather than publishing something wrong.

**The "Re-fit the model constants" checkbox:** leave it *unchecked* normally.
Tick it when (a) another season's data is added or re-scraped, (b) the model
itself changes, or (c) `check.py` warns that `data/tuned.json` has a stale
schema.

It is opt-in because the answer barely moves between runs, **not** because it
is prohibitive — measured end to end on the current (larger) data, the full
1,296-combination grid takes roughly 15 minutes. Most combinations run at about
6 per second; the pauses come at each new `squash_scale` / `prior_games` pair,
where `prev_cache` misses and the full-season prior has to be refitted.

Cheap enough to run locally and inspect the result before pushing, which is how
the current constants were produced:

```bash
python scripts/tune.py --seasons 2023,2024,2025 --holdouts 1,2,4,7,10
```

One ordering trap: in the workflow the re-fit step runs **before** the scrape,
so it fits on the *committed* `games_{yr}.csv`. If the parser has changed,
re-scrape the past seasons first or the fit is done on stale history.

**Because the bot commits to the repo**, the remote is often ahead. Alex uses
GitHub Desktop: **Fetch → Pull → then Push.** Doing this *before* making changes
avoids the "Newer Commits on Remote" dialog entirely.

The workflow no longer loses a deploy to that race. The Pages steps now run
*before* the data commit, and the push rebases and retries. Previously a push
that lost a race failed the job at that step, and the Pages steps below it
never ran — so a perfectly good scrape did not reach the live site.

**Never suggest dragging a *folder* onto a folder in Finder.** macOS "Replace"
deletes the entire destination. Dragging *files* into a folder is safe.

**Credentials.** Alex once offered his GitHub username and password. Declined,
and it should stay declined — GitHub no longer accepts account passwords for
push anyway.

---

## Checking the page renders

The mobile layout was rebuilt twice because it wasn't right. Below 900px the
table becomes a list with every stat as a wrapped caption; the team panel is a
single column at every width. **Alex's standing complaint is horizontal
scrolling** — verify it before claiming a layout works:

```python
# python -m pip install playwright && playwright install chromium
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch()
    for w in (1280, 390, 320):
        pg = b.new_page(viewport={"width": w, "height": 800})
        pg.goto("file:///.../dist/preview.html"); pg.wait_for_timeout(500)
        pg.locator("#rows tr .c-team button.open").first.click()
        print(w, pg.evaluate(
            "()=>document.documentElement.scrollWidth-document.documentElement.clientWidth"))
```

Expect `0` at every width.

---

## Open items

**~~Re-fit the constants~~ — DONE.** Re-fitted on the re-scraped 2023–2025 with
the paired selection rule; `data/tuned.json` is schema 4 and `check.py` runs
clean with zero warnings. Superseded text follows for context only:

**Re-fit the constants — was the top item.** `data/tuned.json` is on
schema 1 while `tune.py` is at 4, and three things have changed under it since
it was written: the out-of-state identity fix moves the 2023–2025 game graphs
that tuning fits on, the selection rule now pairs its standard errors, and the
prior is rebuilt from a corrected 2025. The committed constants were chosen by
the old rule on the old data. **Actions → Update ratings → Run workflow**, tick
*Re-fit the model constants*. Expect the published constants to move, and check
the README table after (it does not self-correct; the site footer does).

**Backfill more seasons — decided against. Do not reopen without new evidence.**
The source has data to 2000 and it is tempting. Two reasons not to:

1. *The old argument for it was wrong.* It rested on 294 configurations tying
   within one standard error, read as "not enough data". That was the wrong
   standard error. With paired differences the same grid ties 20.
2. *The seasons would be fitted, not carried.* `carry` reaches back exactly one
   season, so 2005 could never touch a 2026 rating; old seasons would serve
   only as extra held-out games for choosing constants. But the constants
   describe a sport, and the sport changed — divisions, qualification (16 per
   region through 2024, 12 from 2025), transfer rules, the mercy rule. Fitting
   across an era boundary trades variance for bias, and the bias is invisible
   because more data always *looks* like it should help.

The per-season stability report is the thing to watch: 2024 and 2025 currently
pick different optima only ±0.002 log loss apart, which is not a fit starving
for data. Revisit only if that gap widens.

**`carry=0.5` sat at the grid minimum.** The conservatism rule pushed it down
until it ran out of grid — but it was choosing among a pool the old rule had
inflated. See what the paired rule selects before widening the grid below 0.5.

**README quotes tuned constants and 76.6% accuracy in a table.** The site footer
reads `tuned.json` and self-corrects; the README does not. Check it after any
re-fit — and after the re-fit above, it will be stale.

**~~Re-scrape 2023–2025~~ — DONE.** All three were re-scraped with the fixed
parser and the constants re-fitted on the result. Recovered 811 games in 2023,
1,074 in 2024 and 1,334 in 2025 — roughly a fifth more history — with zero
games lost or altered, verified by set comparison against the previous files.
Note for next time: the workflow's backfill step skips a season whose
`games_{yr}.csv` already exists, so a future re-scrape needs the file deleted
or a manual `scrape.py --season {yr}` run.

**~~Phase 3 before Phase 4~~ — resolved by doing Harbin first**, exactly as this
note advised. The odds are computed under the real qualifier, not under record.

**Phase 6 (weekly trend history), and benchmarking against other models.** Alex
asked for a comparison against public Ohio models — Drew Pasteur's is the one
named. Two honest constraints:

- A retrospective comparison is not possible. It needs *their* week-by-week
  predictions archived, which generally are not published, and reconstructing
  them is guesswork.
- Scraping and republishing a third party's projections is a different
  proposition from joeeitel.com, where the etiquette is long established.
  That needs its own decision, not an assumption.

The defensible version is **forward-looking scorekeeping**: append each week's
predictions to a committed history file with the build timestamp, score them as
results land, and publish an accumulating track record. That is evidence rather
than a claim, it doubles as Phase 6, and it makes a comparison against any
source possible once its numbers can legitimately be recorded. Note the
existing weekly commit already gives a natural place to append.

---

## Source etiquette

joeeitel.com is a one-person site running since 2000. A full season refresh is
~44 requests, rate-limited to one per 1.5s, with a self-identifying User-Agent
pointing at the repo. **Keep all of that.** The Saturday evening run took this
from two passes a week to three (~132 requests); that is still a rounding error
against a public scoreboard, but it is the reason to think twice before adding
a fourth. Credit is on the site footer and in
the README.
