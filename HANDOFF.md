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
138 unit tests pass in CI before anything touches the network.

---

## First five minutes in a new session

```bash
cd ~/Documents/GitHub/sneaky-fb-ratings
git fetch && git status          # the bot commits here; the remote is often ahead
python -m pytest tests/ -q       # expect 138 passed; pip install -r requirements.txt if not
python scripts/build.py --generated-at 2026-08-25T00:00:00+00:00 --out /tmp/check.json --no-site
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

Ratings are in **points**: the difference between two ratings is the expected
neutral-field margin.

### Measured performance

Fitted on 2024 and 2025, evaluated walk-forward on 4,345 held-out games:

| | |
|---|---|
| Accuracy | **76.6%** |
| Log loss | 0.4776 |
| Mean margin error | 17.4 pts |

Per holdout week — the model earns its keep as the schedule graph connects:

| Week | Log loss | Accuracy |
|---|---|---|
| 1 | 0.584 | 71.6% |
| 2 | 0.567 | 72.5% |
| 4 | 0.506 | 71.9% |
| 7 | 0.404 | 80.5% |
| 10 | 0.359 | 83.7% |

### Published constants

In `data/tuned.json`, read automatically by `build.py`:

```
squash_scale     9.0
prior_games      0.5
carry            0.5
division_weight  1.0
margin_cap      49.0   (NOT tuned — a fixed guard against typos/running clock)
probScale a/b   18.62 / 98.73
```

Selected by the **one-standard-error rule**: 294 configurations sat within one
SE of the outright optimum, so the most conservative of those was taken rather
than the single best score.

**That 294 was an artefact, and `tune.py` no longer produces it.** The SE used
was the marginal one — the spread of per-game log loss for the winning config —
but every config is scored on the same games in the same order, so the
comparison is *paired*. Most of that spread is the game, not the config: a
coin-flip upset scores badly under all of them alike. Differencing first removes
the shared noise. On a smoke grid the paired rule ties 1 of 24 where the old
one tied at least 6 of the top 10. `tied_with_best()` in `tune.py` is the
function; `tests/test_tune.py` pins the behaviour. **This takes effect on the
next re-fit and is expected to move the published constants.**

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

Phases 2–6 (Monte Carlo distributions, regional playoff odds, true Harbin
simulation, what-if scenarios, weekly trend history) are sketched in Alex's
roadmap and not started. Monte Carlo belongs at build time in numpy, not in the
browser — 700 teams × 10 games × 10,000 sims is trivial in Python and painful
on a phone.

---

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

**One schedule problem does fail the build: an impossible season length.**
`check.py` refuses a team whose played + scheduled games exceed 16 (ten regular
plus five playoff rounds). This is the assertion that would have caught the
Salem merge above. Every other check passed on it — the projected record was
internally consistent arithmetic, just over a season that cannot happen.

---

## Hard-won gotchas

These cost real debugging time. Don't rediscover them.

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
tests/                   114 unit tests (pytest)
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

**Weekly runs are automatic** — Saturday 08:00 and Sunday 13:00 ET. Every run
re-scrapes all weeks (scores get corrected days later) and commits refreshed
data back to the repo.

**Manual run:** Actions → Update ratings → Run workflow.

**The "Re-fit the model constants" checkbox:** leave it *unchecked* normally.
Tick it only when (a) another season's data is added, (b) the model itself
changes, or (c) `check.py` warns that `data/tuned.json` has a stale schema.

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

**Re-fit the constants — this is now the top item.** `data/tuned.json` is on
schema 1 while `tune.py` is at 4, and three things have changed under it since
it was written: the out-of-state identity fix moves the 2023–2025 game graphs
that tuning fits on, the selection rule now pairs its standard errors, and the
prior is rebuilt from a corrected 2025. The committed constants were chosen by
the old rule on the old data. **Actions → Update ratings → Run workflow**, tick
*Re-fit the model constants*. Expect the published constants to move, and check
the README table after (it does not self-correct; the site footer does).

**Backfill more seasons (medium value, and re-assess after the re-fit).** Source
has data back to 2000. Add years to the backfill loop in
`.github/workflows/update.yml`. Note 2020 is COVID-shortened — exclude it or
report with and without. The old argument for this was that 294 configurations
tied within one SE; that was the wrong SE, not too little data, so judge the
need from what the paired rule reports.

**`carry=0.5` sat at the grid minimum.** The conservatism rule pushed it down
until it ran out of grid — but it was choosing among a pool the old rule had
inflated. See what the paired rule selects before widening the grid below 0.5.

**README quotes tuned constants and 76.6% accuracy in a table.** The site footer
reads `tuned.json` and self-corrects; the README does not. Check it after any
re-fit — and after the re-fit above, it will be stale.

**~28 fixtures sat in rare formats** the probe's sample cap didn't print. The
per-week `UNRECOGNISED` count in the scrape log says whether they still matter.

**`SB_TEAM` still skips a played game with an empty city.** Unchanged and still
the right call — those are out-of-state-vs-out-of-state games that contribute
nothing to Ohio ratings. Noted here only because the state-tag work sits next
to it and it is easy to think it was covered.

**Phase 3 before Phase 4 is questionable.** Regional playoff odds based on
record alone will look wrong to anyone who follows Ohio football, because
seeding is Harbin, not record. Consider doing the Harbin scorer first.

---

## Source etiquette

joeeitel.com is a one-person site running since 2000. A full season refresh is
~44 requests, rate-limited to one per 1.5s, with a self-identifying User-Agent
pointing at the repo. **Keep all of that.** Credit is on the site footer and in
the README.
