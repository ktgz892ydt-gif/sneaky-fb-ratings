# Fable model re-review — 2026-08-28

Second-pass, line-by-line review of the repo at commit `3281277` (clean tree),
covering every script — including the six new ones — the workflow, the page,
the docs, and the data, with specific attention to whether the math holds up
in the real world.

Note on which tree was reviewed: the changes live in
`~/Documents/GitHub/sneaky-fb-ratings`, not in the Desktop "code test" folder
used for the first review. This review covers the real clone.

## Verified before reviewing

- `pytest`: **250 passed**
- `check.py`: passes with 2 expected warnings (week 2 mid-flight; the one
  overdue fixture is the known Dunbar/Stivers case)
- The **weekly build path** — a plain build with history recording enabled,
  the exact code path the workflow runs — completes cleanly (run against a
  scratch copy of the history log)
- A pinned rebuild is **semantically identical** to the committed
  `site/ratings.json`, even on different numpy/scipy versions (the byte diff
  is only key ordering from a code commit made after the last published build)
- Two failure reproductions: an empty schedule, and a post-regular-season
  schedule (see the time bomb below)
- Independent walk-forward validation of the margin and probability
  calibration over **10,512 held-out predictions** from 2024–25

## Verdict

The repo has moved a long way since the first review, and almost entirely in
the right direction. The weekly-update breaker is **fixed and verified at
HEAD**. The re-fit landed (schema 4, on re-scraped parser-v2 history), the
scoreline projections got a real total model with plausibility rules, and the
new playoff simulation, Harbin implementation, prediction history, and
head-to-head machinery are statistically sound — in several places genuinely
better statistics than most published rating systems bother with.

One **verified time bomb** remains: the build will start failing its own
checks the week the regular season ends.

## The weekly update script: fixed, and verified

The breaker was the `record_history` name shadowing in `scripts/build.py` —
the boolean parameter shadowed the imported function, so every plain
`python scripts/build.py` (the command the workflow runs) died with
`TypeError: 'bool' object is not callable`. The fix
(`record as record_snapshot`) is committed at HEAD.

Verified by running the actual weekly code path with history recording
enabled: it completed and correctly **replaced** the week-2 capture, since
none of the week-3 games it forecasts have kicked off — exactly what the
freeze-on-kickoff rule should do.

## The math, tested against reality

### The ×1.47 margin calibration is correct and necessary

A squashed Bradley–Terry fit mathematically *must* compress rating
differences relative to true expected margins — the sigmoid discounts
blowouts during fitting, and that discount survives the rescale (Jensen's
inequality). The measured 1.47 slope is theory showing up in data, not a
fudge.

- It removes essentially all signed bias: 0.58 → −0.06 points.
- Pinning the intercept at zero is validated: a free fit returns slope
  1.4712, intercept −0.06 (home field is already inside the prediction).

**One refinement worth knowing:** a single linear constant slightly misfits
the tails. Bucketing the 10,512 walk-forward predictions by raw predicted
margin:

| displayed spread | n | mean actual | calibrated says | residual |
|---|---|---|---|---|
| ~8–12 pts | 1,266 | 10.2 | 8.0 | understated ~2 |
| ~16–22 pts | 1,500 | 18.9 | 16.5 | understated ~2.4 |
| ~28–37 pts | 1,141 | 29.8 | 28.3 | ≈ right |
| ~44+ pts | 513 | 41.0 | 44.4 | overstated ~3.4 |

Mid-range spreads run about 2 points hot in reality; extreme spreads about
3–4 points cold. Against a per-game residual of ±21 points this is well
inside noise for any single game, but it is systematic — a gently saturating
curve (or a two-piece slope) would fix it. Not urgent; worth a line in the
docs.

### The win probabilities are genuinely calibrated

Recomputed independently from the walk-forward sample (favourite-folded):

| board said | favourite won | n |
|---|---|---|
| 50–60% | 54.8% | 2,124 |
| 60–70% | 66.1% | 1,743 |
| 70–80% | 73.7% | 1,779 |
| 80–90% | 83.7% | 1,863 |
| 90–95% | 93.0% | 1,098 |
| 95%+ | 98.1% | 1,905 |

The three-scale design — squash for fitting, prob-scale for probability,
margin-scale for display — is conceptually right, and `check.py` now
*proves* on every build that the two published numbers came from the right
scales, by reproducing each fixture's probability from its margin ÷
`marginScale`.

### The re-fit is clean

- Schema 4, fitted on re-scraped parser-v2 history (n = 5,065 held-out games)
- The paired one-standard-error rule ties **20 of 1,296** configurations
  (the old marginal rule admitted 294)
- Nothing sits on a grid edge; `carry = 0.6` is now interior
- The honestly-explained "worse" headline (75.8% / 0.485 vs the old
  76.6% / 0.478) is exactly right: the parser fixes recovered a fifth more
  games, and the recovered games are the hard ones. Not comparable, not a
  regression.

### The playoff simulation is sound

- Level-2 Harbin entanglement is why simulation (not multiplied independent
  probabilities) is required, and the vectorised code preserves it correctly.
- What-ifs are exact conditionals read off the same 10,000 seasons; the
  law-of-total-probability check (a team's own odds must sit between its
  if-win and if-lose odds) catches the one invisible mistake, an inverted
  road game.
- The tie-break jitter fix is correct (the documented 40–61% skew from
  alphabetical tie-breaking was real).
- The conservation laws all pass: odds sum to exactly 12 per region, byes to
  4, top seed to 1.
- The Harbin formula recovered by least squares from the source's published
  column, validated at 86% exact agreement with the out-of-state
  approximation flagged per team, is model archaeology done right.

### The head-to-head is more careful than most academic comparisons

Intersection-only scoring, exact McNemar on the discordant pairs, a
small-sample floor on the continuous verdict, attribution treated as a
license term. The current payload correctly reports 2 shared games as
"indistinguishable".

## Findings, ranked

### 1. HIGH — the build will fail its own checks when the regular season ends

Verified by reproduction: when no regular-season fixtures remain,
`build.py` skips the simulation, no team carries `playoffOdds`, and
`check.py` fails with **"0 regions carry playoff odds, expected 28"**.

That is correct behavior for a mid-season schedule loss — but it is also the
legitimate state of every build from about **November 1** onward. The first
Saturday after week 10 completes, the weekly run goes red and the board
stops updating for the playoffs, with the failure issue pointing at a check
rather than a real fault.

**Fix:** gate the conservation checks on the simulation actually having run
— e.g. require odds only while `weeksLoaded` tops out below week 10, or
check an explicit "sim skipped" marker in the payload. Do it before late
October. When fixing, note that this check is currently doing double duty as
the total-schedule-loss detector — keep one explicit floor for that case.

### 2. MEDIUM — `fetch()` still has no retry

Unchanged from the first review: one transient 502 from a one-person site
aborts the run before anything is written. Softened now — three runs a week,
and a failure opens an issue that emails the owner (confirmed working) — but
three attempts with backoff on 5xx/timeouts (keeping 404 as end-of-season)
is still worth the ten lines.

### 3. MEDIUM — season hardcoded in five places; crons run all year

Also unchanged: `2026` and `2023–2025` are wired into the workflow and the
build fallback, and the (now three) crons fire every weekend year-round —
roughly 90 off-season scrapes a year against a site the docs promise to
treat gently, republishing a finished season as current from January.

### 4. MEDIUM — do not apply AUTOMATION_REVIEW_CONCERNS' timezone suggestion

That document recommends adding `timezone: "America/New_York"` to the cron
entries. **GitHub Actions does not support a timezone key on schedules** —
cron is UTC-only, and adding the key risks invalidating the workflow so no
scheduled runs fire at all. The current file rightly didn't adopt it; strike
it from the doc so it doesn't get pasted in later.

The same doc's minute-zero advice (move off `:00` to dodge GitHub's
high-load window) is real and cheap — take that half.

### 5. LOW — stale copy after the re-fit

The tuned `squash_scale` moved from 9 to **8**, but:

- The method panel and README table still show scale-9 fractional wins.
  At scale 8: 3 pts → .59, 7 → .71, 14 → .85, 17 → .89, 21 → .93, 45 → 1.00.
- The panel says "a fixed prior worth about 1.5 pseudo-games" — the tuned
  value is 0.5.
- The footer prints `squash_scale` labeled **"margin scale"**, which now
  collides with the real `marginScale` constant (1.47).
- The README's automation section omits the new Saturday-evening run, and
  its Layout block omits the six new scripts.

### 6. LOW — smaller items

- **The scorecard's calibration table pools live and backtest bins.**
  `history.score()` keys the headline numbers by kind but accumulates one
  shared `bins` dict — a quiet exception to the project's own "never
  pooled" rule. Immaterial at 3 live games; fix before it isn't.
- **`head_to_head` pools snapshots of both kinds** when building `ours_by`.
  Harmless today (rival records exist only for 2026, which has only live
  captures), but a future 2026 backfill would let replayed predictions into
  the live comparison. Filter to `kind == "live"`.
- **`pct()` rounds 0.996 to "100%"** in the playoff-odds column and panels.
  The calibration display carefully caps its label at "95%+"; the odds
  column should cap at ">99%" for the same reason.
- **`winDist` labels count ties as losses** — the bar chart writes records
  as k–(gp−k) with ties folded into the loss column. Rare, cosmetic.
- **Schedule CSVs are still written unguarded** in `scrape.py`. Downstream
  checks now catch the consequences loudly (zero-game teams fail; a
  vanished week fails; a vanished schedule fails via the odds check), so
  this is defense-in-depth rather than a hole.
- **`MAX_SEASON = 16`** still carries the "ten plus five" comment — that
  arithmetic says 15; six playoff rounds says 16 is right. Fix the comment.
- **The Desktop "code test" snapshot is now misleading**: it holds an
  abandoned line of work (the `track.py` ledger, superseded by the better
  `history.jsonl` design) that was never committed. Archive or delete it so
  nobody reviews or resumes the wrong tree again.

## What is now excellent

- **The freeze-on-kickoff history rule** is a better design than the
  write-once ledger it replaced, and the module docstring documents the
  failure that motivated it (a mid-week manual run locking in a forecast
  made from 27 of 357 games).
- **`check.py` has become a real invariant suite** — conservation laws on
  the Monte Carlo, scale-swap detection, date-based "this week didn't
  parse" failures, zero-game team detection (which caught two schools
  silently dropped by a 48-character name limit), and live/backtest
  separation enforced structurally.
- **Parser versioning** (`parser_versions.json`) replacing the mtime guard
  — the retune step now refuses to fit on history the current parser would
  not produce.
- **The decision log**: HANDOFF now argues *against* backfilling to 2000 on
  era-boundary grounds (12-vs-16 qualifiers, mercy rule) with the
  per-season stability report as the tripwire. The right call, made for the
  right reason, written down.

---

*Reviewed by Claude (Fable 5). Verification: 250-test suite, weekly build
path with history recording, check.py, semantic rebuild comparison,
empty-schedule and post-season reproductions, and independent walk-forward
validation of margin and probability calibration over 10,512 held-out
games.*
