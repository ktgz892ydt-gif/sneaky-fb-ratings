# Review Concerns

Reviewed: 2026-08-28

This is a skeptical review of the repository after the latest changes. Overall,
I do not see a fatal math or process problem. The main model, scraper discipline,
team identity handling, Harbin/playoff simulation, and validation suite are much
stronger than a typical public ratings project.

The items below are the concerns I would address, or at least be ready to
explain, before sharing the project with someone likely to scrutinize the math
and methodology.

## Findings

### 1. Projected-score edge case

Two generated fixtures currently have a displayed `-0.5` margin but a tied
projected score:

- Black River (Sullivan) vs Firelands (Oberlin): margin `-0.5`, projected `22-22`
- Sandusky (Sandusky) vs Norwalk (Norwalk): margin `-0.5`, projected `25-25`

That is small numerically, but it reads strangely in public: if the away team is
shown as a half-point favorite, the projected score should not be tied.

Relevant locations:

- `scripts/build.py`: `projected_score()`
- `scripts/check.py`: projected-score consistency checks
- `tests/test_predict.py`: tests cover positive `+0.5`, but not exactly `-0.5`

Suggested fix: make the favorite-score check explicit for both signs, then add a
test for exactly `-0.5`.

### 2. Public methodology wording is stale

The README correctly explains that a rating difference is not itself an expected
margin, and that the displayed margin is calibrated. The public site still has an
older sentence saying:

> the difference between two ratings is the expected margin on a neutral field.

That contradiction is probably the easiest thing for a math-minded reader to
notice and challenge.

Relevant locations:

- `site/app.html`
- generated `site/index.html`
- stale code comment in `scripts/ratings.py`

Suggested fix: update the site method copy to match the README, then rebuild
`site/index.html`.

### 3. Projected scores are not archived in the prediction history

`data/history.jsonl` stores predicted margin and win probability, but not
projected home/away scores. That is fine if the public claim is only about
margin and win-probability accuracy. But if projected scores become a visible
feature, there is no immutable scoreline track record.

Relevant locations:

- `scripts/history.py`
- `scripts/backfill_history.py`

Suggested fix: if projected scores are meant to be evaluated later, add
`projectedHomeScore` and `projectedAwayScore` to the live prediction snapshot
before many public predictions accumulate.

### 4. Head-to-head log-loss verdict can overstate tiny samples

The current head-to-head payload has only 2 shared games against Fantastic 50.
Accuracy is correctly treated as indistinguishable, but the JSON still reports a
`loglossVerdict` of `clear`.

With only two games, that is too strong. The visible page mostly avoids making
that claim, but the payload should not carry it either.

Relevant locations:

- `scripts/rivals.py`
- `scripts/check.py`

Suggested fix: add a minimum shared-game threshold before allowing `loglossVerdict`
to be `clear`, similar to the existing small-sample guard for accuracy.

### 5. Retune stale-data guard relies on filesystem mtimes

The GitHub Actions re-fit step checks whether `scripts/scrape.py` is newer than
historical `data/games_*.csv` files using `-nt`. This is understandable, but
checkout filesystem mtimes are a brittle proxy for whether historical data was
produced by the current parser.

Relevant location:

- `.github/workflows/update.yml`

Suggested fix: use an explicit parser/data version marker, or document this as a
guardrail rather than a proof.

### 6. Undefined CSS variables in the site

The site CSS references variables that are never defined:

- `--display`
- `--mono`
- `--ink-soft`

Browsers fall back, so this does not appear to break the page, but it means the
affected playoff/scorecard styling is less intentional than the source suggests.

Relevant locations:

- `site/app.html`
- generated `site/index.html`

Suggested fix: define those variables in `:root`, or replace those references
with the already-used font/color variables.

### 7. Handoff markdown has stale test counts

`HANDOFF.md` still refers to older test counts. The current suite has 221 tests.
This is not public-facing, but stale handoff notes can confuse future work.

Relevant location:

- `HANDOFF.md`

Suggested fix: update the expected test count after the next code change.

## Verification Run

These checks were run against the current tracked repo:

- `python -m pytest`: 221 passed, with one local urllib3/OpenSSL warning
- `python scripts/check.py`: passed with one warning
- Offline `scripts/build.py`: completed cleanly
- Data syntax validation: CSV, JSON, and JSONL parsed cleanly
- Payload validation: no duplicate exact payload games, no non-finite numbers,
  no impossible season lengths, and no implausible team-score values such as
  `1`, `2`, `4`, `5`, or `11`

The one `check.py` warning was that 427 games across two weeks is low. Given the
repo state was reviewed on Friday, 2026-08-28, this is likely a data-timing issue
while Week 2 results are still incomplete, not a code problem.

## Confidence

The strongest parts of the project are:

- the scraper's loud failure behavior and source-specific parser tests
- team identity preservation using city/state/school ID
- the separation between ratings, official Harbin qualification, and playoff
  forecasting
- the append-only prediction history
- the validation suite, especially conservation checks for playoff odds

The most likely disagreements from outside reviewers are methodological rather
than catastrophic:

- whether margin-informed Bradley-Terry is the right headline model
- whether score projections should be shown at all this early in a season
- whether the projected-score total model is too simple
- whether the public page should emphasize uncertainty even more

My practical recommendation is to fix findings 1 and 2 before sharing widely.
After that, the project is in a defensible shape.
