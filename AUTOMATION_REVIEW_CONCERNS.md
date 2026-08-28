# Automation Review Concerns

Reviewed on 2026-08-28, focusing specifically on the weekly GitHub Actions
update path in `.github/workflows/update.yml` and the scripts it calls.

## Summary

The workflow is active on GitHub and manual runs are reaching Actions. During
review, the committed update path had one blocking bug: the normal
`python scripts/build.py` step failed before the site/data refresh could
complete. The current local working tree appears to contain the right fix for
that bug, and a temporary full-build reproduction passed after that change.

The schedule itself is also close but not ideal: it is pinned to UTC, shifts
after daylight saving time ends, and fires at the top of the hour, which GitHub
documents as a higher-risk time for delayed or dropped scheduled runs.

## Concerns

### 1. Weekly build failed when history recording was enabled

Severity: high.

`scripts/build.py` imports `history.record` as `record_history`, but `main()`
also has a boolean parameter named `record_history`. Inside `main()`, the local
boolean shadows the imported function. When the normal weekly build reaches the
history-writing block, it tries to call that boolean:

```text
TypeError: 'bool' object is not callable
```

This reproduced locally when running the same plain build command the workflow
uses against the committed code:

```bash
python scripts/build.py
```

The issue is hidden by reproducibility checks that pass `--no-history`, because
that avoids the failing call path.

Relevant files:

- `.github/workflows/update.yml`: the workflow calls `python scripts/build.py`.
- `scripts/build.py`: imports `record as record_history`.
- `scripts/build.py`: `main(..., record_history=True)` shadows that name.
- `scripts/build.py`: calls `record_history(hpath, snap, played=played)`.

Suggested fix:

- Rename the boolean parameter to something like `write_history=True`; or
- Rename the imported function to something like `record_history_snapshot`.

Current local status:

- The working tree now imports `record as record_snapshot`.
- The history-writing block now calls `record_snapshot(...)`.
- A temporary full reproduction of the workflow's build/check path passed:
  `python scripts/build.py`, then `python scripts/check.py`.

Suggested follow-up:

- Commit the `scripts/build.py` fix with this review note.
- Add a test that exercises `build.main()` with history enabled, or keep a
  temporary full build/check reproduction as part of the review process before
  changing the workflow.

### 2. The schedule is not truly 8:00 AM Eastern all season

Severity: medium.

The workflow currently uses UTC cron entries:

```yaml
- cron: "0 12 * * 6"
- cron: "0 17 * * 0"
```

Those correspond to:

- Saturday 8:00 AM EDT during daylight saving time.
- Sunday 1:00 PM EDT during daylight saving time.
- Saturday 7:00 AM EST after daylight saving time ends.
- Sunday 12:00 PM EST after daylight saving time ends.

That means the current comments are honest, but the schedule does not preserve
the intended wall-clock time during the playoff weeks after DST ends.

Suggested fix:

Use GitHub Actions' timezone support:

```yaml
schedule:
  - cron: "7 8 * * 6"
    timezone: "America/New_York"
  - cron: "17 13 * * 0"
    timezone: "America/New_York"
```

This keeps the workflow on Eastern time year-round.

References:

- GitHub Actions schedule syntax:
  https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule
- GitHub schedule event behavior:
  https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule

### 3. The current cron fires at the top of the hour

Severity: medium.

GitHub documents that scheduled workflows can be delayed during periods of high
load, especially at the start of every hour. If load is high enough, queued
scheduled jobs can be dropped.

The current schedule fires at minute `0`:

```yaml
- cron: "0 12 * * 6"
- cron: "0 17 * * 0"
```

Suggested fix:

Move off minute `0`, for example:

```yaml
- cron: "7 8 * * 6"
  timezone: "America/New_York"
- cron: "17 13 * * 0"
  timezone: "America/New_York"
```

The exact minute is not important. The important part is avoiding the top of the
hour while keeping the run comfortably after games have finished.

### 4. Scheduled runs have not happened yet

Severity: low.

As of this review, GitHub shows the workflow as active, and manual
`workflow_dispatch` runs are being created. The recent run history is manual
only, which is expected because the workflow was added during the week and the
first scheduled Saturday run had not arrived yet.

The next scheduled run under the current UTC configuration should be Saturday,
2026-08-29 at 12:00 UTC, which is Saturday, 2026-08-29 at 8:00 AM EDT.

Suggested check:

After that time, inspect the Actions tab for a run with event `schedule`. A
successful scheduled run should scrape, build, check, deploy Pages, and commit
refreshed data.

## Positive Notes

- The workflow file exists on the default branch and GitHub reports it as
  active.
- Manual dispatch is working, so the workflow is syntactically registered.
- The workflow includes `workflow_dispatch`, which is a useful fallback.
- The workflow has failure reporting through `actions/github-script`.
- A failed run successfully opened a GitHub issue titled
  `Weekly update run failed`, so the notification path appears to work.
- The data process re-scrapes all weeks instead of appending, which is the right
  approach for corrected scores.
- The history logic is conceptually strong: it records only the next prediction
  week and freezes a capture once any game it predicted has been played.

## Recommended Next Steps

1. Fix the `record_history` name collision in `scripts/build.py`.
2. Run `python scripts/build.py` without `--no-history` in a temporary copy or
   controlled test path to confirm the weekly build path succeeds.
3. Change the schedule to use `timezone: "America/New_York"`.
4. Move the scheduled minutes away from `0`.
5. After Saturday, 2026-08-29 at 8:00 AM EDT, confirm that GitHub created a run
   with event `schedule`.
