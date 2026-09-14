# run_daily TODO

Improvement backlog for the daily benchmark pipeline, DuckDB store, and the
`daily_results` MCP server.

All findings below were verified against the central DB
(`/var/www/html/daily2/daily_llm_benchmark.duckdb`) and against the working
tree at `4b83622` on 2026-09-14 unless the item is explicitly marked
*(assumption)*. Completed items are not kept here — see the git history.

Dependency order for what is left:

```
T19  →  T4 · T5  →  T6(MCP)  →  T7 · T8  →  T9–T14 · T17
                                             T15 · T16 · T20–T23 (independent)
```

`T17` blocks on `T12` (it needs the health columns exposed). Everything else
in P3/P4 is independent.

## Landed

**T1** (remove `short_run`), **T3** (cut off baseline contamination sources)
and the query-layer half of **T6** shipped in `17bbd73` — see that commit
message for the reasoning and the measured before/after. In short: the
measurement shape is now fixed (`OUT_TOKEN_LENGTH` / `BENCHMARK_ITER_NUM`),
`runs_with_flags` is the single definition of `is_partial` / `excluded`,
`perf_stats` and the two trend queries no longer see excluded or non-daily
runs, and the dead `runs.short_run` column is gone from the live DBs.

**T2** (populate the analysis tables centrally) and **T18** (the docs that
claimed they were populated) shipped in `a707684`, and the data layer they
sit on landed across `e6c93e4` … `4b83622`. In short: `daily/data` is now the
only thing that reads or writes benchmark results, ingest carries the
machine's `analysis` block into `analysis_results` / `analysis_comparisons`
(318 of 331 runs back-filled — 11 predate the block, 2 lost their source
file), a validation layer records invariants instead of clamping them, and
the legacy pickle stack is deleted. See `daily/data/README.md` for the rules
and the guards that keep them in one place.

Three things that came out of that work and are worth carrying forward:

- **The fleet cycle is keyed by build, not by date.** Machines start their
  nightly run at their own local times, so a batch straddles midnight
  (observed 23:41 → 00:11) and no `day_start_hour` splits it the same way for
  every machine. `ov_build` maps 1:1 to `ov_sha` across the fleet.
- **`analysis_comparisons` carries only the top regressions**, because that
  is all the summary JSON has room for. Per-verdict totals must come from
  `analysis_results`; counting the comparison rows reports every machine as
  100 % regressed.
- **Validation found 82 historical violations** — 61 runs with perf rows but
  no declared expectation, 21 with `success` above `expected` (LNL-03 on
  2026-08-25 stored 81 token series while declaring it expected 4). All from
  July/August; nothing on or after 2026-09-07. A record of the old pipeline,
  not a live defect.

One constraint that came out of the earlier work and still shapes T19/T5: **`is_partial`
rests on `runs.test_filter` alone.** `selected_cases < expected_cases` cannot
detect a narrowed run, because `_cases(summary)` sums only the tests pytest
*collected* — a `-k llama` run reports the same figure for both. And
`selected_cases` deliberately does not subtract `skipped_cases`: 117 of 208
daily runs skip at least one case, so treating a skip as a narrowing would
mark most of the fleet partial and empty the baseline cohort.

---

## P0 — Data integrity (prerequisite for everything else)

### T19. Record the true suite size, so `is_partial` covers more than `-k`

**Why:** `is_partial` only fires when `test_filter` is set. A run narrowed by
`--tests`, or one where collection itself dropped cases, still looks full.
Historical rows (all 239) have no `test_filter` and cannot be reclassified.

- [ ] Run a `--collect-only` pass (or keep a static case registry) and store
      the full-suite case count on the run, so `selected_cases` has something
      real to be compared against
- [ ] Fold `--tests` into `runs.test_filter` as well, not just `-k`

--

## P1 — `run_kind` classification

### T4. Make `run_kind` declared, not inferred

**Why:** it is currently regex-inferred from the free-text `purpose` field
(`daily/data/filters.py`, `classify_run_kind`). The `test` pattern matches
`\bjenkins\b`, `\bci\b` and `\bvalidation\b`, so a daily run whose purpose
mentions Jenkins is misclassified.

- [ ] Add an explicit `--run-kind {daily,pr,test,manual}` argument to `run.py`
- [ ] Demote `classify_run_kind()` to a legacy fallback for historical rows only
- [ ] Stop depending on free-text `purpose` for classification

### T5. Extend the taxonomy to express the three cases we care about

**Why:** the current four values (`daily | pr | test | manual`) cannot separate
a timer-triggered daily from a manual daily. Measured: `daily pipeline sungeunk`
(90 runs, launched by hand) and `daily_pipeline timer` (54 runs, timer) both
resolve to `run_kind = 'daily'`. Current distribution: `daily` 208, `test` 17,
`pr` 7, `manual` 7.

- [ ] Add `runs.trigger` — `timer` | `manual` | `ci` — orthogonal to `run_kind`
- [ ] Final classification = `run_kind` × `trigger` × `is_partial` (the last
      one already exists in `runs_with_flags`, driven by `test_filter`)
- [ ] Define the default baseline cohort as
      `run_kind = 'daily' AND trigger = 'timer' AND NOT partial`

### T6. Thread `run_kinds` through the MCP layer

**Why:** the query layer takes `run_kinds` everywhere now (`17bbd73`), but
`daily/mcp_server/server.py` never passes it, so every MCP tool still answers
from whatever cohort its own SQL happens to select. Measured contamination in
the last 10 days: `run_kind='pr'` 547 perf rows and `run_kind='test'` 112
against `daily`'s 9393.

- [ ] Expose `run_kinds` on every MCP tool, defaulting to `("daily",)`

---

## P2 — OV-first / GenAI-second attribution

### T7. Lower the GenAI bump frequency (highest impact)

**Why:** measured, the OV build and the GenAI revision **always change
together**. Across every `run_kind='daily'` build in the DB,
`count(DISTINCT genai_commit) = 1` per `ov_build` — checked on the 12 most
recent (23030, 23029, 23019, 23005, 22990, 22987, 22985, 22983, 22981, 22979,
22975, 22967), with no exception. The two variables are perfectly collinear,
so attributing a regression to OV is impossible in principle; no amount of
tooling fixes it.

- [ ] Bump OV daily but GenAI on a fixed, slower cadence (e.g. weekly), so that
      OV-only deltas exist
- [ ] **At each GenAI bump, run a matched pair on one pinned `ov_build`:
      old GenAI vs new GenAI.** Cadence separation alone does not create
      GenAI-only intervals — if OV moves daily, the interval containing a
      weekly GenAI bump also contains an OV bump, and T8's rule 2 never fires.
      The pinned pair is what makes GenAI attribution possible at all.
- [ ] Tag the GenAI bump run separately and treat it as a baseline reset point
- [ ] *Owner: this is a release-process change, not a code change — needs an
      owner outside this repo before T8 is worth building.*

### T8. Hierarchical attribution logic

Blocked on T7: rules 1 and 2 both require intervals that the current bump
schedule never produces.

- [ ] Define the comparison group key as `ov_build` (primary) →
      `genai_commit` (secondary) → `gpu_driver_version` (tertiary) →
      `model_cache` (quaternary)
- [ ] Attribution algorithm:
      1. regression across an interval where only `ov_build` changed within a
         fixed `genai_commit` → **OV**
      2. otherwise, an interval where only `genai_commit` changed within a
         fixed `ov_build` → **GenAI** *(requires T7's pinned pair run)*
      3. otherwise, single-machine only + abnormal `throttled_sample_ratio` /
         `gpu_clock_ratio` → **machine**
      4. otherwise → `unattributed`
- [ ] Add `analysis_comparisons.attributed_to`
      (`ov` | `genai` | `driver` | `model_cache` | `machine` | `unknown`)
- [ ] Use cross-machine agreement as a signal — a single build is deployed to
      all 9 machines, so "N of 9 machines degraded" is a strong discriminator.
      Verified: on build 23030, `qwen3-vl-4b-instruct` `1st` degraded on 4 of 9
      machines.

---

## P3 — MCP tool API

- [ ] **T9.** Add `daily_results_latest_summary()` — one call for the latest
      daily run across all machines. Today this requires hand-writing the
      `latest_run_per_machine ⋈ runs ⋈ functional_issues ⋈ run_machine_health`
      join every time.
- [ ] **T10.** Add `daily_results_compare_builds(machine, ov_build_a, ov_build_b)`.
      There is **no build-scoped comparison tool at all**; `trend_regressions`
      is time-window based (`daily/data/read.py`) and cannot answer
      "what broke in this build?".
- [ ] **T11.** Add `daily_results_functional_issues(build=..., machine=...)` —
      currently only reachable through hand-written `run_sql`.
- [ ] **T12.** Add `daily_results_machine_health(run_id | machine)` exposing
      `run_machine_health` / `machine_monitor_stats`. Needed to automate the
      fluctuation verdict, and a prerequisite for T17. Measured on the latest
      runs: RAPTOR-ELLY `gpu_clock_ratio = 0.18`, MTL-01
      `max_throttle_ratio = 0.83`.
- [ ] **T13.** Make the 500-row cap in `run_sql` a tool argument
      (`daily/mcp_server/server.py`, `MAX_ROWS`) and state "aggregate first" in the docstring. A single model over
      10 days is 516 raw rows, so it always truncates.
- [ ] **T14.** Rename `pct_diff`. The value is the fraction
      `(value - win_median) / win_median` (`daily/data/read.py`,
      `daily/data/schema.sql`), not a percentage, so a 190 % regression
      is reported as `1.9`. Rename to `ratio_diff` or multiply by 100. Same
      applies to `cv` = `win_mad / win_median`.
      *Low risk, can land early: those two definitions are the only sites in
      the tree, and nothing downstream references the name.*

---

## P4 — Analysis quality and docs

- [ ] **T15.** History retention. Measured run counts: PTLH-02 10, MTL-01 11,
      BMG-02 13, dg2alderlake 14, LNL-04 15, ARLH-01 16, RAPTOR-ELLY 17,
      PTLH-01 39, LNL-03 104. `trend_regressions(baseline_days=21,
      min_baseline_points=7)` cannot be satisfied on most machines, so
      regression detection effectively only works on LNL-03 and PTLH-01.
      *(assumption: caused by server-side artifact cleanup — needs confirming)*
- [ ] **T16.** Separate within-`prompt_idx` variance from run-to-run variance.
      `perf_stats` currently mixes raw per-prompt rows into one z-score window;
      measured, a single series carries two different `win_median` values
      (82.91 and 250.47) for the same model/precision/bucket/exec_mode.
- [ ] **T17.** Automatically exclude or flag (`degraded`) runs whose
      `throttled_sample_ratio` / `gpu_clock_ratio` cross a threshold before
      using them in perf comparisons. Blocked on T12.
