# run_daily TODO

Improvement backlog for the daily benchmark pipeline, DuckDB store, and the
`daily_results` MCP server.

All findings below were verified against the central DB
(`/var/www/html/daily2/daily_llm_benchmark.duckdb`) and against the working
tree at `17bbd73` on 2026-09-07 unless the item is explicitly marked
*(assumption)*. Completed items are not kept here — see the git history.

Dependency order for what is left:

```
T2 · T19  →  T4 · T5  →  T6(MCP)  →  T7 · T8  →  T9–T14 · T17 · T18
                                                  T15 · T16 (independent)
```

`T2`'s backfill sits after `T4`/`T5` on purpose: backfilling 239 runs before
the baseline cohort is redefined (`run_kind × trigger × partial`) means doing
it twice. `T18` blocks on `T2` (it can only stop warning once the tables are
populated); `T17` blocks on `T12` (it needs the health columns exposed).

## Landed

**T1** (remove `short_run`), **T3** (cut off baseline contamination sources)
and the query-layer half of **T6** shipped in `17bbd73` — see that commit
message for the reasoning and the measured before/after. In short: the
measurement shape is now fixed (`OUT_TOKEN_LENGTH` / `BENCHMARK_ITER_NUM`),
`runs_with_flags` is the single definition of `is_partial` / `excluded`,
`perf_stats` and the two trend queries no longer see excluded or non-daily
runs, and the dead `runs.short_run` column is gone from the live DBs.

One constraint that came out of it and still shapes T19/T5: **`is_partial`
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

### T2. Populate `analysis_results` / `analysis_comparisons` in the central DB

**Why:** measured **0 rows** in both tables, against 239 `runs` and 167
`functional_issues` rows. `functional_issues` is written by the ingest writer
(`daily/viewer/ingest/writer.py:302`), but the other two are only written by
`analyze_run` (`daily/analysis/engine.py:155`), which the ingest path never
calls.

Consequence: every already-designed column is unusable — `verdict`,
`within_fluctuation`, `history_median`, `history_mad`, `history_sigma`,
`history_cv`, `worsening_z`, `reference_source`, `release_improvement_pct`.

- [ ] Add a post-ingest job that runs `analyze_run` → `write_analysis_to_db`
      against the central DB
- [ ] `write_analysis_to_db` currently swallows failures with a `log.warning`
      (`daily/analysis/persistence.py:88`) and also silently returns via
      `log.debug` when a table is missing (`:75-78`). Add a counter/alert — the
      silent skip is why this went unnoticed.
- [ ] **Backfill the existing 239 runs — do this after T4/T5 land.** The
      backfill bakes in whatever baseline cohort is current, so running it
      before `trigger` / `partial` exist means redoing it.

---

## P1 — `run_kind` classification

### T4. Make `run_kind` declared, not inferred

**Why:** it is currently regex-inferred from the free-text `purpose` field
(`daily/viewer/ingest/loader_new.py:43-58`). The `test` pattern matches
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
      is time-window based (`daily/viewer/queries.py:672`) and cannot answer
      "what broke in this build?".
- [ ] **T11.** Add `daily_results_functional_issues(build=..., machine=...)` —
      currently only reachable through hand-written `run_sql`.
- [ ] **T12.** Add `daily_results_machine_health(run_id | machine)` exposing
      `run_machine_health` / `machine_monitor_stats`. Needed to automate the
      fluctuation verdict, and a prerequisite for T17. Measured on the latest
      runs: RAPTOR-ELLY `gpu_clock_ratio = 0.18`, MTL-01
      `max_throttle_ratio = 0.83`.
- [ ] **T13.** Make the 500-row cap in `run_sql` a tool argument
      (`daily/mcp_server/server.py:26`, `MAX_ROWS`; enforced at `:85` and
      `:138`) and state "aggregate first" in the docstring. A single model over
      10 days is 516 raw rows, so it always truncates.
- [ ] **T14.** Rename `pct_diff`. The value is the fraction
      `(value - win_median) / win_median` (`daily/viewer/queries.py:653`,
      `daily/viewer/schema.sql:413`), not a percentage, so a 190 % regression
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
- [ ] **T18.** Fix the docs. Both the `query-daily-results` SKILL.md and the
      `run_sql` docstring (`daily/mcp_server/server.py:247-250`) claim
      "`analysis_comparisons` — per-series verdicts vs. baseline, already
      computed at ingest time". This is **not true** for the current DB (0
      rows). Replace with a warning until T2 lands.
      - The SKILL.md line lives in the **`openvino-gpu-plugin-skills` repo**,
        not here: `.github/skills/query-daily-results/SKILL.md:40`. That path
        is the source of truth — `.claude/skills/**` are generated shims, so
        edit `.github/skills/` and then run
        `.github/scripts/sync-claude-skills.sh`.
