# run_daily TODO

Improvement backlog for the daily benchmark pipeline, DuckDB store, and the
`daily_results` MCP server.

All findings below were verified against the central DB
(`/var/www/html/daily2/daily_llm_benchmark.duckdb`) and against the working
tree at `9c18c84` on 2026-09-07 unless the item is explicitly marked
*(assumption)*.

Dependency order:

```
T1 · T3  →  T4 · T5  →  T2  →  T6  →  T7 · T8  →  T9–T14 · T17 · T18
  DONE      DONE(-ish)         mostly              T15 · T16 · T19 (independent)
```

**Status (2026-09-07):** T1, T3 and the query-layer half of T6 have landed —
101 dev_only tests pass, and the new schema was validated against a copy of
the central DB. Next up is T4/T5 (declared `run_kind` + `trigger`), then T2's
backfill, then T6's remaining MCP-side plumbing.

`T2`'s backfill sits after `T4`/`T5` on purpose: backfilling 239 runs before
the baseline cohort is redefined (`run_kind × trigger × partial`) means doing
it twice. `T18` blocks on `T2` (it can only stop warning once the tables are
populated); `T17` blocks on `T12` (it needs the health columns exposed).

---

## P0 — Data integrity (prerequisite for everything else)

### T1. Remove `short_run`, replace it with model filtering — **DONE**

**Why:** `short_run=true` changed `out_token 256 → 32` and `benchmark_iter_num
3 → 1`, so the resulting numbers were not comparable with a full run — yet
`short_run` was part of the baseline selection key, which turned it into a
contamination path.

Measured: **0 runs** had `short_run = TRUE`, so removal was low risk.

- [x] Dropped the CLI flag from `daily/run.py` and `daily/conftest.py`,
      including the metadata producer and the module-docstring example
- [x] Replaced the branching in `daily/common/config.py` with the
      `OUT_TOKEN_LENGTH` / `BENCHMARK_ITER_NUM` module constants — the
      measurement shape is now not configurable at all
- [x] Dropped it from the baseline/comparison keys in
      `daily/analysis/{baseline,engine,remote}.py` and
      `scripts/generate_analysis_report.py`, which now read `is_partial` from
      `runs_with_flags`
- [x] Dropped from schema and ingest; added `runs.test_filter` and
      `runs.selected_cases` with `writer.py` migrations so existing DBs pick
      them up
- [x] **Dropped the `runs.short_run` column from the live DBs**, not just
      from `schema.sql` — a dead all-FALSE column means a stale query
      filtering on it silently returns everything instead of failing. Applied
      to the central DB (239 runs), `daily/LNL-03.duckdb` (24) and
      `daily/viewer/bench.duckdb` (0); every value was FALSE, so no
      information was lost. Backup at
      `/var/www/html/daily2/daily_llm_benchmark.duckdb.bak.pre-short_run-drop`.
      `writer._drop_short_run_column()` makes this self-healing for any other
      DB on its next ingest. DuckDB refuses `DROP COLUMN` while *any* view or
      index references the table, so the step tears both down and lets
      `schema.sql` rebuild them — it is guarded on the column still existing
      so the `runs` indexes are not rebuilt on every ingest.
- [x] Added the `runs_with_flags` view (`daily/viewer/schema.sql`) as the one
      definition of `is_partial` / `excluded`, so the analysis layer, the
      viewer queries and `perf_flat` cannot drift apart
- [x] Added `TestPartialRuns` to `daily/tests/test_viewer_queries.py`
- [x] Updated `DESIGN.md` and `DESIGN_REPORT.md`

Two corrections to what this item originally claimed:

1. `test_viewer_queries.py::TestCohort::test_short_runs_can_be_excluded`
   **did not exist** — no test in `daily/tests/` referenced `short_run` at
   all. New coverage was written from scratch instead.
2. Deriving `partial` from `selected_cases < expected_cases` **cannot work**:
   `_cases(summary)` sums over `summary["tests"]`, which only lists tests
   pytest *collected*, so a `-k llama` run reports the same figure for both.
   `runs.test_filter` is therefore the authoritative signal and
   `selected_cases` is informational. Getting a real suite size needs a
   collect-only pass — see T19.
   Relatedly, `selected_cases` deliberately does **not** subtract
   `skipped_cases`: measured, 117 of 208 daily runs skip at least one case,
   so treating a skip as a narrowing would mark most of the fleet partial and
   empty the baseline cohort.

### T19. Record the true suite size, so `is_partial` covers more than `-k`

**Why:** T1 landed `is_partial`, but it only fires when `test_filter` is set.
A run narrowed by `--tests`, or one where collection itself dropped cases,
still looks full. Historical rows (all 239) have no `test_filter` and cannot
be reclassified.

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

### T3. Cut off baseline contamination sources — **DONE**

- [x] `perf_flat` now selects from `runs_with_flags` and exposes `excluded`
      as a **column** rather than filtering it away, because the Excel tab's
      manual run picker must still be able to select an excluded run
      (`perf_for_runs` takes explicit run_ids and is the picker's path).
      `perf_stats` filters `WHERE NOT excluded`, so the rolling median no
      longer includes excluded runs — measured, 107 of 16014 perf rows drop.
      Cohort queries opt in via the new `_perf_flat_exclusion_clause()`.
- [x] `run_exclusions.reason` is now mandatory — enforced in
      `add_exclusion()` rather than as a `NOT NULL` column, because the DDL
      is `IF NOT EXISTS` and would not migrate the 9 existing NULL rows.
- [x] Added `model_cache`, `genai_version`, `genai_commit`,
      `gpu_driver_version` to `perf_flat`, plus `description` (the Run-kinds
      selector matches free text against `purpose`/`description`, so
      filtering `perf_flat` by keyword needs both).
- [x] `latest_run_per_machine` now requires
      `run_kind = 'daily' AND NOT excluded AND NOT is_partial`. Verified
      against a copy of the central DB: still 9 machines, no regression.

Correction to what this item originally claimed: `_exclusion_predicate()`
already existed (`daily/viewer/queries.py:184`) and *was* applied by 6 call
sites, so "not applied by any view" was true only of the SQL views. The real
gap was exactly `series_history` + `trend_regressions` — the same pair as T6,
fixed there.

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
- [ ] Final classification = `run_kind` × `trigger` × `partial` (the last one
      from T1's `test_filter` / `selected_cases`)
- [ ] Define the default baseline cohort as
      `run_kind = 'daily' AND trigger = 'timer' AND NOT partial`

### T6. Fix `series_history` and `trend_regressions` (the two unguarded queries)

**Why:** `queries.py` already has a `run_kinds` parameter and
`DEFAULT_RUN_KINDS = ("daily",)` (`daily/viewer/queries.py:144`), and
`_exclusion_predicate()` at `:184`. Both are threaded through the rest of the
module — but `series_history` and `trend_regressions` accepted only
`purpose_filter` and applied neither guard. These are the same two functions
T3 identifies, so both fixes landed together.

- [x] Added `run_kinds` (default `("daily",)`) to both, via the new
      `_perf_flat_kind_clause()`. `_run_kind_predicate()` now accepts
      `alias=None` for a single-table select straight from `perf_flat`.
- [x] Applied the exclusion filter in both (this was T3's first bullet)
- [x] Added `TestTrendGuards` / `TestExclusionsReachTheBaseline` coverage.
      Verified against a copy of the central DB: one LNL-03 series goes from
      93 points to 80 once the daily-only filter applies.
- [ ] **Still open:** expose `run_kinds` on every MCP tool, defaulting to
      `("daily",)`. The queries accept it now, but
      `daily/mcp_server/server.py` does not pass it, so the MCP path still
      sees the unfiltered default of whatever the tool hardcodes.
      Measured contamination in the last 10 days: `run_kind='pr'` 547 perf
      rows and `run_kind='test'` 112 perf rows against `daily`'s 9393.

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
