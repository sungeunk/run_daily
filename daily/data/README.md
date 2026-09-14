# `daily/data` — the one place that knows what the numbers mean

Everything that reads or writes daily benchmark results goes through this
package: the pytest suite, the Streamlit viewer, the analysis engine, the
MCP server, the fleet report.

## Why it exists

Not tidiness. The same semantics had been written out in several places and
the copies drifted, and the drift produced wrong numbers on the dashboard
that nobody could see were wrong.

The clearest case: commit `a7852ea` started storing an infer-only latency
series (`1st-infer` / `2nd-infer`) alongside the token series. It added
`exclude_infer_sql()` for exactly this and applied it in **one** of the five
places that count rows in `perf`. The other four silently began counting the
new family:

| | before | after `a7852ea` | correct |
|---|---|---|---|
| viewer "Success count", PTLH-01 | 85 | **165** | 85 |
| fleet report `Failed:`, ARLH-01 | 4 | **0** | 4 |

The second row is the damaging one. `series_failed` is
`max(0, expected - skipped - success)`, so once `success` overflowed
`expected` the clamp rendered every genuinely missing series as zero. Four
models had stopped running on ARLH-01 and the report said the fleet was
clean.

An audit found the same shape everywhere: the lower-is-better rule in six
places, seconds-to-milliseconds in two (with *different* conditions), geomean
in four, run filters in five variants plus a hand-written sixth, the series
key tuple in twenty-five. Two different quantities were both called `cv` and
compared against the same threshold — they differ by a factor of 1.4826.

**A shared helper nobody is obliged to call is what we already had.** So the
extraction comes with structural guards (below).

## Layout

```
daily/data/
  series.py    exec_mode families, SeriesKey, direction, units, token buckets
  filters.py   RunScope — which runs a query may see; run identity
  stats.py     geomean, median/MAD, robust z, the two CVs
  counts.py    expected_series <-> success: both halves of the contract
  verdict.py   improved / same / regressed classification
  validate.py  invariants a stored run must satisfy
  read.py      every query (was viewer/queries.py)
  write.py     mutations; ingest entry point
  ingest/      loader, record, writer, CLI
  schema.sql   tables and views
```

Dependencies run one way. `data` imports nothing from `viewer`, `analysis`,
`report` or `mcp_server`; a test enforces it.

## The rules worth knowing

**Two exec_mode families.** `1st`/`2nd` are end-to-end token latency — the
daily metric and the only family carrying a verdict. `1st-infer`/`2nd-infer`
are the infer-only slice, stored so a GPU kernel regression can be told apart
from pipeline overhead. On VLMs infer can understate TTFT by an order of
magnitude, so never quote it as the model's latency. Match it with
`is_infer_exec_mode` / `exclude_infer_sql`, never with a literal.

**Counting is a contract with two halves.** The pytest cases declare
`expected_series`; the queries count what landed. Both live in `counts.py`
so that adding a family forces you past both. Infer is excluded by default;
including it must be asked for explicitly.

**Two CVs, one name.** `mad_ratio` is `MAD/median` — what
`trend_regressions` reports in its `recent_cv` column. `robust_cv` is
`1.4826 × MAD/|median|` — what the analysis engine's fluctuation guard uses
and what `noisy_cv_threshold` is calibrated against. They are not
interchangeable in a threshold comparison.

**Scope is opt-out.** `RunScope` defaults to daily runs, nothing excluded,
nothing partial. A caller that wants something looser has to say so — the
inverse of the old helpers, where forgetting to filter was the easy path.

**Validation records, never rejects.** A nightly run that already cost an
hour of machine time has to land even when its bookkeeping is odd, and a run
that cannot be stored cannot be investigated. Findings go to
`run_validations` and surface in the digest and the report.

## The guards

`daily/tests/test_data_layering.py` fails the build when a new copy appears:

- the lower-is-better unit set is spelled out outside `data`
- an infer name is matched by hand instead of via the helpers
- `count(*) FROM perf` appears outside the counting contract
- the seconds-to-milliseconds conversion is written again
- anything but the MCP server opens a DuckDB connection directly
- `data` imports one of its own consumers

They are regex checks over the tree, which is crude, but they catch the exact
mistake that cost this codebase four simultaneous bugs.

## Known divergences

Deliberate, documented, not yet resolved:

- **Units.** `read.series_values_for_run` serves raw `value`/`unit` while
  `perf_flat` serves `viewer_value`/`viewer_unit`, so the three
  image-generation models that report seconds read as seconds in the analysis
  and as milliseconds on the dashboard. Verdicts are unaffected (current and
  baseline both come through the same function, so the ratio is identical);
  flipping it changes stored absolute values, so it is its own decision.
- **The view and the module.** `schema.sql`'s `perf_with_buckets` spells the
  unit rule out in SQL because a view cannot import Python. One copy remains,
  inside the layer that owns the rule.
- **Tie order.** `trend_regressions` sorts with pandas' default quicksort, so
  rows with equal `worsening_pct` come back in a different order run to run.
  The row set is identical; only the display order moves.
