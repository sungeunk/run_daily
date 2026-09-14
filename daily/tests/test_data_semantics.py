"""Contract tests for daily/data — the semantics every layer shares.

These pin the decisions that were previously spelled out in several places
and drifted apart. Where a decision was made by looking at the production DB,
the observation is recorded in the test so a future change has to argue with
the evidence rather than with a preference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

pytestmark = pytest.mark.dev_only

DAILY_DIR = Path(__file__).resolve().parent.parent
if str(DAILY_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_DIR))

from data import (  # noqa: E402
    MAD_TO_SIGMA, RunScope, SeriesKey, count_success_series,
    counts_are_consistent, direction_sign, exclude_infer_sql, expected_cases,
    expected_series_for_app, expected_series_for_image_gen,
    expected_series_for_llm, failed_series, geomean, infer_twin,
    is_infer_exec_mode, lower_is_better, mad_ratio, normalize_unit,
    normalize_value, robust_cv, robust_z, series_key_sql, token_bucket,
)


class TestSeriesFamilies:
    def test_infer_family_is_matched_by_suffix_not_by_list(self):
        # exclude_infer_sql is written as a suffix match so a future
        # 'pipeline-infer' is excluded without a second edit.
        assert is_infer_exec_mode("1st-infer")
        assert is_infer_exec_mode("pipeline-infer")
        assert not is_infer_exec_mode("1st")
        assert not is_infer_exec_mode(None)

    def test_infer_twin_only_exists_for_token_families(self):
        assert infer_twin("1st") == "1st-infer"
        assert infer_twin("2nd") == "2nd-infer"
        assert infer_twin("1st-infer") is None

    def test_exclusion_sql_filters_the_infer_family(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE perf (exec_mode VARCHAR)")
        con.execute("INSERT INTO perf VALUES ('1st'), ('2nd'), "
                    "('1st-infer'), ('2nd-infer')")
        kept = [r[0] for r in con.execute(
            f"SELECT exec_mode FROM perf WHERE {exclude_infer_sql()} "
            "ORDER BY exec_mode").fetchall()]
        assert kept == ["1st", "2nd"]


class TestSeriesKey:
    def test_sql_column_order_matches_the_dataclass(self):
        assert series_key_sql() == \
            "model, precision, in_token, out_token, exec_mode"
        assert series_key_sql("p").startswith("p.model")

    def test_round_trips_through_a_mapping_and_a_bare_tuple(self):
        mapping = {"model": "llama", "precision": "INT4", "in_token": 32,
                   "out_token": 128, "exec_mode": "1st"}
        from_map = SeriesKey.from_row(mapping)
        from_tuple = SeriesKey.from_row(("llama", "INT4", 32, 128, "1st"))
        assert from_map == from_tuple
        assert from_map.as_tuple() == ("llama", "INT4", 32, 128, "1st")

    def test_pairing_a_token_series_with_its_infer_twin_keeps_the_shape(self):
        # The pairing requires an exact token-shape match; nothing else moves.
        token = SeriesKey("qwen3-vl-4b-instruct", "INT4", 4907, 256, "1st")
        twin = token.with_exec_mode("1st-infer")
        assert twin.is_infer
        assert (twin.model, twin.in_token, twin.out_token) == \
               (token.model, token.in_token, token.out_token)


class TestDirection:
    @pytest.mark.parametrize("unit", ["ms", "s", "%"])
    def test_latency_units_are_lower_is_better(self, unit):
        assert lower_is_better(unit)
        assert direction_sign(unit) == 1

    @pytest.mark.parametrize("unit", ["FPS", "tps", None])
    def test_throughput_units_invert_the_sign(self, unit):
        assert not lower_is_better(unit)
        assert direction_sign(unit) == -1


class TestUnitNormalisation:
    def test_seconds_become_milliseconds(self):
        assert normalize_value(2.5, "s") == 2500.0
        assert normalize_unit("s") == "ms"

    def test_other_units_pass_through_untouched(self):
        assert normalize_value(120.0, "ms") == 120.0
        assert normalize_unit("FPS") == "FPS"

    def test_rule_is_unconditional_not_gated_on_a_model_list(self):
        """schema.sql gated the conversion on three image-gen model names.
        Checked against the production DB: every unit='s' row belongs to a
        model on that list, so the rules agree on all existing data. The
        unconditional one is kept because it also covers the next
        image-generation model without an edit."""
        assert normalize_value(1.0, "s") == 1000.0  # no model argument at all


class TestTokenBucket:
    def test_boundary_is_closed_below_and_open_at_the_threshold(self):
        assert token_bucket(99) == "short"
        assert token_bucket(100) == "long"

    def test_zero_input_is_its_own_bucket(self):
        assert token_bucket(0) == "0"
        assert token_bucket(None) == "0"

    def test_large_inputs_bucket_rather_than_vanish(self):
        """The legacy <400 / 401-1200 banding dropped 10.4% of token rows —
        everything above 1200, plus exactly 400. That banding is scoped to
        legacy_geomean_summary; this one must place every input somewhere."""
        assert token_bucket(400) == "long"
        assert token_bucket(4907) == "long"


class TestStats:
    def test_geomean_ignores_non_positive_sentinels(self):
        # llm_bench reports -1 / 0 for "not measured"; those must not poison
        # the log or drag the mean toward zero.
        assert geomean([1.0, 4.0]) == pytest.approx(2.0)
        assert geomean([1.0, 4.0, -1.0, 0.0]) == pytest.approx(2.0)
        assert geomean([]) is None
        assert geomean([-1.0]) is None

    def test_mad_ratio_and_robust_cv_differ_by_the_scale_factor(self):
        """Both were called 'cv' and compared against the same thresholds.
        trend_regressions means mad_ratio; the analysis engine means
        robust_cv. They are not interchangeable."""
        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        assert robust_cv(values) == pytest.approx(
            mad_ratio(values) * MAD_TO_SIGMA)
        assert mad_ratio(values) != pytest.approx(robust_cv(values))

    def test_robust_z_is_positive_for_worse_whichever_way_the_unit_points(self):
        history = [100.0, 101.0, 99.0, 100.0, 102.0]
        assert robust_z(130.0, history, "ms") > 0    # slower latency is worse
        assert robust_z(70.0, history, "ms") < 0
        assert robust_z(70.0, history, "FPS") > 0    # lower throughput is worse
        assert robust_z(130.0, history, "FPS") < 0

    def test_zero_spread_yields_no_score_rather_than_infinity(self):
        assert robust_z(120.0, [100.0] * 5, "ms") is None


class TestCounts:
    def test_llm_declares_two_series_per_prompt(self, tmp_path):
        prompts = tmp_path / "p.jsonl"
        prompts.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
        assert expected_series_for_llm(prompts) == 4

    def test_a_missing_prompt_file_expects_nothing(self, tmp_path):
        assert expected_series_for_llm(tmp_path / "absent.jsonl") == 0

    def test_image_gen_counts_prompts_unless_one_is_pinned(self, tmp_path):
        prompts = tmp_path / "p.jsonl"
        prompts.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")
        assert expected_series_for_image_gen(prompts) == 3
        assert expected_series_for_image_gen(prompts, prompt_index=1) == 1

    def test_app_declares_one(self):
        assert expected_series_for_app() == 1

    def test_expected_cases_sums_series_not_test_functions(self):
        summary = {"tests": [
            {"outcome": "passed", "metrics": {"expected_series": 8}},
            {"outcome": "skipped", "metrics": {"expected_series": 4}},
            {"outcome": "failed", "metrics": {"expected_series": 2}},
        ]}
        assert expected_cases(summary) == 14
        assert expected_cases(summary, {"skipped"}) == 4


class TestSuccessCounting:
    @pytest.fixture()
    def con(self):
        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE perf (run_id VARCHAR, model VARCHAR, "
            "precision VARCHAR, in_token INTEGER, out_token INTEGER, "
            "exec_mode VARCHAR, prompt_idx INTEGER)")
        con.execute(
            "INSERT INTO perf VALUES "
            "('r1','llama','INT4',32,128,'1st',0),"
            "('r1','llama','INT4',32,128,'2nd',0),"
            "('r1','llama','INT4',32,128,'1st-infer',0),"
            "('r1','llama','INT4',32,128,'2nd-infer',0)")
        yield con
        con.close()

    def test_infer_is_excluded_by_default(self, con):
        # The regression this module exists for: 2 token series read as 4.
        assert count_success_series(con, ["r1"]) == {"r1": 2}

    def test_including_infer_has_to_be_asked_for(self, con):
        assert count_success_series(con, ["r1"], include_infer=True) == {"r1": 4}

    def test_no_run_ids_is_not_a_query(self, con):
        assert count_success_series(con, []) == {}

    def test_runs_that_produced_nothing_are_absent_not_zero(self, con):
        assert count_success_series(con, ["r1", "ghost"]) == {"r1": 2}


class TestCountConsistency:
    def test_the_overflow_that_the_clamp_used_to_hide(self):
        # 85 expected, 165 counted — what the fleet report showed before the
        # infer family was excluded. failed_series clamps to 0, so the
        # inconsistency has to be reported separately.
        assert failed_series(85, 0, 165) == 0
        assert not counts_are_consistent(85, 0, 165)

    def test_a_genuinely_missing_series_still_surfaces(self):
        assert failed_series(85, 4, 77) == 4
        assert counts_are_consistent(85, 4, 77)


class TestRunScope:
    @pytest.fixture()
    def db(self, tmp_path):
        path = tmp_path / "scope.duckdb"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE runs (run_id VARCHAR, purpose VARCHAR, "
                    "description VARCHAR, run_kind VARCHAR)")
        con.execute("CREATE TABLE run_exclusions (run_id VARCHAR)")
        con.execute("CREATE VIEW runs_with_flags AS SELECT r.*, FALSE AS is_partial, "
                    "EXISTS (SELECT 1 FROM run_exclusions e WHERE e.run_id = r.run_id) "
                    "AS excluded FROM runs r")
        con.close()
        return path

    def test_flag_relation_uses_the_derived_columns(self, db):
        clause, _ = RunScope(db, relation="runs_with_flags").where()
        assert "NOT r.excluded" in clause
        assert "NOT r.is_partial" in clause
        assert "NOT EXISTS" not in clause

    def test_raw_table_falls_back_to_the_exclusion_list(self, db):
        clause, _ = RunScope(db, relation="runs").where()
        assert "NOT EXISTS (SELECT 1 FROM run_exclusions" in clause
        # The raw table has no is_partial to filter on.
        assert "is_partial" not in clause

    def test_defaults_are_the_cohort_semantics(self, db):
        clause, params = RunScope(db).where()
        assert "run_kind" in clause and params == ["daily"]

    def test_opting_out_widens_the_scope(self, db):
        clause, params = RunScope(
            db, run_kinds=None, include_excluded=True, include_partial=True,
        ).where()
        assert clause == "" and params == []

    def test_free_text_kinds_match_purpose_and_description(self, db):
        clause, params = RunScope(db, run_kinds=("pr38062",)).where()
        assert "purpose" in clause and "description" in clause
        assert params == ["%pr38062%", "%pr38062%"]

    def test_missing_columns_degrade_instead_of_failing(self, tmp_path):
        # A DB from before the flags existed must still render.
        path = tmp_path / "old.duckdb"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE runs (run_id VARCHAR)")
        con.execute("CREATE VIEW runs_with_flags AS SELECT * FROM runs")
        con.close()
        clause, params = RunScope(path, relation="runs_with_flags").where()
        assert clause == "" and params == []
