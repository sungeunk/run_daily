from __future__ import annotations

import argparse

import pytest

import generate_fleet_report
from generate_fleet_report import (
    _apply_cli_overrides,
    _delivery_key,
    _digest_arguments,
    load_config,
    _parse_ov_version,
)
from report.fleet import render_fleet_html

pytestmark = pytest.mark.dev_only


def test_render_fleet_html_is_compact_and_links_failed_run() -> None:
    digest = {
        "selection": {
            "ov_build": "23107",
            "purpose": "daily_pipeline timer",
            "triggered_by": "timer",
        },
        "summary": {
            "status": "red",
            "expected_machines": 2,
            "completed_machines": 2,
            "failed_machines": 1,
        },
        "machines": [
            {
                "machine": "LNL-03",
                "run_id": "run good",
                "status": "success",
                "ts": "2026-09-09T23:58:00",
                "duration_sec": 125.0,
                "series_total": 7,
                "series_skipped": 0,
                "series_success": 7,
                "series_failed": 0,
                "report_file": "daily.20260909_2358.summary.json",
                "ov_version": "2026.5",
                "passed_tests": 35,
                "performance": {},
            },
            {
                "machine": "ARLH-01",
                "run_id": "run&bad",
                "status": "failed",
                "ts": "2026-09-09T23:43:00",
                "duration_sec": 2466.0,
                "series_total": 5,
                "series_skipped": 1,
                "series_success": 3,
                "series_failed": 1,
                "report_file": "daily.20260909_2343.summary.json",
                "html_report_url": "https://reports.local/ARLH-01/latest.html",
                "ov_version": "2026.4",
                "passed_tests": 33,
                "failed_tests": 1,
                "skipped_tests": 1,
                "performance": {"regressed": 2},
            },
        ],
        "functional_issues": [{
            "machine": "ARLH-01",
            "run_id": "run&bad",
            "outcome": "failed",
            "nodeid": "test_model[x]",
            "message": "bad <output>\n" + "x" * 300,
            "model": "gemma-2-9b-it",
            "precision": "OV_FP16-4BIT_DEFAULT",
            "last_good_run_id": "last-good",
            "last_good_html_report_url": "https://reports.local/ARLH-01/last-good.html",
        }],
        "top_regressions": [{
            "machine": "ARLH-01",
            "run_id": "run&bad",
            "model": "llama",
            "precision": "INT4",
            "in_token": 32,
            "out_token": 128,
            "exec_mode": "2nd",
            "improvement_pct": -0.125,
            "baseline_value": 100.0,
            "current_value": 112.5,
            "unit": "ms",
            "html_report_url": "https://reports.local/ARLH-01/latest.html",
        }],
        "warnings": ["Machines used different OpenVINO versions."],
    }

    rendered = render_fleet_html(
        digest,
        "http://viewer.local",
        "http://reports.local",
    )

    assert "Completed: <strong>2/2</strong>" in rendered
    assert 'href="http://viewer.local"' in rendered
    assert ">Daily Viewer</a>" in rendered
    assert "🟢" in rendered and "🔴" in rendered
    assert "Total: 5 / Skip: 1 / Success: 3 / Failed: 1" in rendered
    assert "41m" in rendered
    assert "https://reports.local/ARLH-01/latest.html" in rendered
    assert "https://reports.local/ARLH-01/last-good.html" in rendered
    assert "View details" not in rendered
    assert "Regression</th>" in rendered
    assert ">2</td>" in rendered
    assert "in: 32 / out: 128" in rendered
    assert "Second token latency" in rendered
    assert "12.5% regression" in rendered
    assert "100.000 ms" in rendered
    assert "112.500 ms" in rendered
    assert "Performance Regressions" in rendered
    assert "Top Performance Regressions" not in rendered
    assert "id=\"performance-regressions\"" in rendered
    assert "function sortFleetTable(header)" in rendered
    assert "data-sort-type=\"number\"" in rendered
    assert 'class="num"' in rendered
    assert ".num{text-align:right;font-variant-numeric:tabular-nums}" in rendered
    assert "<th>Models</th>" in rendered
    assert "<th>Attention reason</th>" in rendered
    assert "<th>Last good</th>" in rendered
    assert "gemma-2-9b-it / OV_FP16-4BIT_DEFAULT" in rendered
    assert "Test assertion failed" in rendered
    assert ">failed<" not in rendered
    assert "test_model[x]" not in rendered
    assert "bad &lt;output&gt;" not in rendered


def test_render_fleet_html_omits_untrusted_report_urls() -> None:
    digest = {
        "summary": {},
        "selection": {},
        "machines": [{
            "machine": "LNL-03", "run_id": "run-1", "status": "success",
            "html_report_url": "javascript:alert(1)",
        }],
    }

    rendered = render_fleet_html(digest, "http://viewer.local")

    assert "View details" not in rendered
    assert "javascript:alert" not in rendered
    assert ">HTML report</a>" not in rendered


def test_render_fleet_html_supports_pr_title_and_raw_logs() -> None:
    digest = {
        "selection": {
            "ov_build": "23164",
            "purpose": "PR#38312 onednn 3.14 sungeunk",
            "triggered_by": "sungeunk",
        },
        "summary": {
            "status": "yellow",
            "expected_machines": 1,
            "completed_machines": 1,
            "failed_machines": 0,
        },
        "machines": [{
            "machine": "PTLH-02",
            "run_id": "run-pr",
            "status": "success",
            "rawlog_path": "/var/www/html/daily2/PTLH-02/2026.09/daily.20260922_1540.raw",
            "report_file": "daily.20260922_1540.summary.json",
        }],
        "functional_issues": [{
            "machine": "PTLH-02",
            "run_id": "run-pr",
            "outcome": "failed",
            "rawlog_path": "/var/www/html/daily2/PTLH-02/2026.09/daily.20260922_1540.raw",
        }],
        "top_regressions": [{
            "machine": "PTLH-02",
            "run_id": "run-pr",
            "model": "gemma-3-4b-it",
            "precision": "OV_FP16-4BIT_DEFAULT",
            "in_token": 1024,
            "out_token": 256,
            "exec_mode": "1st",
            "improvement_pct": -0.08,
            "baseline_value": 180.0,
            "current_value": 194.0,
            "unit": "ms",
        }],
    }

    rendered = render_fleet_html(
        digest,
        "http://viewer.local",
        "http://reports.local",
        title="PR GPU Fleet Review",
    )

    assert "PR GPU Fleet Review" in rendered
    assert "Daily GPU Fleet Summary" not in rendered
    assert "PR#38312 onednn 3.14 sungeunk" in rendered
    assert "Artifacts</th>" in rendered
    assert "Raw log</a>" in rendered
    assert "http://reports.local/daily2/PTLH-02/2026.09/daily.20260922_1540.raw" in rendered


def test_render_fleet_html_appends_to_existing_viewer_query() -> None:
    digest = {
        "summary": {},
        "selection": {},
        "machines": [{"machine": "LNL-03", "run_id": "run-1", "status": "failed"}],
    }
    rendered = render_fleet_html(digest, "http://viewer.local/?page=fleet")
    assert "run-1" not in rendered


def test_load_config_and_digest_arguments(tmp_path) -> None:
        config_path = tmp_path / "fleet.json"
        config_path.write_text("""{
            "mcp_url": "http://mcp.local",
            "viewer_base_url": "http://viewer.local",
            "purpose": "daily_pipeline timer",
            "triggered_by": "timer",
            "expected_machines": ["LNL-03", "LNL-03", "MTL-01"],
            "mail": {"recipients": ["test@example.com"]}
        }""", encoding="utf-8")

        config = load_config(config_path)
        arguments = _digest_arguments(config, "23107")

        assert config.expected_machines == ("LNL-03", "MTL-01")
        assert arguments["ov_build"] == "23107"
        assert arguments["top_regressions"] == 10
        assert "report_date" not in arguments
        assert "day_start_hour" not in arguments
        assert arguments["purpose"] == "daily_pipeline timer"
        assert arguments["triggered_by"] == "timer"
        assert config.output_dir == (tmp_path / "../output/fleet").resolve()


def test_ov_version_extracts_build_and_sha_and_ignores_suffix() -> None:
        assert _parse_ov_version("2026.5.0-23164-749d332ac8b") == (
            "23164", "749d332ac8b"
        )
        assert _parse_ov_version("2026.5.0-23164-749d332ac8b-PR38312") == (
            "23164", "749d332ac8b"
        )


def test_ov_version_rejects_invalid_format() -> None:
        with pytest.raises(ValueError, match="--ov-ver"):
            _parse_ov_version("23164-749d332ac8b")


def test_sha_override_is_forwarded_to_digest_arguments(tmp_path) -> None:
        config_path = tmp_path / "fleet.json"
        config_path.write_text("""{
            "mcp_url": "http://mcp.local",
            "viewer_base_url": "http://viewer.local",
            "purpose": "PR#38312 onednn 3.14 sungeunk",
            "triggered_by": "sungeunk",
            "expected_machines": ["PTLH-02"]
        }""", encoding="utf-8")

        config = _apply_cli_overrides(
            load_config(config_path),
            argparse.Namespace(
                purpose=None,
                triggered_by=None,
                ov_sha="749d332ac8b",
                machines=None,
                exclude_machine=[],
                title=None,
                output_prefix=None,
            ),
        )

        arguments = _digest_arguments(config, "23164")

        assert config.ov_sha == "749d332ac8b"
        assert arguments["ov_sha"] == "749d332ac8b"


def test_cli_overrides_build_pr_review_scope(tmp_path) -> None:
        config_path = tmp_path / "fleet.json"
        config_path.write_text("""{
            "mcp_url": "http://mcp.local",
            "viewer_base_url": "http://viewer.local",
            "purpose": "daily_pipeline timer",
            "triggered_by": "timer",
            "expected_machines": ["PTLH-01", "PTLH-02", "LNL-03"]
        }""", encoding="utf-8")

        config = _apply_cli_overrides(
            load_config(config_path),
            argparse.Namespace(
                purpose="PR#38312 onednn 3.14 sungeunk",
                triggered_by="sungeunk",
                machines=None,
                exclude_machine=["PTLH-01"],
                title="PR GPU Fleet Review",
                output_prefix="pr-fleet",
            ),
        )

        assert config.purpose == "PR#38312 onednn 3.14 sungeunk"
        assert config.triggered_by == "sungeunk"
        assert config.expected_machines == ("PTLH-02", "LNL-03")
        assert config.report_title == "PR GPU Fleet Review"
        assert config.output_prefix == "pr-fleet"


def _config(tmp_path):
    path = tmp_path / "fleet.json"
    path.write_text("""{
        "mcp_url": "http://mcp.local",
        "viewer_base_url": "http://viewer.local",
        "purpose": "daily_pipeline timer",
        "triggered_by": "timer",
        "expected_machines": ["LNL-03", "MTL-01"],
        "mail": {"recipients": ["test@example.com"]}
    }""", encoding="utf-8")
    return load_config(path)


def test_delivery_key_distinguishes_a_retested_build(tmp_path) -> None:
    """A build that stays current for a second night is re-tested, and that
    is a new report. Keying on the build alone would swallow it."""
    config = _config(tmp_path)
    first = {"machines": [{"run_id": "run-a"}, {"run_id": "run-b"}]}
    retest = {"machines": [{"run_id": "run-c"}, {"run_id": "run-d"}]}

    assert _delivery_key(config, "23107", first) != _delivery_key(config, "23107", retest)


def test_delivery_key_is_stable_for_the_same_runs(tmp_path) -> None:
    config = _config(tmp_path)
    one = {"machines": [{"run_id": "run-a"}, {"run_id": "run-b"}]}
    same_reordered = {"machines": [{"run_id": "run-b"}, {"run_id": "run-a"}]}

    assert _delivery_key(config, "23107", one) == _delivery_key(config, "23107", same_reordered)


def test_load_config_allows_mail_free_dry_run_config(tmp_path) -> None:
        config_path = tmp_path / "fleet.json"
        config_path.write_text("""{
            "mcp_url": "http://mcp.local",
            "viewer_base_url": "http://viewer.local",
            "purpose": "daily_pipeline timer",
            "triggered_by": "scheduler",
            "expected_machines": ["LNL-03"]
        }""", encoding="utf-8")
        assert load_config(config_path).recipients == ()


def test_null_regression_limit_requests_all_rows(tmp_path) -> None:
    config_path = tmp_path / "fleet.json"
    config_path.write_text("""{
        "mcp_url": "http://mcp.local",
        "viewer_base_url": "http://viewer.local",
        "purpose": "daily_pipeline timer",
        "triggered_by": "timer",
        "expected_machines": ["LNL-03"],
        "report": {"top_regressions": null}
    }""", encoding="utf-8")

    assert _digest_arguments(load_config(config_path), "23107")["top_regressions"] == 500


def test_dry_run_reuses_one_mcp_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "fleet.json"
    config_path.write_text("""{
        "mcp_url": "http://mcp.local",
        "viewer_base_url": "http://viewer.local",
        "purpose": "daily_pipeline timer",
        "triggered_by": "timer",
        "expected_machines": ["LNL-03"]
    }""", encoding="utf-8")
    calls: list[str] = []

    class Client:
        instances = 0

        def __init__(self, *_args, **_kwargs) -> None:
            Client.instances += 1

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def call_json_tool(self, name: str, _arguments: dict):
            calls.append(name)
            if name == "daily_results_list_builds":
                return [{"ov_build": "23107"}]
            return {
                "schema_version": 1,
                "summary": {"status": "green"},
                "selection": {"ov_build": "23107"},
                "machines": [],
            }

    monkeypatch.setattr(generate_fleet_report, "McpHttpClient", Client)
    monkeypatch.setattr(
        generate_fleet_report, "_parse_args",
        lambda: argparse.Namespace(config=config_path, build=None, dry_run=True, force=False),
    )

    assert generate_fleet_report.main() == 0
    assert Client.instances == 1
    assert calls == ["daily_results_list_builds", "daily_results_daily_digest"]