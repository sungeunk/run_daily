from __future__ import annotations

import pytest

from generate_fleet_report import _delivery_key, _digest_arguments, load_config
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
            "message": "bad <output>",
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
        }],
        "warnings": ["Machines used different OpenVINO versions."],
    }

    rendered = render_fleet_html(
        digest,
        "http://viewer.local",
        "http://reports.local",
    )

    assert "Completed: <strong>2/2</strong>" in rendered
    assert "🟢" in rendered and "🔴" in rendered
    assert "Total: 5 / Skip: 1 / Success: 3 / Failed: 1" in rendered
    assert "41m" in rendered
    assert "http://reports.local/daily2/ARLH-01/2026.09/daily.20260909_2343.html" in rendered
    assert "Regression</th>" in rendered
    assert ">2</td>" in rendered
    assert "run%26bad" in rendered
    assert "-12.5%" in rendered
    assert "<th>Test</th>" not in rendered
    assert "<th>Failure summary</th>" not in rendered


def test_render_fleet_html_appends_to_existing_viewer_query() -> None:
    digest = {
        "summary": {},
        "selection": {},
        "machines": [{"machine": "LNL-03", "run_id": "run-1", "status": "failed"}],
    }
    rendered = render_fleet_html(digest, "http://viewer.local/?page=fleet")
    assert "?page=fleet&amp;run_id=run-1" in rendered


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
        assert "report_date" not in arguments
        assert "day_start_hour" not in arguments
        assert arguments["purpose"] == "daily_pipeline timer"
        assert arguments["triggered_by"] == "timer"
        assert config.output_dir == (tmp_path / "../output/fleet").resolve()


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