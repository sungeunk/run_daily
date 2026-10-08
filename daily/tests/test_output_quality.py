from __future__ import annotations

import io
import builtins
from pathlib import Path

import pytest

from analysis.output_quality import OutputSample, inspect_outputs, quality_for_runs, samples_from_summary, text_iou, text_warnings
from analysis.types import AnalysisConfig
from common.output_capture import generated_texts

pytestmark = pytest.mark.dev_only


def _text(text: str | None, fingerprint: str | None = "same") -> OutputSample:
    return OutputSample("model", "FP16", "0 / 1", "text", fingerprint, text=text)


def _image(blank: bool = False, invert: bool = False) -> bytes:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    if not blank:
        array[::2] = 255
    if invert:
        array = 255 - array
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_text_similarity_and_anomalies() -> None:
    assert text_iou("Hello   world", "hello world") == 1.0
    assert text_iou("abc", "xyz") == 0.0
    assert text_iou("", "") is None
    assert text_iou("a", "b") == 0.0
    assert text_warnings("!!!!!!!!!!!!")
    assert text_warnings("bad\ufffdoutput")
    assert text_warnings("repeat words " * 10)
    assert text_warnings("")
    assert not text_warnings("A normal sentence with punctuation.\nAnother line.")


def test_generated_text_capture_is_bounded_and_marks_truncation() -> None:
    output = "\n".join(
        f"[1][P{index}] Generated: " + ("x" * 100_100)
        for index in range(30)
    )

    records = generated_texts(output)

    assert len(records) <= 256
    assert all(len(record["generated_text"]) <= 100_000 for record in records)
    assert sum(len(record["generated_text"]) for record in records) <= 2_000_000
    assert any(record["truncated"] for record in records)


def test_pairwise_text_scores_and_advisory_status() -> None:
    result = inspect_outputs([_text("normal answer")], [_text("normal answer")], [_text("xyz")], AnalysisConfig())
    row = result.rows[0]
    assert row.status == "pass"
    assert row.release_warning and not row.baseline_warning
    assert row.baseline_score == 1.0 and row.release_score == 0.0
    assert any("release: IoU" in reason for reason in row.reasons)


def test_output_links_preserve_each_run_and_original_image_slot() -> None:
    summary = {"tests": [
        {"outcome": "passed", "metrics": {
            "test_type": "llm_benchmark", "model": "model", "precision": "FP16",
            "generated_outputs": [{"prompt_idx": 0, "iteration": 1, "generated_text": "normal answer"}],
        }},
        {"outcome": "passed", "metrics": {
            "test_type": "image_generation", "model": "image-model", "precision": "FP16",
            "data": [{"image_path": "model_p0_iter1_pid42_output.png"}],
        }},
    ]}
    stamps = ["20261008_1310", "20260930_2342", "20260901_2103"]
    samples = [
        samples_from_summary(
            summary, "", lambda _: None, stamp,
            artifact_base_url=f"https://reports.example/daily2/RAPTOR-ELLY/{stamp[:4]}.{stamp[4:6]}",
        )
        for stamp in stamps
    ]

    rows = inspect_outputs(*samples, AnalysisConfig()).rows

    for label, stamp in zip(("current", "baseline", "release"), stamps):
        base = f"https://reports.example/daily2/RAPTOR-ELLY/{stamp[:4]}.{stamp[4:6]}/daily.{stamp}"
        assert getattr(rows[0], f"{label}_url") == f"{base}.raw"
        assert getattr(rows[1], f"{label}_url") == f"{base}.image.image-model_FP16_0.png"


def test_truncated_generated_text_is_not_compared_as_complete() -> None:
    current = OutputSample("model", "FP16", "0 / 1", "text", "same", text="partial", text_truncated=True)
    reference = _text("partial")

    row = inspect_outputs([current], [reference], [], AnalysisConfig()).rows[0]

    assert row.status == "unavailable"
    assert row.baseline_score is None
    assert any("truncated text comparison unavailable" in item for item in row.reasons)
    assert inspect_outputs([_text("abc")], [_text("xyz")], [_text("xyz")], AnalysisConfig(output_text_iou_threshold=0)).rows[0].status == "pass"


@pytest.mark.parametrize("reference, condition, score", [
    (_text(None), "unavailable", None),
    (_text("text", None), "unverified", 1.0),
    (_text("text", "different"), "different", 1.0),
])
def test_comparison_conditions_do_not_override_output_status(
    reference: OutputSample, condition: str, score: float | None,
) -> None:
    row = inspect_outputs([_text("text")], [reference], [], AnalysisConfig()).rows[0]
    assert row.status == "pass"
    assert row.baseline_comparison == condition and row.baseline_score == score
    assert row.release_comparison == "unavailable" and row.release_score is None


@pytest.mark.parametrize("fingerprint", [None, "different"])
def test_reference_only_low_similarity_still_warns(fingerprint: str | None) -> None:
    row = inspect_outputs([_text("abc")], [_text("xyz", fingerprint)], [], AnalysisConfig()).rows[0]
    assert row.status == "pass" and row.baseline_score == 0.0 and row.baseline_warning
    assert row.baseline_comparison == ("unverified" if fingerprint is None else "different")
    assert any("reference only" in reason for reason in row.reasons)


def test_images_use_ssim_and_detect_blank_or_corrupt() -> None:
    pytest.importorskip("skimage.metrics")
    current = OutputSample("model", "FP16", "0 / 1", "image", "same", image=_image())
    changed = OutputSample("model", "FP16", "0 / 1", "image", "same", image=_image(invert=True))
    row = inspect_outputs([current], [current], [changed], AnalysisConfig()).rows[0]
    assert row.status == "pass" and row.baseline_score == 1.0 and row.release_score < 0.9
    assert row.release_warning
    assert row.current_preview.startswith("data:image/jpeg;base64,")


def test_blank_or_corrupt_images_are_detected_without_ssim_dependency() -> None:
    for content in (_image(blank=True), b"invalid image"):
        sample = OutputSample("model", "FP16", "0 / 1", "image", "same", image=content)
        assert inspect_outputs([sample], [], [], AnalysisConfig()).rows[0].status == "warning"


def test_ssim_import_error_degrades_only_the_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = OutputSample("model", "FP16", "0 / 1", "image", "same", image=_image())
    real_import = builtins.__import__

    def without_skimage(name, *args, **kwargs):
        if name == "skimage.metrics":
            raise ImportError("optional SSIM dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_skimage)
    row = inspect_outputs([sample], [sample], [], AnalysisConfig()).rows[0]

    assert row.status == "pass"
    assert row.baseline_score is None
    assert any("SSIM dependency unavailable" in reason for reason in row.reasons)
    assert row.reasons.count("baseline: SSIM dependency unavailable") == 1
    assert "baseline: SSIM comparison unavailable" not in row.reasons


def test_ssim_value_error_does_not_discard_other_output_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("skimage.metrics")
    import skimage.metrics

    monkeypatch.setattr(
        skimage.metrics, "structural_similarity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid comparison")),
    )
    sample = OutputSample("model", "FP16", "0 / 1", "image", "same", image=_image())
    row = inspect_outputs([sample], [sample], [], AnalysisConfig()).rows[0]

    assert row.status == "pass"
    assert row.baseline_score is None
    assert any("SSIM comparison unavailable" in reason for reason in row.reasons)


def test_historical_raw_is_scoped_to_test_command() -> None:
    summary = {"tests": [{"outcome": "passed", "metrics": {
        "test_type": "llm_benchmark", "model": "model", "precision": "FP16", "cmd": "python one", "data": [{"prompt_idx": 0}],
    }}]}
    raw = "[CMD] python other\n[ INFO ] [1][P0] Generated: wrong\n[CMD] python one\n[ INFO ] [warm-up][P0] Generated: correct\nsecond line"
    samples = samples_from_summary(summary, raw, lambda _: None, "20261005_2349")
    assert samples[0].text == "correct\nsecond line"
    assert samples[0].prompt == "0 / warm-up" and samples[0].fingerprint is None


def test_historical_images_have_reference_only_ssim_scores() -> None:
    pytest.importorskip("skimage.metrics")
    current = OutputSample("model", "FP16", "0 / 1", "image", None, image=_image())
    changed = OutputSample("model", "FP16", "0 / 1", "image", None, image=_image(invert=True))
    row = inspect_outputs([current], [current], [changed], AnalysisConfig()).rows[0]
    assert row.status == "pass"
    assert row.baseline_score == 1.0 and row.release_score is not None
    assert row.baseline_comparison == row.release_comparison == "unverified"
    assert not row.baseline_warning and row.release_warning


def test_missing_archive_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "analysis.output_quality._read_artifact",
        lambda *_args, **_kwargs: None,
    )
    result = quality_for_runs(AnalysisConfig(), "machine", "20261005_2349", None, None)
    assert not result.rows and "not verified" in result.detail


def test_image_slot_recovery_does_not_open_original_path() -> None:
    names: list[str] = []
    def read(name: str) -> bytes:
        names.append(name)
        return _image()
    summary = {"tests": [{"metrics": {"test_type": "image_generation", "model": "model", "precision": "FP16", "data": [
        {"image_path": "C:\\elsewhere\\model_p3_iter1_pid42_output.png"},
    ]}}]}
    samples = samples_from_summary(summary, "", read, "20261005_2349")
    assert names == ["daily.20261005_2349.image.model_FP16_0.png"]
    assert samples[0].prompt == "3 / 1"


def test_image_slots_preserve_source_indices_suffixes_and_deduplicate() -> None:
    names: list[str] = []

    def read(name: str) -> bytes:
        names.append(name)
        return _image()

    image_path = "/cache/model_p2_iter1_pid42_output.webp"
    summary = {"tests": [
        {"metrics": {
            "test_type": "image_generation", "model": "model", "precision": "FP16",
            "data": ["malformed", {"image_path": image_path}],
        }},
        {"metrics": {
            "test_type": "image_generation", "model": "model", "precision": "FP16",
            "data": [{"image_path": image_path}],
        }},
    ]}

    samples = samples_from_summary(summary, "", read, "20261005_2349")

    assert names == ["daily.20261005_2349.image.model_FP16_1.webp"]
    assert len(samples) == 1 and samples[0].prompt == "2 / 1"


def test_different_image_sizes_are_not_compared() -> None:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    current = OutputSample("model", "FP16", "0 / 1", "image", "same", image=_image())
    reference = OutputSample("model", "FP16", "0 / 1", "image", "same", image=buffer.getvalue())
    row = inspect_outputs([current], [reference], [reference], AnalysisConfig()).rows[0]
    assert row.status == "pass" and row.baseline_score is None
    assert row.baseline_comparison == "unavailable"
    assert any("dimensions differ" in reason for reason in row.reasons)


def test_empty_missing_and_oversized_text() -> None:
    for content, status in [("", "warning"), (None, "unavailable"), ("x" * 200_001, "unavailable")]:
        sample = _text(content)
        row = inspect_outputs([sample], [sample], [sample], AnalysisConfig()).rows[0]
        assert row.status == status
    assert not text_warnings("x" * 50_000)


def test_one_malformed_image_does_not_discard_other_quality_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analysis import output_quality

    original = output_quality._image_data
    calls = 0

    def decode(content: bytes):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SyntaxError("malformed image payload")
        return original(content)

    monkeypatch.setattr(output_quality, "_image_data", decode)
    result = inspect_outputs([
        OutputSample("bad", "FP16", "0 / 1", "image", image=b"bad"),
        OutputSample("good", "FP16", "0 / 1", "image", image=_image()),
    ], [], [], AnalysisConfig())

    assert [row.status for row in result.rows] == ["unavailable", "pass"]
    assert result.rows[0].reasons == ["Output inspection failed"]


def test_disabled_release_does_not_block_pass() -> None:
    sample = _text("normal text")
    result = inspect_outputs([sample], [sample], [], AnalysisConfig(release_enabled=False))
    assert result.rows[0].status == "pass"


def test_quality_findings_persist_without_preview_duplication() -> None:
    from analysis.persistence import _result_to_dict
    from analysis.types import AnalysisResult, BaselineInfo, FunctionalResult, PerformanceResult
    quality = inspect_outputs([_text("!!!!!!!!!!!!")], [], [], AnalysisConfig())
    result = AnalysisResult(
        overall_status="green", baseline=BaselineInfo(status="found"),
        functional=FunctionalResult(1, 1, 0, 0, 0), performance=PerformanceResult(0, 0, 0, 0, 0),
        models=[], top_regressions=[], rows=[], output_quality=quality,
        release_status="yellow", baseline_release_status="green",
    )
    saved = _result_to_dict(result, AnalysisConfig(output_text_iou_threshold=0.4))
    assert saved["output_quality"]["rows"][0]["status"] == "warning"
    assert "current_preview" not in saved["output_quality"]["rows"][0]
    assert saved["config_snapshot"]["output_text_iou_threshold"] == 0.4
    assert saved["overall_status"] == "green"
    assert saved["release_status"] == "yellow"
    assert saved["baseline_release_status"] == "green"


def test_local_archive_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    monkeypatch.setattr("common.delivery.REMOTE_BASE_DIR", str(tmp_path))
    config = AnalysisConfig(output_artifact_base_url="")
    directory = tmp_path / "machine" / "2026.10"
    directory.mkdir(parents=True)
    summary = {"tests": [{"metrics": {
        "test_type": "llm_benchmark", "model": "model", "precision": "FP16", "generation_fingerprint": "same",
        "generated_outputs": [{"prompt_idx": 0, "iteration": "1", "generated_text": "normal text"}],
    }}]}
    for stamp in ("20261005_2349", "20261004_2349", "20261005_1032"):
        (directory / f"daily.{stamp}.summary.json").write_text(json.dumps(summary), encoding="utf-8")
    result = quality_for_runs(config, "machine", "20261005_2349", "20261004_2349", "20261005_1032")
    assert len(result.rows) == 1 and result.rows[0].status == "pass"
    assert result.rows[0].baseline_score == 1 and result.rows[0].release_score == 1


def test_artifact_path_restrictions(tmp_path: Path) -> None:
    from analysis.output_quality import _read_artifact
    config = AnalysisConfig(output_artifact_base_url="")
    for machine, name in [("../other", "daily.20261005_2349.raw"), ("machine", "../private.raw"), ("machine", "private.raw")]:
        assert _read_artifact(config, machine, "20261005_2349", name, tmp_path) is None


def test_read_artifact_accepts_staged_webp_images(tmp_path: Path) -> None:
    from analysis.output_quality import _read_artifact

    content = _image()
    artifact = tmp_path / "daily.20261005_2349.image.model_FP16_0.webp"
    artifact.write_bytes(content)
    config = AnalysisConfig(output_artifact_base_url="")

    assert _read_artifact(
        config, "LNL-03", "20261005_2349", artifact.name, tmp_path,
    ) == content


def test_read_artifact_handles_truncated_http_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import http.client

    from analysis.output_quality import _read_artifact

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read1(self, _size: int) -> bytes:
            raise http.client.IncompleteRead(b"partial")

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("common.delivery.REMOTE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr("urllib.request.build_opener", lambda *_args, **_kwargs: Opener())
    config = AnalysisConfig(output_artifact_base_url="http://reports.example.com/daily2")

    assert _read_artifact(
        config, "LNL-03", "20261005_2349", "daily.20261005_2349.raw", None,
    ) is None


@pytest.mark.parametrize("summary", [
    {"tests": ["not-a-test-object"]},
    {"tests": [{"metrics": {
        "test_type": "llm_benchmark",
        "generated_outputs": "not-a-record-list",
        "data": ["not-a-record", {"prompt_idx": 0, "generated_text": "ok"}],
    }}]},
    {"tests": [{"metrics": {
        "test_type": "image_generation",
        "data": ["not-an-image-record"],
    }}]},
])
def test_malformed_archived_summary_records_are_ignored(summary: dict) -> None:
    samples = samples_from_summary(summary, "", lambda _: None, "20261005_2349")
    assert all(isinstance(sample, OutputSample) for sample in samples)


def test_expired_artifact_fetch_budget_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from analysis.output_quality import _read_artifact

    monkeypatch.setattr("common.delivery.REMOTE_BASE_DIR", "/tmp/nonexistent-daily-test-archive")
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_args, **_kwargs: pytest.fail("network fetch should be skipped"),
    )
    config = AnalysisConfig(output_artifact_base_url="http://reports.example.com")

    assert _read_artifact(
        config, "LNL-03", "20261005_2349", "daily.20261005_2349.raw", None,
        deadline=time.monotonic() - 1,
    ) is None


@pytest.mark.parametrize("threshold", [-1.0, 1.1, float("nan")])
def test_invalid_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        inspect_outputs([], [], [], AnalysisConfig(output_text_iou_threshold=threshold))