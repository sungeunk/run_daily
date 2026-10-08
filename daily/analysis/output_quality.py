"""Advisory generated-output checks, separate from performance verdicts."""

from __future__ import annotations

import base64
import http.client
import io
import json
import logging
import math
import re
import struct
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import quote, urlparse

from common.output_capture import generated_texts
from .types import AnalysisConfig, OutputQualityResult, OutputQualityRow

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_TEXT = 200_000
_MAX_IMAGE_PIXELS = 16_000_000
_ARTIFACT_FETCH_BUDGET_SEC = 20.0
_IMAGE_ARTIFACT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
log = logging.getLogger(__name__)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


@dataclass
class OutputSample:
    model: str
    precision: str
    prompt: str
    kind: str
    fingerprint: str | None = None
    text: str | None = None
    image: bytes | None = None
    image_error: str = ""
    text_truncated: bool = False

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.model, self.precision, self.prompt, self.kind


def text_iou(left: str, right: str) -> float | None:
    """Character 3-gram Jaccard similarity after Unicode/whitespace normalization."""
    def grams(text: str) -> set[str]:
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if not normalized:
            return set()
        if len(normalized) < 3:
            return {normalized}
        return {normalized[index:index + 3] for index in range(len(normalized) - 2)}

    left_set, right_set = grams(left), grams(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else None


def text_warnings(text: str) -> list[str]:
    reasons: list[str] = []
    if not text.strip():
        return ["Empty generated text"]
    if "\ufffd" in text or any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text):
        reasons.append("Invalid/control characters in generated text")
    if re.search(r"([^\w\s])\1{11,}", text):
        reasons.append("Repeated special character (12 or more)")
    words = re.findall(r"\w+", text.casefold())
    for width in range(1, 9):
        consecutive = 0
        for index in range(width, len(words)):
            consecutive = consecutive + 1 if words[index] == words[index - width] else 0
            if consecutive >= 7 * width:
                reasons.append("Repeated word/phrase (8 or more)")
                return reasons
    return reasons


def _image_data(content: bytes) -> tuple[NDArray[np.uint8] | None, str, list[str], str, tuple[int, int] | None]:
    try:
        import numpy as np
        from PIL import Image, ImageStat, UnidentifiedImageError
    except ImportError:
        return None, "", [], "Image inspection dependencies unavailable", None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                size = source.size
                if source.width * source.height > _MAX_IMAGE_PIXELS:
                    return None, "", [], "Image exceeds inspection pixel limit", size
                source.load()
                image = source.convert("RGB")
                reasons = []
                if max(ImageStat.Stat(image).stddev) < 1.0:
                    reasons.append("Nearly constant/blank image")
                if "A" in source.getbands() and source.getchannel("A").getextrema()[1] == 0:
                    reasons.append("Fully transparent image")
                image.thumbnail((256, 256))
                array = np.asarray(image)
                preview = image.copy()
                preview.thumbnail((128, 128))
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG")
                uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                return array, uri, reasons, "", size
    except (
        OSError, ValueError, SyntaxError, struct.error, zlib.error, TypeError,
        MemoryError, UnidentifiedImageError, Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return None, "", ["Image cannot be decoded"], "Image cannot be decoded", None


def _inspect_sample(
    sample: OutputSample,
    references: dict[str, dict[tuple[str, str, str, str], OutputSample]],
    config: AnalysisConfig,
) -> OutputQualityRow:
    row = OutputQualityRow(sample.model, sample.precision, sample.prompt, sample.kind, "pass")
    failures: list[str] = []
    unavailable: list[str] = []
    array, image_size = None, None
    if sample.kind == "text":
        if sample.text_truncated:
            unavailable.append("Current generated text was truncated before inspection")
        elif sample.text is None:
            unavailable.append("Current generated text not retained")
        elif len(sample.text) > _MAX_TEXT:
            unavailable.append("Text exceeds inspection size limit")
        else:
            failures.extend(text_warnings(sample.text))
            row.current_preview = sample.text[:400]
    elif sample.image is None:
        unavailable.append(sample.image_error or "Current image unavailable")
    else:
        array, row.current_preview, image_warnings, error, image_size = _image_data(sample.image)
        failures.extend(image_warnings)
        if error:
            unavailable.append(error)

    row.status = "warning" if failures else "unavailable" if unavailable else "pass"
    comparison_notes: list[str] = []
    for label, samples in references.items():
        reference = samples.get(sample.key)
        if reference is None:
            comparison_notes.append(f"{label}: matching output unavailable")
            continue
        reference_array, reference_size = None, None
        if sample.kind == "text":
            setattr(row, f"{label}_preview", (reference.text or "")[:400])
        elif reference.image is not None:
            reference_array, preview, _, error, reference_size = _image_data(reference.image)
            setattr(row, f"{label}_preview", preview)
            if error:
                comparison_notes.append(f"{label}: {error}")
        condition = "verified"
        if not sample.fingerprint or not reference.fingerprint:
            condition = "unverified"
            comparison_notes.append(f"{label}: conditions unverified; similarity is reference only")
        elif sample.fingerprint != reference.fingerprint:
            condition = "different"
            comparison_notes.append(f"{label}: prompt/generation settings differ; similarity is reference only")
        score = None
        if sample.kind == "text":
            if sample.text_truncated or reference.text_truncated:
                comparison_notes.append(f"{label}: truncated text comparison unavailable")
            elif sample.text is not None and reference.text is not None and max(len(sample.text), len(reference.text)) <= _MAX_TEXT:
                score = text_iou(sample.text, reference.text)
            threshold = config.output_text_iou_threshold
            metric = "IoU"
        else:
            threshold = config.output_image_ssim_threshold
            metric = "SSIM"
            if array is not None and reference_array is not None and image_size == reference_size:
                try:
                    from skimage.metrics import structural_similarity
                    window = min(7, array.shape[0], array.shape[1])
                    window -= 1 - window % 2
                    if window >= 3:
                        score = float(structural_similarity(array, reference_array, channel_axis=2, data_range=255, win_size=window))
                except ImportError:
                    comparison_notes.append(f"{label}: SSIM dependency unavailable")
                except (ValueError, TypeError, IndexError, RuntimeError):
                    comparison_notes.append(f"{label}: SSIM comparison unavailable")
            elif image_size is not None and reference_size is not None and image_size != reference_size:
                comparison_notes.append(f"{label}: image dimensions differ")
        if score is None or not math.isfinite(score):
            comparison_notes.append(f"{label}: {metric} comparison unavailable")
        else:
            setattr(row, f"{label}_score", score)
            setattr(row, f"{label}_comparison", condition)
            if score < threshold:
                setattr(row, f"{label}_warning", True)
                comparison_notes.append(f"{label}: {metric} {score:.3f} < {threshold:.3f}; review required")
    row.reasons = list(dict.fromkeys(failures + unavailable + comparison_notes))
    return row


def inspect_outputs(
    current: list[OutputSample], baseline: list[OutputSample], release: list[OutputSample],
    config: AnalysisConfig,
) -> OutputQualityResult:
    """Separate current-output checks from advisory, condition-labeled comparisons."""
    for threshold in (config.output_text_iou_threshold, config.output_image_ssim_threshold):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Output quality thresholds must be between 0 and 1")
    references = {"baseline": {sample.key: sample for sample in baseline}}
    if config.release_enabled:
        references["release"] = {sample.key: sample for sample in release}
    result = OutputQualityResult()
    for sample in current:
        try:
            row = _inspect_sample(sample, references, config)
        except Exception:
            log.warning("Output-quality inspection failed for one sample", exc_info=True)
            row = OutputQualityRow(
                sample.model, sample.precision, sample.prompt, sample.kind,
                "unavailable", reasons=["Output inspection failed"],
            )
        result.rows.append(row)
    if not current:
        result.detail = "No generated output records available; quality was not verified."
    return result


def _raw_sections(raw: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    command: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("[CMD] "):
            if command is not None:
                sections[command] = "\n".join(lines)
            command, lines = line[6:].strip(), []
        elif command is not None:
            lines.append(line)
    if command is not None:
        sections[command] = "\n".join(lines)
    return sections


def samples_from_summary(
    summary: dict, raw: str, image_reader: Callable[[str], bytes | None], stamp: str,
) -> list[OutputSample]:
    from common.delivery import staged_image_slot

    sections = _raw_sections(raw)
    samples: list[OutputSample] = []
    seen_images: set[str] = set()
    tests = summary.get("tests")
    if not isinstance(tests, list):
        return samples
    for test in tests:
        if not isinstance(test, dict):
            continue
        metrics = test.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        kind = metrics.get("test_type")
        if kind not in {"llm_benchmark", "image_generation"} or test.get("outcome") == "skipped":
            continue
        model, precision = str(metrics.get("model", "")), str(metrics.get("precision", ""))
        fingerprint = metrics.get("generation_fingerprint")
        fingerprint = fingerprint if isinstance(fingerprint, str) else None
        if kind == "llm_benchmark":
            generated = metrics.get("generated_outputs")
            records = (
                [record for record in generated if isinstance(record, dict)]
                if isinstance(generated, list) else []
            )
            needs_raw = not records or any(record.get("truncated") for record in records)
            raw_records = (
                generated_texts(sections.get(str(metrics.get("cmd", "")), ""))
                if raw and needs_raw else []
            )
            if raw_records:
                records = raw_records
            if not records:
                data = metrics.get("data")
                records = [
                    {**item, "iteration": "selected"}
                    for item in data if isinstance(item, dict)
                ] if isinstance(data, list) else []
            for record in records or [{"prompt_idx": "unknown", "iteration": "unknown"}]:
                samples.append(OutputSample(
                    model, precision, f"{record.get('prompt_idx', 'unknown')} / {record.get('iteration', 'unknown')}",
                    "text", fingerprint,
                    text=record.get("generated_text") if isinstance(record.get("generated_text"), str) else None,
                    text_truncated=bool(record.get("truncated", False)),
                ))
        else:
            data = metrics.get("data")
            records = data if isinstance(data, list) and data else [{}]
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                original = str(record.get("image_path") or "")
                if not original or original in seen_images:
                    continue
                seen_images.add(original)
                match = re.search(r"_p(\d+)(?:_iter(\d+))?_pid\d+_output\.", original)
                prompt = f"{match[1]} / {match[2] or 'warm-up'}" if match else f"{index} / unknown"
                suffix = Path(original).suffix.lower()
                if suffix not in _IMAGE_ARTIFACT_SUFFIXES:
                    continue
                slot = staged_image_slot(model, precision, index, suffix)
                image = image_reader(f"daily.{stamp}.image.{slot}") if original else None
                settings = [record.get(key) for key in ("width", "height", "steps", "batch_size")]
                identity = fingerprint + json.dumps(settings) if fingerprint and all(value is not None for value in settings) else None
                samples.append(OutputSample(model, precision, prompt, "image", identity, image=image))
    return samples


def _read_artifact(config: AnalysisConfig, machine: str, stamp: str, name: str,
                   local_dir: Path | None, *, deadline: float | None = None) -> bytes | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", machine) or not re.fullmatch(r"\d{8}_\d{4}", stamp):
        return None
    if Path(name).name != name or "/" in name or "\\" in name:
        return None
    if not name.startswith(f"daily.{stamp}.") or not name.endswith(
        (".summary.json", ".raw", *_IMAGE_ARTIFACT_SUFFIXES)
    ):
        return None
    from common.delivery import REMOTE_BASE_DIR

    archive = Path(REMOTE_BASE_DIR) / machine / f"{stamp[:4]}.{stamp[4:6]}"
    for directory in (local_dir, archive):
        if directory is not None:
            path = directory / name
            try:
                data = bytearray()
                with path.open("rb") as stream:
                    while len(data) < _MAX_ARTIFACT_BYTES + 1:
                        if deadline is not None and time.monotonic() >= deadline:
                            return None
                        chunk = stream.read(min(64 * 1024, _MAX_ARTIFACT_BYTES + 1 - len(data)))
                        if not chunk:
                            break
                        data.extend(chunk)
                        if deadline is not None and time.monotonic() >= deadline:
                            return None
                if data and len(data) <= _MAX_ARTIFACT_BYTES:
                    return bytes(data)
            except OSError:
                pass
    if urlparse(config.output_artifact_base_url).scheme not in {"http", "https"}:
        return None
    remaining = (
        _ARTIFACT_FETCH_BUDGET_SEC if deadline is None
        else deadline - time.monotonic()
    )
    if remaining <= 0:
        return None
    url = f"{config.output_artifact_base_url.rstrip('/')}/{quote(machine)}/{stamp[:4]}.{stamp[4:6]}/{quote(name)}"
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler(),
        )
        chunks = bytearray()
        with opener.open(url, timeout=min(config.mcp_timeout_sec, 10, remaining)) as response:
            while len(chunks) < _MAX_ARTIFACT_BYTES + 1:
                remaining = (
                    _ARTIFACT_FETCH_BUDGET_SEC if deadline is None
                    else deadline - time.monotonic()
                )
                if remaining <= 0:
                    return None
                read_size = min(64 * 1024, _MAX_ARTIFACT_BYTES + 1 - len(chunks))
                chunk = response.read1(read_size)
                if not chunk:
                    break
                chunks.extend(chunk)
                if deadline is not None and time.monotonic() >= deadline:
                    return None
        data = bytes(chunks)
        return data if len(data) <= _MAX_ARTIFACT_BYTES else None
    except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
        return None


def quality_for_runs(
    config: AnalysisConfig, machine: str, current_stamp: str,
    baseline_stamp: str | None, release_stamp: str | None,
    *, summary: dict | None = None, local_dir: Path | None = None,
) -> OutputQualityResult:
    """Load only named run artifacts, never a local benchmark database."""
    deadline = time.monotonic() + _ARTIFACT_FETCH_BUDGET_SEC

    def load(stamp: str | None, current: bool = False) -> list[OutputSample]:
        if not stamp:
            return []
        directory = local_dir if current else None
        def read(name: str) -> bytes | None:
            return _read_artifact(
                config, machine, stamp, name, directory, deadline=deadline,
            )

        payload = summary if current else None
        if payload is None:
            data = read(f"daily.{stamp}.summary.json")
            if data is None:
                return []
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError):
                return []
        if not isinstance(payload, dict) or not isinstance(payload.get("tests"), list):
            return []
        needs_raw = any(
            isinstance(test, dict)
            and isinstance(test.get("metrics"), dict)
            and test["metrics"].get("test_type") == "llm_benchmark"
            and (
                not test["metrics"].get("generated_outputs")
                or any(
                    isinstance(record, dict) and record.get("truncated")
                    for record in test["metrics"].get("generated_outputs", [])
                )
            )
            for test in payload["tests"]
        )
        raw = read(f"daily.{stamp}.raw") if needs_raw else None
        return samples_from_summary(payload, raw.decode("utf-8", errors="replace") if raw else "", read, stamp)

    return inspect_outputs(load(current_stamp, True), load(baseline_stamp), load(release_stamp), config)