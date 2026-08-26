from __future__ import annotations

from analysis.report import _gpu_memory_text
from run import _adapter_name_matches


def test_adapter_name_matches_ignores_punctuation_and_suffix() -> None:
    assert _adapter_name_matches(
        'Intel(R) Arc(TM) 140V GPU (16GB)', 'Intel(R) Arc(TM) 140V GPU (16GB) '
    )
    assert _adapter_name_matches('Intel(R) Arc(TM) A770 Graphics', 'Intel Arc A770 Graphics')
    assert not _adapter_name_matches('Intel(R) Arc(TM) A770 Graphics', 'Intel(R) UHD Graphics 770')
    assert not _adapter_name_matches(None, 'Intel(R) UHD Graphics 770')


def _text(**meta) -> tuple[str, str]:
    return _gpu_memory_text({'meta': meta})


def test_igpu_dedicated_memory_is_labelled_as_carve_out() -> None:
    dedicated, _ = _text(gpu_dedicated_memory_mb=128.0, gpu_shared_memory_mb=18392.0)
    assert dedicated == '0.12 GB (128 MB) — iGPU carve-out, usable memory is shared'


def test_dgpu_dedicated_memory_has_no_carve_out_note() -> None:
    dedicated, _ = _text(gpu_dedicated_memory_mb=16256.0, gpu_shared_memory_mb=16384.0)
    assert dedicated == '15.88 GB (16,256 MB)'


def test_shared_memory_override_states() -> None:
    _, present_default = _text(
        gpu_shared_memory_mb=18392.0,
        gpu_shared_memory_override=0,
        gpu_shared_memory_override_present=True,
    )
    assert present_default.endswith('— driver default')

    _, absent_default = _text(
        gpu_shared_memory_mb=18392.0,
        gpu_shared_memory_override=0,
        gpu_shared_memory_override_present=False,
    )
    assert absent_default.endswith('— driver default (IncreaseFixedSegment unset)')

    _, overridden = _text(
        gpu_shared_memory_mb=18392.0,
        gpu_shared_memory_override=1,
        gpu_shared_memory_override_present=True,
    )
    assert overridden.endswith('— overridden (IncreaseFixedSegment=1)')

    _, unknown = _text(gpu_shared_memory_mb=18392.0)
    assert unknown == '17.96 GB (18,392 MB)'
