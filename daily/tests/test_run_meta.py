from __future__ import annotations

import pytest

from analysis.report import _gpu_memory_text
from run import _adapter_name_matches

pytestmark = pytest.mark.dev_only


def test_adapter_name_matches_ignores_punctuation_and_suffix() -> None:
    assert _adapter_name_matches(
        'Intel(R) Arc(TM) 140V GPU (16GB)', 'Intel(R) Arc(TM) 140V GPU (16GB) '
    )
    assert _adapter_name_matches('Intel(R) Arc(TM) A770 Graphics', 'Intel Arc A770 Graphics')
    assert not _adapter_name_matches('Intel(R) Arc(TM) A770 Graphics', 'Intel(R) UHD Graphics 770')
    assert not _adapter_name_matches(None, 'Intel(R) UHD Graphics 770')


def _text(**meta) -> tuple[str | None, str]:
    return _gpu_memory_text({'meta': meta})


def test_igpu_dedicated_memory_is_hidden() -> None:
    # The DXGI figure is only the BIOS carve-out there, so it is not reported.
    dedicated, _ = _text(gpu_dedicated_memory_mb=128.0, gpu_shared_memory_mb=18392.0)
    assert dedicated is None


def test_dgpu_dedicated_memory_is_reported() -> None:
    dedicated, _ = _text(gpu_dedicated_memory_mb=16256.0, gpu_shared_memory_mb=16384.0)
    assert dedicated == '15.88 GB (16,256 MB)'


def test_shared_memory_override_states() -> None:
    _, default_unset = _text(
        gpu_shared_memory_mb=18392.0,
        gpu_shared_memory_override=0,
    )
    assert default_unset == '17.96 GB (18,392 MB)'

    _, overridden = _text(
        gpu_shared_memory_mb=18392.0,
        gpu_shared_memory_override=1,
    )
    assert overridden.endswith('— overridden (IncreaseFixedSegment=1)')

    _, unknown = _text(gpu_shared_memory_mb=18392.0)
    assert unknown == '17.96 GB (18,392 MB)'
