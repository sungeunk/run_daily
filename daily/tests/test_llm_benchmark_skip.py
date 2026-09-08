from __future__ import annotations

import pytest

from common import llm_benchmark_skip


@pytest.fixture(autouse=True)
def isolate_test():
    yield


@pytest.mark.parametrize(
    ('platform_key', 'sku', 'should_skip'),
    [
        ('PTL', None, False),
        ('PTL', 'B70', False),
        ('BMG', 'B580', True),
        ('BMG', 'B70', False),
        ('BMG', 'B60', True),
        (None, None, False),
    ],
)
def test_run_only_policy(platform_key: str | None, sku: str | None,
                         should_skip: bool) -> None:
    reason = llm_benchmark_skip.evaluate_policy('qwen3.6-35b-a3b', platform_key, sku)

    assert (reason is not None) is should_skip


def test_platform_and_sku_blocks_are_combined() -> None:
    assert llm_benchmark_skip.evaluate_policy('gpt-oss-20b', 'MTL', None) is not None
    assert llm_benchmark_skip.evaluate_policy('gpt-oss-20b', 'BMG', 'B580') is not None
    assert llm_benchmark_skip.evaluate_policy('gpt-oss-20b', 'PTL', 'B70') is None


def test_sku_policy_has_priority_over_platform_policy() -> None:
    assert llm_benchmark_skip.evaluate_policy('qwen3.6-35b-a3b', 'BMG', 'B580') is not None
    assert llm_benchmark_skip.evaluate_policy('qwen3.6-35b-a3b', 'PTL', 'B70') is None


def test_incompatible_sku_falls_back_to_platform() -> None:
    platform_key, sku = llm_benchmark_skip.normalize_device_identity('MTL', 'B70')
    reason = llm_benchmark_skip.evaluate_policy('qwen3.6-35b-a3b', platform_key, sku)

    assert (platform_key, sku) == ('MTL', None)
    assert reason is not None


def test_explicit_block_takes_precedence_over_allow_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(llm_benchmark_skip.RUN_ONLY_ON, 'test-model', ['PTL'])
    monkeypatch.setitem(llm_benchmark_skip.SKIP_MODELS_BY_PLATFORM, 'PTL', ['test-model'])

    reason = llm_benchmark_skip.evaluate_policy('test-model', 'PTL', None)

    assert reason == 'test-model is skipped on PTL (timeout risk)'


def test_get_skip_reason_uses_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_benchmark_skip, 'resolve_platform', lambda _: ('BMG', 'B580'))

    reason = llm_benchmark_skip.get_skip_reason('qwen3.6-35b-a3b', 'GPU')

    assert reason is not None
    assert 'B580' in reason