from __future__ import annotations

from common.gpu_platform import get_device_platform_key, get_device_sku_key


# Shared timeout-based skip policy for both the legacy benchmark runner and
# the pytest benchmark suite.
SKIP_MODELS_BY_PLATFORM: dict[str, list[str]] = {
    'MTL': ['gemma-4-26b-a4b-it', 'gpt-oss-20b', 'qwen3.6-35b-a3b'],
    'LNL': ['qwen3.6-35b-a3b'],
    'PTL': [],
    'ARL': ['qwen3.6-35b-a3b'],
}

SKIP_MODELS_BY_SKU: dict[str, list[str]] = {
    'B580': ['gpt-oss-20b', 'gemma-4-26b-a4b-it', 'qwen3.6-35b-a3b'],
    'B70': [],
    'A770': ['qwen3.6-35b-a3b'],
}


def resolve_platform(device: str) -> tuple[str | None, str | None]:
    """Return ``(platform_key, sku_key)`` for the selected device."""
    platform_key = get_device_platform_key(device)
    return (
        platform_key.strip().upper() if platform_key else None,
        get_device_sku_key(device),
    )


def skipped_models(platform_key: str | None, sku: str | None) -> list[str]:
    """Return the effective skip list for the resolved platform/SKU."""
    models = list(SKIP_MODELS_BY_PLATFORM.get(platform_key, [])) if platform_key else []
    if sku:
        models.extend(SKIP_MODELS_BY_SKU.get(sku, []))
    return models


def get_skip_reason(model: str, device: str) -> str | None:
    """Return a skip reason if *model* is blocked on the selected device."""
    platform_key, sku = resolve_platform(device)
    if not platform_key and not sku:
        return None

    if model in skipped_models(platform_key, sku):
        label = '/'.join(dict.fromkeys(key for key in (platform_key, sku) if key))
        return f'{model} is skipped on {label} (timeout risk)'

    return None