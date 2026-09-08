from __future__ import annotations

from .gpu_platform import get_device_platform_key, get_device_sku_key


# Shared timeout-based skip policy for both the legacy benchmark runner and
# the pytest benchmark suite.
SKIP_MODELS_BY_PLATFORM: dict[str, list[str]] = {
    'MTL': ['gemma-4-26b-a4b-it', 'gpt-oss-20b'],
    'LNL': [],
    'PTL': [],
    'ARL': [],
}

SKIP_MODELS_BY_SKU: dict[str, list[str]] = {
    'B580': ['gemma-4-26b-a4b-it', 'gpt-oss-20b'],
    'B70': [],
    'A770': [],
}

# Allow-listed models: they run *only* where they are listed, and are skipped
# everywhere else. Use this instead of enumerating every platform/SKU in the
# skip lists above when a model needs more memory or run time than most parts
# have. An unknown SKU falls back to the platform rule, and an unknown
# platform/SKU pair runs by default. Entries may be either platform keys
# ('PTL') or marketing SKU labels ('B70'); a matching known key lets the model
# run.
RUN_ONLY_ON: dict[str, list[str]] = {
    # Of the BMG parts only Pro B70 (16GB) has the memory to finish this one
    # inside the daily window, so B580 and the smaller Pro SKUs stay out.
    'qwen3.6-35b-a3b': ['PTL', 'B70'],
}

_SKU_PLATFORM: dict[str, str] = {
    'B580': 'BMG',
    'B570': 'BMG',
    'B70': 'BMG',
    'B60': 'BMG',
    'B50': 'BMG',
    'A770': 'DG2',
    'A750': 'DG2',
}


def normalize_device_identity(platform_key: str | None,
                              sku: str | None) -> tuple[str | None, str | None]:
    """Normalize keys and discard a SKU that conflicts with its platform."""
    platform_key = platform_key.strip().upper() if platform_key else None
    sku = sku.strip().upper() if sku else None
    if platform_key in _SKU_PLATFORM:
        platform_key = _SKU_PLATFORM[platform_key]
    sku_platform = _SKU_PLATFORM.get(sku) if sku else None
    if sku_platform and platform_key and sku_platform != platform_key:
        sku = None
    return platform_key, sku


def resolve_platform(device: str) -> tuple[str | None, str | None]:
    """Return a compatible ``(platform_key, sku_key)`` device identity."""
    return normalize_device_identity(
        get_device_platform_key(device),
        get_device_sku_key(device),
    )


def skipped_models(platform_key: str | None, sku: str | None) -> list[str]:
    """Return models blocked by the highest-priority known device rule."""
    if sku in SKIP_MODELS_BY_SKU:
        return list(SKIP_MODELS_BY_SKU[sku])
    return list(SKIP_MODELS_BY_PLATFORM.get(platform_key, [])) if platform_key else []


def evaluate_policy(model: str, platform_key: str | None,
                    sku: str | None) -> str | None:
    """Return a policy reason using SKU, platform, then run-by-default rules."""
    allowed = RUN_ONLY_ON.get(model)
    if allowed is not None:
        target = sku or platform_key
        if target and target not in allowed:
            return f'{model} runs only on {"/".join(allowed)}, not on {target}'

    if model in skipped_models(platform_key, sku):
        target = sku or platform_key or 'unknown platform/SKU'
        return f'{model} is skipped on {target} (timeout risk)'

    return None


def get_skip_reason(model: str, device: str) -> str | None:
    """Return a skip reason if *model* is blocked on the selected device."""
    platform_key, sku = resolve_platform(device)
    reason = evaluate_policy(model, platform_key, sku)
    if reason and not platform_key and not sku:
        return f'{reason} for device {device!r}'
    return reason