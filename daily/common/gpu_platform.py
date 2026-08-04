from __future__ import annotations

from functools import lru_cache
import importlib
import os
import platform


_PLATFORM_LABEL_KEYS = ('A770', 'B70', 'B580',
                        'NVL', 'PTL', 'LNL', 'ARL', 'MTL', 'ADL', 'BMG', 'DG2', 'PVC', 'DG1', 'XE_LP')


def _get_platform_key_from_labels(labels: str) -> str | None:
    label_set = {label.strip().upper() for label in labels.split() if label.strip()}
    for platform_key in _PLATFORM_LABEL_KEYS:
        if platform_key in label_set:
            return platform_key
    return None


def _get_pyopencl_signature(device: str) -> str | None:
    """Best-effort Intel GPU signature from pyopencl for the selected device."""
    try:
        pyopencl = importlib.import_module('pyopencl')
    except Exception:
        return None

    try:
        if device == 'GPU':
            target_index = 0
        elif device.startswith('GPU.'):
            target_index = int(device.split('.', 1)[1])
        else:
            target_index = 0
    except Exception:
        target_index = 0

    try:
        intel_gpus: list[tuple[object, object]] = []
        for platform_obj in pyopencl.get_platforms():
            try:
                devices = platform_obj.get_devices(device_type=pyopencl.device_type.GPU)
            except Exception:
                continue
            for dev in devices:
                vendor_sig = f'{getattr(platform_obj, "vendor", "")} {getattr(dev, "vendor", "")}'.upper()
                if 'INTEL' in vendor_sig:
                    intel_gpus.append((platform_obj, dev))

        if not intel_gpus:
            return None

        if target_index < 0 or target_index >= len(intel_gpus):
            target_index = 0

        platform_obj, dev = intel_gpus[target_index]
        fields = [
            str(getattr(platform_obj, 'name', '') or ''),
            str(getattr(platform_obj, 'vendor', '') or ''),
            str(getattr(dev, 'name', '') or ''),
            str(getattr(dev, 'vendor', '') or ''),
            str(getattr(dev, 'version', '') or ''),
            str(getattr(dev, 'driver_version', '') or ''),
        ]
        return ' '.join(fields).upper()
    except Exception:
        return None


def _collect_openvino_signature(device: str) -> str | None:
    try:
        ov_module = importlib.import_module('openvino')
        core_cls = getattr(ov_module, 'Core')
        core = core_cls()
    except Exception:
        return None

    arch_prop_names = ['DEVICE_ARCHITECTURE', 'GPU_DEVICE_ARCHITECTURE', 'ARCHITECTURE']
    try:
        supported = core.get_property(device, 'SUPPORTED_PROPERTIES')
    except Exception:
        supported = []
    for prop in supported or []:
        prop_name = str(prop)
        if 'ARCHITECTURE' in prop_name.upper() and prop_name not in arch_prop_names:
            arch_prop_names.append(prop_name)

    props: list[str] = []
    for prop_name in ('FULL_DEVICE_NAME', *arch_prop_names):
        try:
            value = core.get_property(device, prop_name)
        except Exception:
            value = None
        if value:
            props.append(str(value).upper())

    if not props:
        return None

    signature = ' '.join(props)
    generic_igpu_name = any(token in signature for token in ('INTEL(R) GRAPHICS (IGPU)', 'ARC(TM) GRAPHICS (IGPU)'))
    pyopencl_signature = _get_pyopencl_signature(device)
    if pyopencl_signature and not generic_igpu_name:
        signature = f'{signature} {pyopencl_signature}'
    return signature


# Marketing SKU labels worth distinguishing *within* one architecture,
# because capability differs enough to change which models can run (e.g.
# BMG B580 12GB vs. Pro B70 16GB). Ordered longest-first so 'B580' wins
# before a hypothetical 'B5' prefix.
_SKU_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('B580', ('ARC(TM) B580', 'B580')),
    ('B570', ('ARC(TM) B570', 'B570')),
    ('B70',  ('ARC(TM) PRO B70', 'ARC PRO B70', 'PRO B70', 'B70')),
    ('B60',  ('ARC(TM) PRO B60', 'ARC PRO B60', 'PRO B60', 'B60')),
    ('B50',  ('ARC(TM) PRO B50', 'ARC PRO B50', 'PRO B50', 'B50')),
    ('A770', ('ARC(TM) A770', 'A770')),
    ('A750', ('ARC(TM) A750', 'A750')),
)


@lru_cache(maxsize=32)
def get_device_sku_key(device: str, host_name: str | None = None) -> str | None:
    """Resolve the marketing SKU label (e.g. 'B580', 'B70') for a device.

    Returns None when the SKU can't be determined or isn't one we
    distinguish. Callers should treat that as "use the architecture key
    alone" — see SKIP_MODELS_BY_SKU in tests/test_llm_benchmark.py.
    """
    explicit = os.environ.get('TARGET_DEVICE_NAME')
    if explicit:
        candidate = explicit.strip().upper()
        if any(candidate == sku for sku, _ in _SKU_PATTERNS):
            return candidate

    sources = [
        os.environ.get('TARGET_DEVICE_NAME', ''),
        os.environ.get('NODE_LABELS', ''),
        _collect_openvino_signature(device) or '',
    ]
    haystack = ' '.join(s.upper() for s in sources if s)
    if not haystack:
        return None

    for sku, tokens in _SKU_PATTERNS:
        if any(token in haystack for token in tokens):
            return sku
    return None


@lru_cache(maxsize=32)
def get_device_platform_key(device: str, host_name: str | None = None) -> str | None:
    """Resolve a normalized Intel GPU platform key for an OpenVINO device."""
    target_device_name = os.environ.get('TARGET_DEVICE_NAME')
    if target_device_name:
        return target_device_name.strip().upper()

    jenkins_platform_key = _get_platform_key_from_labels(os.environ.get('NODE_LABELS', ''))
    if jenkins_platform_key:
        return jenkins_platform_key

    signature = _collect_openvino_signature(device)
    if not signature:
        return None

    if any(token in signature for token in ('PANTHER LAKE', 'PTL', 'ARC(TM) B390', 'ARC(TM) B370', 'B390', 'B370')):
        return 'PTL'
    if any(token in signature for token in ('LUNAR LAKE', 'LNL', 'ARC(TM) 140V', 'ARC(TM) 130V')):
        return 'LNL'
    if any(token in signature for token in ('ARROW LAKE', 'ARL', 'ARC(TM) 140T', 'ARC(TM) 130T', 'XE-LPG+')):
        return 'ARL'
    if any(token in signature for token in ('METEOR LAKE', 'MTL', 'XE-LPG')):
        return 'MTL'
    if any(token in signature for token in ('ALDER LAKE', 'RAPTOR LAKE', 'ADL', 'RPL', 'UHD GRAPHICS 770')):
        return 'ADL'

    # 'ARC PRO B' does not match the real device name 'Arc(TM) Pro B70
    # Graphics' because of the '(TM)' between the two words — match the
    # Pro form explicitly as well.
    if any(token in signature for token in (
        'BMG', 'BATTLEMAGE', 'XE2', 'ARC(TM) B570', 'ARC(TM) B580',
        'ARC(TM) B50', 'ARC(TM) B60', 'ARC(TM) B70',
        'ARC PRO B', 'ARC(TM) PRO B',
    )):
        return 'BMG'

    if any(token in signature for token in (
        'DG2', 'ALCHEMIST', 'XE-HPG', 'ARC(TM) A310', 'ARC(TM) A350',
        'ARC(TM) A370', 'ARC(TM) A380', 'ARC(TM) A580', 'ARC(TM) A750',
        'ARC(TM) A770', 'ARC PRO A', 'FLEX 140', 'FLEX 170',
    )):
        return 'DG2'

    if any(token in signature for token in ('PONTE VECCHIO', 'PVC', 'DATA CENTER GPU MAX', 'MAX 1100', 'MAX 1550')):
        return 'PVC'
    if any(token in signature for token in ('DG1', 'IRIS XE MAX')):
        return 'DG1'
    if any(token in signature for token in ('IRIS XE', 'IRIS PLUS', 'IRIS PRO', 'UHD GRAPHICS', 'HD GRAPHICS', 'INTEL(R) GRAPHICS')):
        return 'XE_LP'

    if any(token in signature for token in ('ARC(TM) GRAPHICS', 'INTEL(R) GRAPHICS')):
        node = (host_name or platform.node()).upper()
        if 'MTL' in node:
            return 'MTL'
        if 'ARL' in node:
            return 'ARL'
        if 'PTL' in node:
            return 'PTL'
        if 'LNL' in node:
            return 'LNL'

    return None
