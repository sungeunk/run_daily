#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil>=6.0.0",
#   "wmi>=1.5.1",
# ]
# ///

"""Sample first-token latency related system factors on Windows.

The script records a time series for:
- CPU clock, temperature, and usage
- GPU clock, throttle reasons, utilization, and memory usage
- Host memory speed, total size, and usage
- Benchmark process identity, priority, affinity, and foreground ownership
- Windows timer resolution

Notes:
- Intel GPU name comes from Win32_VideoController via CIM.
- Intel GPU telemetry uses Level Zero Sysman (ze_loader.dll) and needs no external tooling.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import http.client
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Final


LOGGER: Final[logging.Logger] = logging.getLogger("first_token_monitor")
PSUTIL: object | None = None
WMI_LIB: object | None = None

# Connections and adapter identity are hoisted out of the sampling loop:
# re-creating them per sample adds enough CPU load to perturb the measurement.
_WMI_CONNECTIONS: dict[str, object | None] = {}
_INTEL_GPU_NAMES: dict[int, str | None] = {}
_INTEL_GPU_DETECTED: bool = False

_SYSMAN: "SysmanHandles | None" = None
_SYSMAN_DEVICE_INDEX: int | None = None
_SYSMAN_RESOLVED: bool = False
_LAST_ENERGY: tuple[int, int] | None = None
_LAST_ENGINE: tuple[int, int] | None = None
_LAST_PAGE_FAULTS: tuple[int, float] | None = None
_LAST_PROCESS_CPU: dict[int, tuple[str, float]] = {}
_LAST_PROCESS_CPU_TIME: float | None = None
_LAST_TEMPS: "CpuTemperatures | None" = None
_LAST_TEMP_AT: float | None = None
_LAST_CPU_SNAPSHOT: tuple[float, float, float] | None = None
_LAST_CPU_USAGE: float | None = None
_LAST_CPU_USAGE_AT: float | None = None

# Below roughly one scheduler tick the busy/idle delta is dominated by quantization.
CPU_USAGE_MIN_WINDOW_SEC: Final[float] = 0.2
LHM_WEB_URL: Final[str] = "http://127.0.0.1:8085/data.json"

PROCESS_POWER_THROTTLING_CURRENT_VERSION: Final[int] = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED: Final[int] = 0x1
PROCESS_INFORMATION_CLASS_POWER_THROTTLING: Final[int] = 4
PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000

# Windows priority class constants
PRIORITY_CLASSES: Final[dict[int, str]] = {
    0x00000040: "IDLE",
    0x00004000: "BELOW_NORMAL",
    0x00000020: "NORMAL",
    0x00008000: "ABOVE_NORMAL",
    0x00000080: "HIGH",
    0x00000100: "REALTIME",
}


@dataclass(frozen=True, slots=True)
class TimerResolution:
    minimum_ms: float
    maximum_ms: float
    current_ms: float


@dataclass(frozen=True, slots=True)
class ProcessPriority:
    pid: int
    class_value: int
    class_name: str


@dataclass(frozen=True, slots=True)
class CpuTemperatures:
    overall_c: float | None
    core_max_c: float | None
    core_avg_c: float | None
    core_temps_c: list[float]
    source: str


@dataclass(frozen=True, slots=True)
class HostMemoryInfo:
    speed_mts: float | None
    total_gb: float | None


@dataclass(frozen=True, slots=True)
class LhmGpuMetrics:
    clock_mhz: float | None
    memory_clock_mhz: float | None
    utilization_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    power_watts: float | None
    temperature_c: float | None
    fan_rpm: float | None


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    name: str | None
    cmdline: str | None
    parent_pid: int | None
    session_id: int | None
    create_time_utc: str | None


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def run_command(command: list[str], timeout_sec: float = 3.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        LOGGER.debug("Command failed to execute: %s (%s)", command, exc)
        return None
    if completed.returncode != 0:
        LOGGER.debug(
            "Command returned non-zero exit code (%s): %s, stderr=%s",
            completed.returncode,
            command,
            completed.stderr.strip(),
        )
        return None
    return completed.stdout


def run_powershell(command: str, timeout_sec: float = 4.0) -> str | None:
    return run_command(["powershell", "-NoProfile", "-Command", command], timeout_sec)


def ensure_required_modules() -> tuple[object, object]:
    missing: list[str] = []
    for module_name in ("psutil", "wmi"):
        if find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        missing_csv = ", ".join(missing)
        raise ModuleNotFoundError(
            "Required Python modules are missing: "
            f"{missing_csv}. Install with: uv pip install psutil wmi"
        )

    import psutil  # type: ignore
    import wmi  # type: ignore

    return psutil, wmi


def get_wmi_connection(namespace: str = "root\\cimv2") -> object | None:
    if WMI_LIB is None:
        return None
    if namespace in _WMI_CONNECTIONS:
        return _WMI_CONNECTIONS[namespace]
    try:
        connection = WMI_LIB.WMI(namespace=namespace)
    except Exception:
        connection = None
    _WMI_CONNECTIONS[namespace] = connection
    return connection


def get_cpu_clock_mhz() -> float | None:
    # psutil reads this via NtPowerInformation; the WMI path below costs ~1s per sample.
    if PSUTIL is not None:
        try:
            freq = PSUTIL.cpu_freq()
        except Exception:
            freq = None
        if freq is not None and getattr(freq, "current", 0):
            return float(freq.current)

    conn = get_wmi_connection()
    if conn is not None:
        try:
            cpus = conn.Win32_Processor()
        except Exception:
            cpus = []
        values: list[float] = []
        for cpu in cpus:
            raw = getattr(cpu, "CurrentClockSpeed", None)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            return sum(values) / len(values)

    # Fallback for systems where WMIC is unavailable.
    ps_output = run_powershell(
        "(Get-CimInstance Win32_Processor | Measure-Object -Property CurrentClockSpeed -Average).Average"
    )
    if not ps_output:
        return None
    try:
        return float(ps_output.strip())
    except ValueError:
        return None


def get_cpu_temp_c() -> float | None:
    # ACPI thermal zone is motherboard-level and may be unavailable on many systems.
    conn = get_wmi_connection("root\\wmi")
    if conn is None:
        return None

    temperatures: list[float] = []
    try:
        sensors = conn.MSAcpi_ThermalZoneTemperature()
    except Exception:
        sensors = []

    for sensor in sensors:
        raw = getattr(sensor, "CurrentTemperature", None)
        if raw is None:
            continue
        try:
            deci_kelvin = float(raw)
            celsius = (deci_kelvin / 10.0) - 273.15
        except (TypeError, ValueError):
            continue
        if -30.0 <= celsius <= 130.0:
            temperatures.append(celsius)
    if not temperatures:
        return None
    return max(temperatures)


def parse_sensor_float(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def valid_range(value: float | None, minimum: float, maximum: float) -> float | None:
    if value is None or not (minimum <= value <= maximum):
        return None
    return value


def iter_lhm_nodes(node: dict[str, object], in_temperatures: bool = False) -> Iterator[tuple[dict[str, object], bool]]:
    text = str(node.get("Text", ""))
    next_in_temperatures = in_temperatures or text == "Temperatures"
    yield node, next_in_temperatures
    children = node.get("Children")
    if not isinstance(children, list):
        return
    for child in children:
        if isinstance(child, dict):
            yield from iter_lhm_nodes(child, next_in_temperatures)


_LHM_PAYLOAD_CACHE: dict[str, object] | None = None
_LHM_PAYLOAD_VALID: bool = False


def reset_lhm_payload_cache() -> None:
    """Start a new sample so its probes share a single LibreHardwareMonitor fetch."""
    global _LHM_PAYLOAD_CACHE, _LHM_PAYLOAD_VALID
    _LHM_PAYLOAD_CACHE = None
    _LHM_PAYLOAD_VALID = False


def get_lhm_web_payload(timeout_sec: float = 1.0) -> dict[str, object] | None:
    global _LHM_PAYLOAD_CACHE, _LHM_PAYLOAD_VALID
    if _LHM_PAYLOAD_VALID:
        return _LHM_PAYLOAD_CACHE

    _LHM_PAYLOAD_VALID = True
    _LHM_PAYLOAD_CACHE = _fetch_lhm_web_payload(timeout_sec)
    return _LHM_PAYLOAD_CACHE


def _fetch_lhm_web_payload(timeout_sec: float) -> dict[str, object] | None:
    prefix = "http://127.0.0.1:8085"
    if not LHM_WEB_URL.startswith(prefix):
        return None
    path = LHM_WEB_URL[len(prefix) :]
    if not path:
        path = "/"

    conn = http.client.HTTPConnection("127.0.0.1", 8085, timeout=timeout_sec)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        if response.status != 200:
            return None
        payload = response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, TimeoutError):
        return None
    finally:
        conn.close()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def get_cpu_temps_lhm_web() -> CpuTemperatures | None:
    payload = get_lhm_web_payload()
    if payload is None:
        return None

    package_c: float | None = None
    core_max_c: float | None = None
    core_avg_c: float | None = None
    core_map: dict[str, float] = {}
    for node, in_temperatures in iter_lhm_nodes(payload):
        text = str(node.get("Text", ""))
        if not in_temperatures:
            continue
        value = parse_sensor_float(str(node.get("Value", "")))
        if value is None or not (-30.0 <= value <= 130.0):
            continue

        if text == "CPU Package":
            package_c = value
        elif text == "Core Max":
            core_max_c = value
        elif text == "Core Average":
            core_avg_c = value
        elif re.fullmatch(r"(?:P|E)-Core #\d+", text) or re.fullmatch(r"CPU Core #\d+", text):
            core_map[text] = value

    if package_c is None and core_max_c is None and core_avg_c is None and not core_map:
        return None

    if not core_map:
        core_temps: list[float] = []
    else:
        ordered = sorted(core_map.items(), key=lambda kv: kv[0])
        core_temps = [value for _, value in ordered]

    if core_max_c is None and core_temps:
        core_max_c = max(core_temps)
    if core_avg_c is None and core_temps:
        core_avg_c = sum(core_temps) / len(core_temps)

    overall_c = package_c if package_c is not None else core_max_c
    return CpuTemperatures(
        overall_c=overall_c,
        core_max_c=core_max_c,
        core_avg_c=core_avg_c,
        core_temps_c=core_temps,
        source="lhm-web",
    )


def lhm_hardware_matches(hardware_name: str, gpu_name: str) -> bool:
    left = re.sub(r"[^a-z0-9]+", " ", hardware_name.lower()).strip()
    right = re.sub(r"[^a-z0-9]+", " ", gpu_name.lower()).strip()
    return bool(left and right and (left in right or right in left))


def find_lhm_sensor_value(
    payload: dict[str, object],
    gpu_name: str,
    category: str,
    sensor: str,
) -> float | None:
    stack: list[tuple[dict[str, object], tuple[str, ...]]] = [(payload, ())]
    while stack:
        node, parts = stack.pop()
        text = str(node.get("Text", ""))
        current = (*parts, text)
        value = parse_sensor_float(str(node.get("Value", "")))
        if (
            len(current) >= 5
            and value is not None
            and lhm_hardware_matches(current[2], gpu_name)
            and current[3] == category
            and current[4] == sensor
        ):
            return value

        children = node.get("Children")
        if isinstance(children, list):
            for child in reversed(children):
                if isinstance(child, dict):
                    stack.append((child, current))
    return None


def get_lhm_gpu_metrics(gpu_name: str | None) -> tuple[LhmGpuMetrics | None, float | None]:
    if not gpu_name:
        return None, None

    start = time.perf_counter()
    payload = get_lhm_web_payload()
    query_duration_ms = (time.perf_counter() - start) * 1000.0
    if payload is None:
        return None, query_duration_ms

    render_compute = valid_range(
        find_lhm_sensor_value(payload, gpu_name, "Load", "GPU Render/Compute"), 0.0, 100.0
    )
    core_load = valid_range(find_lhm_sensor_value(payload, gpu_name, "Load", "GPU Core"), 0.0, 100.0)

    return LhmGpuMetrics(
        clock_mhz=valid_range(find_lhm_sensor_value(payload, gpu_name, "Clocks", "GPU Core"), 1.0, 10000.0),
        memory_clock_mhz=valid_range(find_lhm_sensor_value(payload, gpu_name, "Clocks", "GPU Memory"), 1.0, 10000.0),
        utilization_percent=render_compute if render_compute is not None else core_load,
        memory_used_mb=valid_range(find_lhm_sensor_value(payload, gpu_name, "Data", "GPU Memory Used"), 0.0, 1000000.0),
        memory_total_mb=valid_range(find_lhm_sensor_value(payload, gpu_name, "Data", "GPU Memory Total"), 1.0, 1000000.0),
        power_watts=valid_range(find_lhm_sensor_value(payload, gpu_name, "Powers", "GPU Package"), 0.0, 10000.0),
        temperature_c=valid_range(find_lhm_sensor_value(payload, gpu_name, "Temperatures", "GPU Core"), 1.0, 130.0),
        fan_rpm=valid_range(find_lhm_sensor_value(payload, gpu_name, "Fans", "GPU Fan"), 0.0, 10000.0),
    ), query_duration_ms


def get_cpu_temps() -> CpuTemperatures | None:
    """CPU temperature via LHM web API first, then ACPI thermal zone fallback."""
    lhm_web = get_cpu_temps_lhm_web()
    if lhm_web is not None:
        return lhm_web

    overall_c = get_cpu_temp_c()
    if overall_c is None:
        return None
    return CpuTemperatures(
        overall_c=overall_c,
        core_max_c=None,
        core_avg_c=None,
        core_temps_c=[],
        source="acpi-thermal-zone",
    )


def get_cpu_temp_cached(
    now: float, interval_sec: float, force: bool
) -> tuple["CpuTemperatures | None", float | None, float | None]:
    """Return (temps, age_ms, query_duration_ms). query_duration_ms is set only on a fresh probe."""
    global _LAST_TEMPS, _LAST_TEMP_AT

    if force or _LAST_TEMP_AT is None or (now - _LAST_TEMP_AT) >= interval_sec:
        t_probe_start = time.perf_counter()
        temps = get_cpu_temps()
        query_duration_ms = (time.perf_counter() - t_probe_start) * 1000.0
        if temps is not None:
            _LAST_TEMPS = temps
            _LAST_TEMP_AT = now
            return temps, 0.0, query_duration_ms
        if _LAST_TEMP_AT is None:
            return None, None, query_duration_ms

    return _LAST_TEMPS, (now - _LAST_TEMP_AT) * 1000.0, None


def get_cpu_usage_windowed(now: float) -> tuple[float | None, float | None]:
    """Return (usage_percent, age_ms) accumulated over at least CPU_USAGE_MIN_WINDOW_SEC."""
    global _LAST_CPU_SNAPSHOT, _LAST_CPU_USAGE, _LAST_CPU_USAGE_AT
    if PSUTIL is None:
        return None, None

    try:
        times = PSUTIL.cpu_times()
        total = float(sum(getattr(times, field) for field in times._fields))
        idle = float(times.idle)
    except Exception:
        return None, None

    age_ms = None if _LAST_CPU_USAGE_AT is None else (now - _LAST_CPU_USAGE_AT) * 1000.0
    if _LAST_CPU_SNAPSHOT is None:
        _LAST_CPU_SNAPSHOT = (total, idle, now)
        return None, None

    prev_total, prev_idle, prev_at = _LAST_CPU_SNAPSHOT
    if now - prev_at < CPU_USAGE_MIN_WINDOW_SEC:
        return _LAST_CPU_USAGE, age_ms

    _LAST_CPU_SNAPSHOT = (total, idle, now)
    delta_total = total - prev_total
    if delta_total <= 0:
        return _LAST_CPU_USAGE, age_ms

    usage = 100.0 * (1.0 - (idle - prev_idle) / delta_total)
    _LAST_CPU_USAGE = max(0.0, min(100.0, usage))
    _LAST_CPU_USAGE_AT = now
    return _LAST_CPU_USAGE, 0.0


def get_host_memory_info() -> HostMemoryInfo:
    conn = get_wmi_connection()
    speeds: list[float] = []
    total_bytes: int = 0
    if conn is not None:
        try:
            mem_modules = conn.Win32_PhysicalMemory()
        except Exception:
            mem_modules = []
        for mem in mem_modules:
            try:
                capacity = getattr(mem, "Capacity", None)
                if capacity is not None:
                    total_bytes += int(capacity)
            except (TypeError, ValueError):
                pass
            for field in ("ConfiguredClockSpeed", "Speed"):
                raw = getattr(mem, field, None)
                if raw is None:
                    continue
                try:
                    parsed = float(raw)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    speeds.append(parsed)
                    break

    speed_mts: float | None = None
    if speeds:
        speed_mts = sum(speeds) / len(speeds)
    else:
        ps_output = run_powershell(
            "$m=Get-CimInstance Win32_PhysicalMemory; "
            "if(-not $m){return}; "
            "$vals=@(); "
            "foreach($x in $m){ "
            "  if($x.ConfiguredClockSpeed -gt 0){$vals+=[double]$x.ConfiguredClockSpeed} "
            "  elseif($x.Speed -gt 0){$vals+=[double]$x.Speed} "
            "}; "
            "if($vals.Count -gt 0){ ($vals | Measure-Object -Average).Average }"
        )
        if ps_output:
            try:
                speed_mts = float(ps_output.strip())
            except ValueError:
                pass

    total_gb: float | None = round(total_bytes / (1024**3), 1) if total_bytes > 0 else None
    return HostMemoryInfo(speed_mts=speed_mts, total_gb=total_gb)


def get_memory_usage() -> tuple[float | None, int | None]:
    """Return (usage_percent, available_mb)."""
    if PSUTIL is None:
        return None, None
    try:
        mem = PSUTIL.virtual_memory()
        return float(mem.percent), int(mem.available) // (1024**2)
    except Exception:
        return None, None


def query_timer_resolution() -> TimerResolution | None:
    ntdll = ctypes.WinDLL("ntdll")
    min_units = ctypes.c_ulong()
    max_units = ctypes.c_ulong()
    cur_units = ctypes.c_ulong()

    status = ntdll.NtQueryTimerResolution(
        ctypes.byref(min_units),
        ctypes.byref(max_units),
        ctypes.byref(cur_units),
    )
    if status != 0:
        return None

    # Units are 100ns.
    unit_to_ms = 1.0 / 10_000.0
    return TimerResolution(
        minimum_ms=min_units.value * unit_to_ms,
        maximum_ms=max_units.value * unit_to_ms,
        current_ms=cur_units.value * unit_to_ms,
    )


def get_process_priority(pid: int) -> ProcessPriority | None:
    if PSUTIL is None:
        return None

    try:
        proc = PSUTIL.Process(pid)
        cls = int(proc.nice())
        return ProcessPriority(
            pid=pid,
            class_value=cls,
            class_name=PRIORITY_CLASSES.get(cls, f"UNKNOWN_{cls}"),
        )
    except Exception:
        return None


def get_process_session_id(pid: int) -> int | None:
    kernel32 = ctypes.WinDLL("kernel32")
    session_id = ctypes.c_ulong()
    if kernel32.ProcessIdToSessionId(ctypes.c_ulong(pid), ctypes.byref(session_id)) == 0:
        return None
    return int(session_id.value)


def get_process_identity(pid: int) -> ProcessIdentity | None:
    """Resolve stable identity fields once so a sample series is attributable."""
    if PSUTIL is None:
        return None

    try:
        proc = PSUTIL.Process(pid)
        with proc.oneshot():
            name = proc.name()
            try:
                cmdline = " ".join(proc.cmdline())
            except Exception:
                cmdline = None
            parent_pid = proc.ppid()
            created = proc.create_time()
    except Exception:
        return None

    create_time_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    return ProcessIdentity(
        pid=pid,
        name=name,
        cmdline=cmdline,
        parent_pid=parent_pid,
        session_id=get_process_session_id(pid),
        create_time_utc=create_time_utc,
    )


def get_process_affinity(pid: int) -> tuple[str | None, int | None]:
    """Return (hex mask, cpu count) so affinity changes are visible per sample."""
    if PSUTIL is None:
        return None, None
    try:
        cpus = PSUTIL.Process(pid).cpu_affinity()
    except Exception:
        return None, None

    mask = 0
    for cpu in cpus:
        mask |= 1 << int(cpu)
    return f"0x{mask:x}", len(cpus)


def is_process_alive(pid: int) -> bool:
    if PSUTIL is None:
        return False
    try:
        return bool(PSUTIL.pid_exists(pid))
    except Exception:
        return False


def get_foreground_pid() -> int | None:
    user32 = ctypes.WinDLL("user32")
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    owner_pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
    return int(owner_pid.value) or None


class _ProcessPowerThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_uint32),
        ("ControlMask", ctypes.c_uint32),
        ("StateMask", ctypes.c_uint32),
    ]


def get_process_power_throttling(pid: int) -> str | None:
    """Return EcoQoS execution-speed state: throttled, unthrottled, or system-managed."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None

    state = _ProcessPowerThrottlingState()
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
    try:
        ok = kernel32.GetProcessInformation(
            ctypes.c_void_p(handle),
            PROCESS_INFORMATION_CLASS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
    except (OSError, AttributeError):
        ok = 0
    finally:
        kernel32.CloseHandle(handle)

    if not ok:
        return None
    if not state.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED:
        return "system-managed"
    if state.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED:
        return "throttled"
    return "unthrottled"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def get_commit_charge_mb() -> tuple[int | None, int | None]:
    """Return (commit_used_mb, commit_limit_mb) from the system commit charge."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    limit_mb = status.ullTotalPageFile // (1024 ** 2)
    used_mb = (status.ullTotalPageFile - status.ullAvailPageFile) // (1024 ** 2)
    return used_mb, limit_mb


def get_page_fault_rate(pid: int, now: float) -> float | None:
    """Return page faults per second for the target process since the previous sample."""
    global _LAST_PAGE_FAULTS
    if PSUTIL is None:
        return None
    try:
        faults = int(PSUTIL.Process(pid).memory_info().num_page_faults)
    except Exception:
        return None

    previous = _LAST_PAGE_FAULTS
    _LAST_PAGE_FAULTS = (faults, now)
    if previous is None:
        return None
    delta_faults = faults - previous[0]
    delta_time = now - previous[1]
    if delta_time <= 0 or delta_faults < 0:
        return None
    return delta_faults / delta_time


def get_top_cpu_processes(now: float, top_n: int) -> list[dict[str, object]] | None:
    """Return the busiest processes since the previous sample, excluding this monitor."""
    global _LAST_PROCESS_CPU, _LAST_PROCESS_CPU_TIME
    if PSUTIL is None or top_n <= 0:
        return None

    current: dict[int, tuple[str, float]] = {}
    self_pid = os.getpid()
    for proc in PSUTIL.process_iter(["pid", "name", "cpu_times"]):
        try:
            pid = int(proc.info["pid"])
            # pid 0 is System Idle Process and would always dominate the ranking.
            if pid in (0, self_pid):
                continue
            times = proc.info["cpu_times"]
            if times is None:
                continue
            current[pid] = (proc.info["name"] or "", float(times.user) + float(times.system))
        except (PSUTIL.NoSuchProcess, PSUTIL.AccessDenied, PSUTIL.ZombieProcess, TypeError, ValueError):
            continue

    previous = _LAST_PROCESS_CPU
    previous_time = _LAST_PROCESS_CPU_TIME
    _LAST_PROCESS_CPU = current
    _LAST_PROCESS_CPU_TIME = now

    if previous_time is None:
        return None
    elapsed = now - previous_time
    if elapsed <= 0:
        return None

    cpu_count = PSUTIL.cpu_count() or 1
    deltas: list[tuple[float, int, str]] = []
    for pid, (name, cpu_time) in current.items():
        before = previous.get(pid)
        if before is None:
            continue
        delta = cpu_time - before[1]
        if delta <= 0:
            continue
        deltas.append((100.0 * delta / elapsed / cpu_count, pid, name))

    deltas.sort(reverse=True)
    return [
        {"pid": pid, "name": name, "cpu_percent": round(percent, 2)}
        for percent, pid, name in deltas[:top_n]
    ]


def detect_intel_gpu_name(device_index: int = 0) -> str | None:
    global _INTEL_GPU_DETECTED
    if _INTEL_GPU_DETECTED:
        return _INTEL_GPU_NAMES.get(device_index)

    _INTEL_GPU_DETECTED = True
    conn = get_wmi_connection()
    if conn is None:
        return None

    try:
        devices = conn.Win32_VideoController()
    except Exception:
        devices = []

    intel_names: list[str] = []
    for device in devices:
        name = str(getattr(device, "Name", ""))
        compat = str(getattr(device, "AdapterCompatibility", ""))
        pnp = str(getattr(device, "PNPDeviceID", ""))
        text = f"{name} {compat} {pnp}".lower()
        if "intel" in text or "ven_8086" in text:
            intel_names.append(name or compat or "Intel GPU")

    for index, name in enumerate(intel_names):
        _INTEL_GPU_NAMES[index] = name
    return _INTEL_GPU_NAMES.get(device_index)


ZE_BOOL = ctypes.c_uint8
ZES_FREQ_DOMAIN_GPU: Final[int] = 0
# COMPUTE_ALL is the only group worth polling: ALL averages over every engine
# instance, so a saturated compute workload reads as ~1/N.
ZES_ENGINE_GROUP_ALL: Final[int] = 0
ZES_ENGINE_GROUP_COMPUTE_ALL: Final[int] = 1


class _ZesFreqProperties(ctypes.Structure):
    # ze_bool_t is uint8; widening these fields silently shifts min/max.
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("onSubdevice", ZE_BOOL),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ZE_BOOL),
        ("isThrottleEventSupported", ZE_BOOL),
        ("min", ctypes.c_double),
        ("max", ctypes.c_double),
    ]


class _ZesFreqState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("currentVoltage", ctypes.c_double),
        ("request", ctypes.c_double),
        ("tdp", ctypes.c_double),
        ("efficient", ctypes.c_double),
        ("actual", ctypes.c_double),
        ("throttleReasons", ctypes.c_uint32),
    ]


class _ZesPowerEnergyCounter(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("energy", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
    ]


class _ZesMemState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_uint32),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ZesEngineStats(ctypes.Structure):
    _fields_ = [("activeTime", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class _ZesEngineProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("onSubdevice", ZE_BOOL),
        ("subdeviceId", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class SysmanHandles:
    lib: object
    freq: int | None
    power: int | None
    memory: int | None
    engine: int | None
    engine_group: str | None
    max_clock_mhz: float | None


def _enumerate_sysman(lib: object, func_name: str, device: int) -> list[int]:
    func = getattr(lib, func_name)
    count = ctypes.c_uint32(0)
    if func(ctypes.c_void_p(device), ctypes.byref(count), None) != 0 or count.value == 0:
        return []
    handles = (ctypes.c_void_p * count.value)()
    if func(ctypes.c_void_p(device), ctypes.byref(count), handles) != 0:
        return []
    return [handles[i] for i in range(count.value)]


def init_sysman(device_index: int = 0) -> SysmanHandles | None:
    """Intel GPU telemetry via Level Zero Sysman; needs no external tooling or elevation."""
    try:
        lib = ctypes.CDLL("ze_loader.dll")
    except OSError:
        return None
    if lib.zesInit(0) != 0:
        return None

    driver_count = ctypes.c_uint32(0)
    if lib.zesDriverGet(ctypes.byref(driver_count), None) != 0 or driver_count.value == 0:
        return None
    drivers = (ctypes.c_void_p * driver_count.value)()
    if lib.zesDriverGet(ctypes.byref(driver_count), drivers) != 0:
        return None

    devices: list[int] = []
    for i in range(driver_count.value):
        count = ctypes.c_uint32(0)
        if lib.zesDeviceGet(ctypes.c_void_p(drivers[i]), ctypes.byref(count), None) != 0:
            continue
        if count.value == 0:
            continue
        handles = (ctypes.c_void_p * count.value)()
        if lib.zesDeviceGet(ctypes.c_void_p(drivers[i]), ctypes.byref(count), handles) != 0:
            continue
        devices.extend(handles[j] for j in range(count.value))

    if device_index < 0 or device_index >= len(devices):
        return None
    device = devices[device_index]

    freq_handle: int | None = None
    max_clock: float | None = None
    for handle in _enumerate_sysman(lib, "zesDeviceEnumFrequencyDomains", device):
        props = _ZesFreqProperties()
        props.stype = 0x9
        if lib.zesFrequencyGetProperties(ctypes.c_void_p(handle), ctypes.byref(props)) != 0:
            continue
        if props.type == ZES_FREQ_DOMAIN_GPU:
            freq_handle = handle
            max_clock = props.max
            break

    engine_handle: int | None = None
    engine_group: str | None = None
    for handle in _enumerate_sysman(lib, "zesDeviceEnumEngineGroups", device):
        props = _ZesEngineProperties()
        if lib.zesEngineGetProperties(ctypes.c_void_p(handle), ctypes.byref(props)) != 0:
            continue
        if props.type == ZES_ENGINE_GROUP_COMPUTE_ALL:
            engine_handle, engine_group = handle, "compute"
            break
        if props.type == ZES_ENGINE_GROUP_ALL and engine_handle is None:
            engine_handle, engine_group = handle, "all"

    power_handles = _enumerate_sysman(lib, "zesDeviceEnumPowerDomains", device)
    memory_handles = _enumerate_sysman(lib, "zesDeviceEnumMemoryModules", device)

    return SysmanHandles(
        lib=lib,
        freq=freq_handle,
        power=power_handles[0] if power_handles else None,
        memory=memory_handles[0] if memory_handles else None,
        engine=engine_handle,
        engine_group=engine_group,
        max_clock_mhz=max_clock,
    )


def get_sysman(device_index: int = 0) -> SysmanHandles | None:
    global _LAST_ENERGY, _LAST_ENGINE, _SYSMAN, _SYSMAN_DEVICE_INDEX, _SYSMAN_RESOLVED
    if _SYSMAN_DEVICE_INDEX != device_index:
        _SYSMAN = None
        _SYSMAN_RESOLVED = False
        _SYSMAN_DEVICE_INDEX = device_index
        _LAST_ENERGY = None
        _LAST_ENGINE = None
    if not _SYSMAN_RESOLVED:
        _SYSMAN_RESOLVED = True
        _SYSMAN = init_sysman(device_index)
        if _SYSMAN is None:
            LOGGER.warning(
                "Level Zero Sysman unavailable for device index %d; GPU telemetry will be skipped.",
                device_index,
            )
    return _SYSMAN


def get_gpu_metrics(device_index: int = 0) -> dict[str, object]:
    """Return Intel GPU telemetry: clock, throttle, utilization, memory and power."""
    global _LAST_ENERGY, _LAST_ENGINE

    metrics: dict[str, object] = {
        "gpu_device_index": device_index,
        "gpu_name": None,
        "gpu_source": None,
        "gpu_clock_mhz": None,
        "gpu_clock_request_mhz": None,
        "gpu_clock_max_mhz": None,
        "gpu_throttle_reasons": None,
        "gpu_power_watts": None,
        "gpu_utilization_percent": None,
        "gpu_utilization_source": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "lhm_gpu_power_watts": None,
        "lhm_gpu_temp_c": None,
        "lhm_gpu_memory_clock_mhz": None,
        "lhm_gpu_fan_rpm": None,
        "lhm_gpu_sample_valid": None,
        "lhm_gpu_query_duration_ms": None,
    }

    metrics["gpu_name"] = detect_intel_gpu_name(device_index)
    sysman = get_sysman(device_index)
    if sysman is None:
        metrics["gpu_source"] = "level-zero-unavailable"
    else:
        metrics["gpu_source"] = "level-zero-sysman"

        lib = sysman.lib
        metrics["gpu_clock_max_mhz"] = sysman.max_clock_mhz

        if sysman.freq is not None:
            state = _ZesFreqState()
            if lib.zesFrequencyGetState(ctypes.c_void_p(sysman.freq), ctypes.byref(state)) == 0:
                metrics["gpu_clock_mhz"] = state.actual if state.actual >= 0 else None
                metrics["gpu_clock_request_mhz"] = state.request if state.request >= 0 else None
                metrics["gpu_throttle_reasons"] = f"0x{state.throttleReasons:x}"

        if sysman.power is not None:
            counter = _ZesPowerEnergyCounter()
            if lib.zesPowerGetEnergyCounter(ctypes.c_void_p(sysman.power), ctypes.byref(counter)) == 0:
                previous = _LAST_ENERGY
                _LAST_ENERGY = (counter.energy, counter.timestamp)
                if previous is not None:
                    # energy is microjoules and timestamp microseconds, so the ratio is watts.
                    delta_energy = counter.energy - previous[0]
                    delta_time = counter.timestamp - previous[1]
                    if delta_time > 0 and delta_energy >= 0:
                        metrics["gpu_power_watts"] = delta_energy / delta_time

        if sysman.engine is not None:
            metrics["gpu_utilization_source"] = sysman.engine_group
            stats = _ZesEngineStats()
            if lib.zesEngineGetActivity(ctypes.c_void_p(sysman.engine), ctypes.byref(stats)) == 0:
                previous = _LAST_ENGINE
                _LAST_ENGINE = (stats.activeTime, stats.timestamp)
                if previous is not None:
                    delta_active = stats.activeTime - previous[0]
                    delta_time = stats.timestamp - previous[1]
                    if delta_time > 0 and delta_active >= 0:
                        metrics["gpu_utilization_percent"] = min(100.0, 100.0 * delta_active / delta_time)

        if sysman.memory is not None:
            mem = _ZesMemState()
            if lib.zesMemoryGetState(ctypes.c_void_p(sysman.memory), ctypes.byref(mem)) == 0 and mem.size > 0:
                metrics["gpu_memory_total_mb"] = mem.size // (1024 ** 2)
                metrics["gpu_memory_used_mb"] = (mem.size - mem.free) // (1024 ** 2)

    lhm, lhm_duration_ms = get_lhm_gpu_metrics(metrics["gpu_name"] if isinstance(metrics["gpu_name"], str) else None)
    metrics["lhm_gpu_query_duration_ms"] = lhm_duration_ms
    metrics["lhm_gpu_sample_valid"] = lhm is not None
    if lhm is not None:
        metrics["lhm_gpu_power_watts"] = lhm.power_watts
        metrics["lhm_gpu_temp_c"] = lhm.temperature_c
        metrics["lhm_gpu_memory_clock_mhz"] = lhm.memory_clock_mhz
        metrics["lhm_gpu_fan_rpm"] = lhm.fan_rpm

    return metrics


def iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"


def resolve_target_pid(pid: int | None, process_name: str | None) -> int | None:
    if pid is not None:
        return pid
    if not process_name:
        return None

    if PSUTIL is None:
        return None

    target_name = process_name.lower().removesuffix(".exe")
    newest_pid: int | None = None
    newest_time = -1.0
    for proc in PSUTIL.process_iter(["name", "create_time", "pid"]):
        try:
            name = (proc.info.get("name") or "").lower().removesuffix(".exe")
            if name != target_name:
                continue
            created = float(proc.info.get("create_time") or 0.0)
            if created > newest_time:
                newest_time = created
                newest_pid = int(proc.info["pid"])
        except (PSUTIL.NoSuchProcess, PSUTIL.AccessDenied, PSUTIL.ZombieProcess, ValueError, TypeError, KeyError):
            continue

    return newest_pid


def parse_gpu_device_index(target_device: str | None) -> int:
    if target_device is None or not target_device.strip():
        return 0

    normalized = target_device.strip().upper()
    if normalized == "GPU":
        return 0

    match = re.fullmatch(r"GPU\.(\d+)", normalized)
    if match is None:
        raise ValueError(f"Unsupported target device: {target_device}. Expected GPU, GPU.0, GPU.1, ...")
    return int(match.group(1))


def sample_once(
    target_pid: int | None,
    host_mem_info: HostMemoryInfo,
    identity: ProcessIdentity | None = None,
    top_processes: int = 0,
    temp_interval_sec: float = 5.0,
    force_temp: bool = False,
    gpu_device_index: int = 0,
) -> dict[str, object]:
    # Stamp before probing so sample times stay aligned regardless of probe cost.
    timestamp_utc = iso_utc_now()
    t_monotonic = time.perf_counter()
    reset_lhm_payload_cache()

    timer = query_timer_resolution()
    cpu_clock = get_cpu_clock_mhz()
    cpu_temps, cpu_temp_age_ms, cpu_temp_query_duration_ms = get_cpu_temp_cached(
        t_monotonic, temp_interval_sec, force_temp
    )
    cpu_usage, cpu_usage_age_ms = get_cpu_usage_windowed(t_monotonic)
    t_gpu_start = time.perf_counter()
    gpu = get_gpu_metrics(gpu_device_index)
    gpu_query_duration_ms = (time.perf_counter() - t_gpu_start) * 1000.0
    mem_usage, mem_available_mb = get_memory_usage()

    proc = get_process_priority(target_pid) if target_pid is not None else None
    affinity_mask, affinity_count = (
        get_process_affinity(target_pid) if target_pid is not None else (None, None)
    )
    foreground_pid = get_foreground_pid()
    commit_used_mb, commit_limit_mb = get_commit_charge_mb()
    power_throttling = get_process_power_throttling(target_pid) if target_pid is not None else None
    page_fault_rate = get_page_fault_rate(target_pid, t_monotonic) if target_pid is not None else None
    top_cpu = get_top_cpu_processes(t_monotonic, top_processes)

    record: dict[str, object] = {
        "timestamp_utc": timestamp_utc,
        "t_monotonic": t_monotonic,
        "cpu_clock_mhz": cpu_clock,
        "cpu_temp_c": cpu_temps.overall_c if cpu_temps else None,
        "cpu_temp_age_ms": cpu_temp_age_ms,
        "cpu_temp_core_max_c": cpu_temps.core_max_c if cpu_temps else None,
        "cpu_temp_core_avg_c": cpu_temps.core_avg_c if cpu_temps else None,
        "cpu_temp_core_values_c": cpu_temps.core_temps_c if cpu_temps else None,
        "cpu_temp_source": cpu_temps.source if cpu_temps else None,
        "cpu_temp_query_duration_ms": cpu_temp_query_duration_ms,
        "cpu_usage_percent": cpu_usage,
        "cpu_usage_age_ms": cpu_usage_age_ms,
        **gpu,
        "gpu_query_duration_ms": gpu_query_duration_ms,
        "host_memory_speed_mts": host_mem_info.speed_mts,
        "host_memory_total_gb": host_mem_info.total_gb,
        "host_memory_usage_percent": mem_usage,
        "host_memory_available_mb": mem_available_mb,
        "host_commit_used_mb": commit_used_mb,
        "host_commit_limit_mb": commit_limit_mb,
        "process_pid": proc.pid if proc else target_pid,
        "process_name": identity.name if identity else None,
        "process_cmdline": identity.cmdline if identity else None,
        "process_parent_pid": identity.parent_pid if identity else None,
        "process_session_id": identity.session_id if identity else None,
        "process_create_time_utc": identity.create_time_utc if identity else None,
        "process_alive": is_process_alive(target_pid) if target_pid is not None else None,
        "process_priority_class": proc.class_name if proc else None,
        "process_priority_class_value": proc.class_value if proc else None,
        "process_cpu_affinity_mask": affinity_mask,
        "process_cpu_affinity_count": affinity_count,
        "process_power_throttling": power_throttling,
        "process_page_faults_per_sec": page_fault_rate,
        "top_cpu_processes": top_cpu,
        "foreground_pid": foreground_pid,
        "process_is_foreground": (
            foreground_pid == target_pid if target_pid is not None and foreground_pid else None
        ),
        "timer_resolution_current_ms": timer.current_ms if timer else None,
        "timer_resolution_minimum_ms": timer.minimum_ms if timer else None,
        "timer_resolution_maximum_ms": timer.maximum_ms if timer else None,
    }
    # Driver calls occasionally stall; this lets analysis drop samples that took too long.
    record["sample_duration_ms"] = (time.perf_counter() - t_monotonic) * 1000.0
    return record


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stream_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor factors that can affect first-token latency on Windows.",
    )
    parser.add_argument("--pid", type=int, default=None, help="Target process PID.")
    parser.add_argument(
        "--process-name",
        type=str,
        default=None,
        help="Target process image name (e.g. python.exe). Used when --pid is absent.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Sampling duration in seconds.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("test_output") / "first_token_factors.jsonl",
        help="Output file path. Extension decides format: .jsonl or .csv.",
    )
    parser.add_argument(
        "--top-processes",
        type=int,
        nargs="?",
        const=5,
        default=0,
        metavar="N",
        help=(
            "Record the N busiest processes per sample (bare flag uses 5). Off by default: "
            "the scan walks every process and adds about 1 ms per sample plus occasional stalls."
        ),
    )
    parser.add_argument(
        "--temp-interval-sec",
        type=float,
        default=None,
        help=(
            "CPU temperature sampling period. Defaults to --interval-sec, which costs almost "
            "nothing because the sample already fetches the LibreHardwareMonitor payload. "
            "The first and last sample are always read."
        ),
    )
    parser.add_argument(
        "--target-device",
        type=str,
        default=os.environ.get("TARGET_DEVICE", "GPU"),
        help="GPU target device to monitor: GPU, GPU.0, GPU.1, ... (default: TARGET_DEVICE env or GPU).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    global PSUTIL, WMI_LIB
    try:
        PSUTIL, WMI_LIB = ensure_required_modules()
    except ModuleNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 3

    if args.interval_sec <= 0:
        LOGGER.error("--interval-sec must be > 0")
        return 2
    if args.duration_sec <= 0:
        LOGGER.error("--duration-sec must be > 0")
        return 2
    temp_interval_sec = args.interval_sec if args.temp_interval_sec is None else args.temp_interval_sec
    try:
        gpu_device_index = parse_gpu_device_index(args.target_device)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    target_pid = resolve_target_pid(args.pid, args.process_name)
    identity = get_process_identity(target_pid) if target_pid is not None else None
    host_mem_info = get_host_memory_info()
    LOGGER.info("Target PID: %s", target_pid)
    if identity is not None:
        LOGGER.info("Target process: %s (session %s)", identity.name, identity.session_id)
    LOGGER.info("Host memory speed (MT/s): %s", host_mem_info.speed_mts)
    LOGGER.info("Host memory total (GB): %s", host_mem_info.total_gb)
    LOGGER.info(
        "GPU telemetry: clock, throttle, utilization, memory, power (target=%s, device_index=%d)",
        args.target_device,
        gpu_device_index,
    )
    # The first Sysman/LHM query pays a one-off resolution cost; warm it up so it does not
    # land inside the first sample.
    get_gpu_metrics(gpu_device_index)

    try:
        rows: list[dict[str, object]] = []
        sample_index = 0
        end_time = time.monotonic() + args.duration_sec
        suffix = args.out.suffix.lower()
        stream_mode = suffix != ".csv"

        if stream_mode:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            # Clear existing file first to avoid mixing multiple runs.
            args.out.write_text("", encoding="utf-8")

        while True:
            now = time.monotonic()
            if now > end_time:
                break

            is_first = not rows and sample_index == 0
            is_last = now + args.interval_sec > end_time
            row = sample_once(
                target_pid,
                host_mem_info,
                identity,
                args.top_processes,
                temp_interval_sec,
                force_temp=is_first or is_last,
                gpu_device_index=gpu_device_index,
            )
            sample_index += 1
            if stream_mode:
                stream_jsonl(args.out, row)
            else:
                rows.append(row)

            next_wakeup = now + args.interval_sec
            sleep_sec = max(0.0, next_wakeup - time.monotonic())
            time.sleep(sleep_sec)

        if suffix == ".csv":
            write_csv(args.out, rows)

        if suffix == ".csv":
            sample_count = len(rows)
        else:
            with args.out.open("r", encoding="utf-8") as handle:
                sample_count = sum(1 for _ in handle)
        LOGGER.info("Wrote %d samples to %s", sample_count, args.out)
    finally:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
