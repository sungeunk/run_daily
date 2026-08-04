"""
Benchmark Environment Setup Script

SSH 환경에서 RDP/로컬과 동일한 스케줄링 조건을 만들기 위한 스크립트.
관리자 권한으로 실행 필요 (set/set_max/restore 명령).

Usage:
    python benchmark_env_setup.py status    - 현재 설정 출력
    python benchmark_env_setup.py set       - 로컬과 동일 조건 설정
    python benchmark_env_setup.py set_max   - set + Timer Resolution 1ms
    python benchmark_env_setup.py restore   - 원래 설정 복원 (sleep timeout 제외)

set이 적용하는 것 (SSH에서 로컬/RDP와 동일 조건):
    - High Performance 전원 계획
    - Core Parking 비활성화 (min cores 100%)
    - Min Processor State 100%
    - Sleep / hibernate timeout을 never로 (AC/DC 모두)
    - System sleep 억제 holder (측정 중 세션이 끊기는 것을 막는다)
    - Windows Defender 실시간 보호 해제 (스캔이 GPU 커널 시간까지 왜곡한다)

**sleep timeout은 restore도 never로 유지한다.** 복원하지 않는 유일한 항목이다. 이 머신은
SSH로만 접근하므로 절전에 들어가면 세션이 끊기고 머신을 제어할 수 없게 된다. 다음 측정이
아니라 복구 자체가 불가능해지는 문제이므로, 되돌리지 않는 쪽이 안전하다. 원래 값은
backup 파일의 `*_original` 키에 기록만 남긴다.

set_max가 추가로 적용하는 것:
    - Timer Resolution 1ms (background holder 프로세스로 유지)

Timer Resolution은 요청한 프로세스가 종료되면 OS가 되돌리므로, set_max는 이 스크립트를
background로 다시 띄워 값을 유지한다. restore가 그 프로세스를 종료시킨다.
set_max를 여러 번 실행해도 named mutex로 중복 실행을 막는다.
"""

import sys
import os
import json
import re
import ctypes
import ctypes.wintypes
import subprocess
import time
import winreg


BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark-env-backup.json")

# Power plan GUIDs
POWER_HIGH_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

# Processor subgroup and settings GUIDs
SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
PROCTHROTTLEMIN = "893dee8e-2bef-41e0-89c6-b55d0929964c"
PROCTHROTTLEMAX = "bc5038f7-23e0-4960-96da-33abaf5935ec"
CPMINCORES = "0cc5b647-c1df-4637-891a-dec35c318583"

# Sleep subgroup. The execution-state holder suppresses sleep only while it runs;
# these timeouts are the scheme policy underneath it. Setting them to 0 (never)
# means a killed holder cannot leave the machine sleeping mid-measurement.
#
# Note the settings UI's "Sleep after N minutes" maps to STANDBYIDLE, while the
# separate "Turn my screen off after" is VIDEOIDLE. On DUT2434-NVLS the active
# schemes already had STANDBYIDLE=0 and VIDEOIDLE=600 (10 min) - a display
# timeout, not sleep. Screen-off alone does not stop a benchmark.
SUB_SLEEP = "238c9fa8-0aad-41ed-83f4-97be242c8f20"
STANDBYIDLE = "29f6c1db-86da-48c5-9fdb-f2b67b1f44da"
HIBERNATEIDLE = "9d7815a6-7ee4-497e-8888-515a05f02364"

# Timer resolution은 설정한 프로세스가 종료되면 OS가 되돌린다. 따라서 값을 유지하려면
# 프로세스가 살아 있어야 하고, set은 background holder를 띄운다.
#
# 중복 방지는 named mutex로 한다. PID 파일보다 견고하다: holder가 어떤 이유로 죽어도
# OS가 mutex를 자동 해제하므로 stale 상태가 남지 않는다.
# SSH는 session 0에서 동작하므로 "Global\" prefix가 필요하다.
TIMER_HOLDER_MUTEX = "Global\\BenchmarkEnvSetup_TimerResolutionHolder"
TIMER_HOLDER_ARG = "--hold-timer-resolution"
TIMER_HOLDER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "benchmark-env-timer-holder.log")

AWAKE_HOLDER_MUTEX = "Global\\BenchmarkEnvSetup_AwakeHolder"
AWAKE_HOLDER_ARG = "--hold-system-awake"
AWAKE_HOLDER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "benchmark-env-awake-holder.log")

ERROR_ALREADY_EXISTS = 183
SYNCHRONIZE = 0x00100000
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040
ES_CONTINUOUS = 0x80000000

# GPU frequency backup file
GPU_BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark-env-gpu-backup.json")
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

def run_powercfg(*args):
    result = subprocess.run(
        ["powercfg"] + list(args),
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.stdout.strip(), result.returncode


def list_power_schemes():
    """활성 여부와 무관하게 존재하는 모든 전원 계획 GUID를 반환한다."""
    out, rc = run_powercfg("/list")
    if rc != 0:
        return []
    return re.findall(r"Power Scheme GUID:\s*([0-9a-fA-F-]{36})", out)


def disable_sleep_all_schemes():
    """모든 전원 계획에서 sleep/hibernate timeout을 never로 만든다.

    활성 스킴만 바꾸면 나중에 다른 스킴(예: Power Saver)으로 전환됐을 때 다시 절전에
    들어간다. SSH 전용 머신에서는 그 순간 제어가 불가능해지므로 전 스킴에 적용한다.
    """
    failed = False
    schemes = list_power_schemes() or ["SCHEME_CURRENT"]
    for scheme in schemes:
        for setting in (STANDBYIDLE, HIBERNATEIDLE):
            for fn in ("/setacvalueindex", "/setdcvalueindex"):
                _, rc = run_powercfg(fn, scheme, SUB_SLEEP, setting, "0")
                failed |= rc != 0
    _, rc = run_powercfg("/setactive", "SCHEME_CURRENT")
    failed |= rc != 0
    return len(schemes), failed


def get_active_scheme():
    out, _ = run_powercfg("/getactivescheme")
    # "Power Scheme GUID: 381b4222-... (Balanced)"
    parts = out.split()
    guid_idx = parts.index("GUID:") + 1 if "GUID:" in parts else None
    if guid_idx:
        guid = parts[guid_idx]
        name = out.split("(")[-1].rstrip(")") if "(" in out else "Unknown"
        return guid, name
    return None, None


def query_power_setting(subgroup, setting):
    out, rc = run_powercfg("/query", "SCHEME_CURRENT", subgroup, setting)
    if rc != 0:
        return None, None
    ac_value = None
    dc_value = None
    for line in out.splitlines():
        line_lower = line.lower()
        if "current ac power setting index" in line_lower:
            ac_value = int(line.strip().split(":")[-1].strip(), 0)
        elif "current dc power setting index" in line_lower:
            dc_value = int(line.strip().split(":")[-1].strip(), 0)
    return ac_value, dc_value


def get_win32_priority_separation():
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\PriorityControl"
        )
        value, _ = winreg.QueryValueEx(key, "Win32PrioritySeparation")
        winreg.CloseKey(key)
        return value
    except OSError:
        return None


def get_timer_resolution():
    ntdll = ctypes.WinDLL("ntdll")
    min_res = ctypes.c_uint32()
    max_res = ctypes.c_uint32()
    cur_res = ctypes.c_uint32()
    ntdll.NtQueryTimerResolution(
        ctypes.byref(min_res), ctypes.byref(max_res), ctypes.byref(cur_res)
    )
    return min_res.value, max_res.value, cur_res.value


def set_timer_resolution(requested_100ns):
    """NtSetTimerResolution으로 타이머 해상도를 요청한다.

    반환값은 (성공여부, 실제 적용된 100ns 단위 값).
    이 설정은 호출한 프로세스가 살아 있는 동안만 유지된다.
    """
    ntdll = ctypes.WinDLL("ntdll")
    actual = ctypes.c_uint32()
    # NtSetTimerResolution(DesiredResolution, SetResolution=TRUE, CurrentResolution)
    status = ntdll.NtSetTimerResolution(
        ctypes.c_uint32(requested_100ns), ctypes.c_bool(True), ctypes.byref(actual)
    )
    return status == 0, actual.value


# ---------------------------------------------------------------------------
# GPU Frequency Control (Level Zero Sysman API)
# ---------------------------------------------------------------------------

class _ZesFreqProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),       # 0=GPU, 1=MEMORY
        ("onSubdevice", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("canControl", ctypes.c_uint32),
        ("isThrottleEventSupported", ctypes.c_uint32),
        ("min", ctypes.c_double),
        ("max", ctypes.c_double),
    ]


class _ZesFreqRange(ctypes.Structure):
    _fields_ = [("min", ctypes.c_double), ("max", ctypes.c_double)]


def _load_level_zero():
    """Level Zero loader를 로드한다. 없으면 None."""
    try:
        return ctypes.CDLL("ze_loader.dll")
    except OSError:
        return None


def get_gpu_frequency_info():
    """GPU frequency domain 정보를 조회한다.

    반환: list of dict (domain별) 또는 None (L0 없음)
    각 dict: {domain, can_control, hw_min, hw_max, cur_min, cur_max, handle, device}
    """
    ze = _load_level_zero()
    if not ze:
        return None

    # zesInit을 먼저 호출해야 한다. 이 순서를 어기면 zesDriverGet이
    # ZE_RESULT_ERROR_UNSUPPORTED_FEATURE(0x78000001)를 반환해 이 함수가 None을 돌려주고,
    # 호출측은 "Level Zero not available"로 오해한다. set_gpu_frequency_range()는 이미
    # 이 순서를 지키고 있어서, 그쪽만 동작하고 조회는 실패하는 상태였다.
    if ze.zesInit(0) != 0:
        return None

    # Sysman driver/device
    drv_count = ctypes.c_uint32(0)
    if ze.zesDriverGet(ctypes.byref(drv_count), None) != 0:
        return None
    if drv_count.value == 0:
        return None
    drivers = (ctypes.c_void_p * drv_count.value)()
    if ze.zesDriverGet(ctypes.byref(drv_count), drivers) != 0:
        return None

    dev_count = ctypes.c_uint32(0)
    if ze.zesDeviceGet(ctypes.c_void_p(drivers[0]), ctypes.byref(dev_count), None) != 0:
        return None
    if dev_count.value == 0:
        return None
    devices = (ctypes.c_void_p * dev_count.value)()
    if ze.zesDeviceGet(ctypes.c_void_p(drivers[0]), ctypes.byref(dev_count), devices) != 0:
        return None

    # enumerate frequency domains
    freq_count = ctypes.c_uint32(0)
    rc = ze.zesDeviceEnumFrequencyDomains(
        ctypes.c_void_p(devices[0]), ctypes.byref(freq_count), None
    )
    if rc != 0 or freq_count.value == 0:
        return None
    freq_handles = (ctypes.c_void_p * freq_count.value)()
    ze.zesDeviceEnumFrequencyDomains(
        ctypes.c_void_p(devices[0]), ctypes.byref(freq_count), freq_handles
    )

    results = []
    domain_names = {0: "GPU", 1: "MEMORY"}
    for i in range(freq_count.value):
        props = _ZesFreqProperties()
        props.stype = 0x9  # ZES_STRUCTURE_TYPE_FREQ_PROPERTIES
        ze.zesFrequencyGetProperties(ctypes.c_void_p(freq_handles[i]), ctypes.byref(props))

        cur_range = _ZesFreqRange()
        ze.zesFrequencyGetRange(ctypes.c_void_p(freq_handles[i]), ctypes.byref(cur_range))

        results.append({
            "domain": domain_names.get(props.type, f"type{props.type}"),
            "can_control": bool(props.canControl),
            "hw_min": props.min,
            "hw_max": props.max,
            "cur_min": cur_range.min,
            "cur_max": cur_range.max,
            "handle": freq_handles[i],
            "device": devices[0],
        })
    return results


def set_gpu_frequency_range(freq_min, freq_max):
    """GPU frequency range를 설정한다. (min=max로 하면 고정)

    반환: (성공여부, 설정 후 실제 range)
    """
    ze = _load_level_zero()
    if not ze:
        return False, None

    if ze.zesInit(0) != 0:
        return False, None

    drv_count = ctypes.c_uint32(0)
    if ze.zesDriverGet(ctypes.byref(drv_count), None) != 0 or drv_count.value == 0:
        return False, None
    drivers = (ctypes.c_void_p * drv_count.value)()
    if ze.zesDriverGet(ctypes.byref(drv_count), drivers) != 0:
        return False, None

    dev_count = ctypes.c_uint32(0)
    if (ze.zesDeviceGet(ctypes.c_void_p(drivers[0]), ctypes.byref(dev_count), None) != 0
            or dev_count.value == 0):
        return False, None
    devices = (ctypes.c_void_p * dev_count.value)()
    if ze.zesDeviceGet(ctypes.c_void_p(drivers[0]), ctypes.byref(dev_count), devices) != 0:
        return False, None

    freq_count = ctypes.c_uint32(0)
    ze.zesDeviceEnumFrequencyDomains(
        ctypes.c_void_p(devices[0]), ctypes.byref(freq_count), None
    )
    if freq_count.value == 0:
        return False, None
    freq_handles = (ctypes.c_void_p * freq_count.value)()
    ze.zesDeviceEnumFrequencyDomains(
        ctypes.c_void_p(devices[0]), ctypes.byref(freq_count), freq_handles
    )

    # Find GPU domain (type=0)
    gpu_handle = None
    for i in range(freq_count.value):
        props = _ZesFreqProperties()
        props.stype = 0x9
        ze.zesFrequencyGetProperties(ctypes.c_void_p(freq_handles[i]), ctypes.byref(props))
        if props.type == 0 and props.canControl:
            gpu_handle = freq_handles[i]
            break

    if gpu_handle is None:
        return False, None

    # Set range
    new_range = _ZesFreqRange(min=freq_min, max=freq_max)
    rc = ze.zesFrequencySetRange(ctypes.c_void_p(gpu_handle), ctypes.byref(new_range))

    # Read back
    actual = _ZesFreqRange()
    ze.zesFrequencyGetRange(ctypes.c_void_p(gpu_handle), ctypes.byref(actual))

    return rc == 0, (actual.min, actual.max)


def is_holder_running(mutex_name):
    """named mutex의 존재로 holder 생존을 확인한다.

    OpenMutex가 성공하면 누군가 mutex를 쥐고 있다는 뜻이다. holder가 죽으면 커널이
    핸들을 정리하므로 stale 판정이 생기지 않는다.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = ctypes.wintypes.HANDLE
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, mutex_name)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return False


def find_holder_pids(holder_arg):
    """holder 프로세스의 PID를 찾는다 (stop용).

    커맨드라인에 TIMER_HOLDER_ARG를 가진 python 프로세스를 찾는다. 자기 자신은 제외한다.
    """
    query = (
        "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{holder_arg}*' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    pids = []
    for line in result.stdout.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != me:
            pids.append(pid)
    return pids


def start_timer_holder(target_100ns):
    """timer resolution을 유지하는 background 프로세스를 띄운다.

    이미 실행 중이면 아무것도 하지 않는다 (set을 여러 번 호출해도 중복되지 않음).
    반환값: "started" | "already-running" | "failed"
    """
    if is_holder_running(TIMER_HOLDER_MUTEX):
        return "already-running"

    # DETACHED_PROCESS: 부모가 끝나도 살아남는다. SSH 세션이 끊겨도 유지된다.
    # CREATE_BREAKAWAY_FROM_JOB: 부모가 job object에 묶여 있으면 함께 종료되는 것을 막는다.
    flags = DETACHED_PROCESS | CREATE_NO_WINDOW
    try:
        flags |= CREATE_BREAKAWAY_FROM_JOB
        with open(TIMER_HOLDER_LOG, "a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 TIMER_HOLDER_ARG, str(target_100ns)],
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=flags, close_fds=True,
            )
    except OSError:
        # job object가 breakaway를 허용하지 않는 환경이면 플래그 없이 재시도한다.
        try:
            with open(TIMER_HOLDER_LOG, "a", encoding="utf-8") as log:
                subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__),
                     TIMER_HOLDER_ARG, str(target_100ns)],
                    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW, close_fds=True,
                )
        except OSError:
            return "failed"

    # mutex가 잡히기를 기다린다. 여기서 확인하지 않으면 set이 성공을 보고한 뒤
    # holder가 조용히 죽어도 알 수 없다.
    for _ in range(50):
        time.sleep(0.1)
        if is_holder_running(TIMER_HOLDER_MUTEX):
            return "started"
    return "failed"


def start_awake_holder():
    """system sleep을 막는 background 프로세스를 띄운다."""
    if is_holder_running(AWAKE_HOLDER_MUTEX):
        return "already-running"

    command = [sys.executable, os.path.abspath(__file__), AWAKE_HOLDER_ARG]
    for flags in (
        DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB,
        DETACHED_PROCESS | CREATE_NO_WINDOW,
    ):
        try:
            with open(AWAKE_HOLDER_LOG, "a", encoding="utf-8") as log:
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    creationflags=flags,
                    close_fds=True,
                )
            break
        except OSError:
            continue
    else:
        return "failed"

    for _ in range(50):
        time.sleep(0.1)
        if is_holder_running(AWAKE_HOLDER_MUTEX):
            return "started"
    return "failed"


def stop_timer_holder():
    """holder를 종료한다. 종료되면 OS가 timer resolution을 자동 복원한다.

    반환값: 종료시킨 프로세스 수 (-1은 실패)
    """
    if not is_holder_running(TIMER_HOLDER_MUTEX):
        return 0

    pids = find_holder_pids(TIMER_HOLDER_ARG)
    if not pids:
        # mutex는 있는데 PID를 못 찾은 경우. 다른 계정이 띄웠거나 조회가 막힌 상황이다.
        return -1

    killed = 0
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=20)
            killed += 1
        except (OSError, subprocess.SubprocessError):
            pass

    # mutex가 풀리는 것을 확인한다
    for _ in range(50):
        if not is_holder_running(TIMER_HOLDER_MUTEX):
            break
        time.sleep(0.1)
    return killed


def stop_awake_holder():
    """system sleep 억제 holder를 종료한다."""
    if not is_holder_running(AWAKE_HOLDER_MUTEX):
        return 0

    pids = find_holder_pids(AWAKE_HOLDER_ARG)
    if not pids:
        return -1

    killed = 0
    for pid in pids:
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=20
            )
            if result.returncode == 0:
                killed += 1
        except (OSError, subprocess.SubprocessError):
            pass

    for _ in range(50):
        if not is_holder_running(AWAKE_HOLDER_MUTEX):
            break
        time.sleep(0.1)
    return killed if not is_holder_running(AWAKE_HOLDER_MUTEX) else -1


def run_timer_holder(target_100ns):
    """background holder의 본체. timer resolution을 설정하고 계속 살아 있는다."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, True, TIMER_HOLDER_MUTEX)
    if not handle or ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        # 이미 다른 holder가 있다. 중복 실행이므로 조용히 물러난다.
        return 0

    ok, actual = set_timer_resolution(target_100ns)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] holder pid={os.getpid()} requested={target_100ns / 10000.0:.3f}ms "
          f"actual={actual / 10000.0:.3f}ms ok={ok}", flush=True)
    if not ok:
        kernel32.CloseHandle(handle)
        return 1

    # 종료될 때까지 대기한다. mutex를 쥐고 있는 동안 timer resolution이 유지된다.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        kernel32.CloseHandle(handle)
    return 0


def run_awake_holder():
    """background holder의 본체. system sleep 억제 요청을 계속 유지한다."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE
    kernel32.SetThreadExecutionState.argtypes = [ctypes.wintypes.DWORD]
    kernel32.SetThreadExecutionState.restype = ctypes.wintypes.DWORD
    handle = kernel32.CreateMutexW(None, True, AWAKE_HOLDER_MUTEX)
    if not handle or ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        return 0

    request = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    previous = kernel32.SetThreadExecutionState(request)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] awake holder pid={os.getpid()} active={previous != 0}", flush=True)
    if previous == 0:
        kernel32.CloseHandle(handle)
        return 1

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        kernel32.CloseHandle(handle)
    return 0


def run_powershell(script):
    """Run a PowerShell snippet and return (stdout, returncode)."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.stdout.strip(), result.returncode


# ---------------------------------------------------------------------------
# Windows Defender real-time protection
# ---------------------------------------------------------------------------

def get_defender_state():
    """Return True/False for real-time protection, or None if Defender is absent."""
    out, rc = run_powershell(
        "try { (Get-MpComputerStatus).RealTimeProtectionEnabled } catch { 'ERR' }"
    )
    if rc != 0 or "ERR" in out or not out:
        return None
    return out.strip().lower() == "true"


def set_defender_realtime(enable):
    """Enable or disable Defender real-time protection.

    Why this matters for benchmarking: a post-boot Defender scan inflated SDPA
    kernel time from 334 ms to 687-940 ms. SDPA is pure GPU kernel time and normally
    reproduces within 0.2%, so a 2-3x shift there means the whole measurement is
    void - CPU-side load does reach GPU kernel timings on this integrated part.

    Returns (ok, state_after).
    """
    flag = "$false" if enable else "$true"
    _, rc = run_powershell(
        f"try {{ Set-MpPreference -DisableRealtimeMonitoring {flag} -ErrorAction Stop; "
        f"'OK' }} catch {{ 'ERR: ' + $_.Exception.Message }}"
    )
    time.sleep(3)
    return rc == 0, get_defender_state()


def get_os_product_type():
    """1=Workstation, 2=Domain Controller, 3=Server"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\ProductOptions"
        )
        value, _ = winreg.QueryValueEx(key, "ProductType")
        winreg.CloseKey(key)
        if value == "WinNT":
            return 1  # Workstation
        return 3  # Server
    except OSError:
        return 1


def decode_w32ps(value):
    ql = (value >> 4) & 0x3
    fb = (value >> 2) & 0x3
    qt = value & 0x3

    ql_desc = {0: "System default", 1: "Variable", 2: "Variable", 3: "Fixed (equal for all)"}
    fb_desc = {0: "System default", 1: "+1 quantum unit boost", 2: "+2 quantum units boost", 3: "+3 quantum units boost (max)"}
    qt_desc = {0: "System default", 1: "Short (workstation)", 2: "Long (server)", 3: "Short (workstation)"}

    return {
        "quantum_length": {"bits": ql, "desc": ql_desc[ql]},
        "foreground_boost": {"bits": fb, "desc": fb_desc[fb]},
        "quantum_type": {"bits": qt, "desc": qt_desc[qt]},
    }


def calculate_quantum(w32ps):
    qt = w32ps & 0x3
    fb = (w32ps >> 2) & 0x3
    product_type = get_os_product_type()

    # Base quantum units: Short=6, Long=36
    if qt == 1 or qt == 3:
        base_units = 6
    elif qt == 2:
        base_units = 36
    else:
        base_units = 6 if product_type == 1 else 36

    # Foreground boost units
    if fb == 0:
        boost_units = 6 if product_type == 1 else 0
    else:
        boost_units = fb * 2

    return base_units, boost_units


def get_windows_version():
    """Windows 빌드 번호와 timer resolution 동작 모드를 반환한다."""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        )
        build = int(winreg.QueryValueEx(key, "CurrentBuildNumber")[0])
        display = winreg.QueryValueEx(key, "DisplayVersion")[0]
        winreg.CloseKey(key)
    except (OSError, ValueError):
        build = 0
        display = "Unknown"

    # Timer resolution 동작 모드 판별
    # build 19041 = Windows 10 2004 (per-process 도입)
    # build 22000+ = Windows 11
    if build >= 22000:
        timer_mode = "Per-process (Win11)"
    elif build >= 19041:
        timer_mode = "Per-process (Win10 2004+)"
    else:
        timer_mode = "Global (legacy)"

    return {
        "build": build,
        "display_version": display,
        "timer_mode": timer_mode,
    }


def get_global_timer_resolution_requests():
    """GlobalTimerResolutionRequests 레지스트리 값을 확인한다.
    Windows 11에서 1로 설정하면 per-process를 전역(legacy) 동작으로 되돌린다.
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Kernel"
        )
        value, _ = winreg.QueryValueEx(key, "GlobalTimerResolutionRequests")
        winreg.CloseKey(key)
        return value
    except OSError:
        return None


def get_session_info():
    """현재 프로세스의 세션 정보를 가져온다."""
    kernel32 = ctypes.WinDLL("kernel32")
    pid = os.getpid()

    # Session ID
    session_id = ctypes.wintypes.DWORD()
    kernel32.ProcessIdToSessionId(pid, ctypes.byref(session_id))
    sid = session_id.value

    # Session type 판별
    # Session 0 = Services/SSH (Windows Session 0 Isolation)
    # Session > 0 = Interactive user session (Console/RDP)
    #
    # SSH의 sshd는 Session 0에서 동작하며, 자식 프로세스도 Session 0에 생성됨.
    # RDP/Console은 항상 Session >= 1.
    if sid == 0:
        session_type = "Non-interactive (SSH/Service)"
        is_interactive = False
    else:
        # Session > 0: query session으로 RDP vs Console 구분
        session_type = "Interactive"
        is_interactive = True
        try:
            result = subprocess.run(
                ["query", "session"],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == str(sid) or p == f"ID":
                        continue
                    if str(sid) in parts:
                        if "rdp" in line.lower():
                            session_type = "Interactive (RDP)"
                        elif "console" in line.lower():
                            session_type = "Interactive (Console)"
                        break
                if session_type != "Interactive":
                    break
        except (FileNotFoundError, OSError):
            pass

    # Quantum boost eligibility:
    # Interactive 세션(RDP/Console)에서 foreground window를 가진 프로세스가 boost를 받음.
    # 판별: 이 세션이 interactive인지 (구조적으로 boost를 받을 수 있는 환경인지)
    # python.exe 자체가 foreground를 소유하진 않지만, 같은 세션의 프로세스가 boost 대상.
    can_receive_boost = is_interactive

    return {
        "session_id": sid,
        "session_type": session_type,
        "is_interactive": is_interactive,
        "can_receive_boost": can_receive_boost,
    }


def cmd_status():
    print()
    print("  Benchmark Environment Status")
    print("  " + "=" * 60)

    # Collect all data
    guid, plan_name = get_active_scheme()
    min_ac, _ = query_power_setting(SUB_PROCESSOR, PROCTHROTTLEMIN)
    max_ac, _ = query_power_setting(SUB_PROCESSOR, PROCTHROTTLEMAX)
    cp_ac, _ = query_power_setting(SUB_PROCESSOR, CPMINCORES)
    w32ps = get_win32_priority_separation()
    min_res, max_res, cur_res = get_timer_resolution()
    clock_ms = min_res / 10000.0

    pid = os.getpid()
    kernel32 = ctypes.WinDLL("kernel32")
    handle = kernel32.OpenProcess(0x0400, False, pid)
    priority_class = kernel32.GetPriorityClass(handle)
    kernel32.CloseHandle(handle)
    priority_class_names = {
        0x40: ("IDLE", 4), 0x4000: ("BELOW_NORMAL", 6),
        0x20: ("NORMAL", 8), 0x8000: ("ABOVE_NORMAL", 10),
        0x80: ("HIGH", 13), 0x100: ("REALTIME", 24),
    }
    class_name, base_priority = priority_class_names.get(priority_class, (f"0x{priority_class:x}", "?"))

    # Session info
    session = get_session_info()
    winver = get_windows_version()
    global_timer_reg = get_global_timer_resolution_requests()

    # --- Session Info ---
    can_boost = session["can_receive_boost"]

    # Timer mode display
    if global_timer_reg == 1 and winver["build"] >= 22000:
        timer_mode_str = f"{winver['timer_mode']} -> FORCED GLOBAL"
    else:
        timer_mode_str = winver["timer_mode"]

    # --- Summary Table ---
    print()
    print("  +-----------------------------+--------------------------------+")
    print("  | Setting                     | Value                          |")
    print("  +-----------------------------+--------------------------------+")
    win_str = f"Build {winver['build']} ({winver['display_version']})"
    print(f"  | Windows                     | {win_str:<30} |")
    print(f"  | Timer Resolution Mode       | {timer_mode_str:<30} |")
    print(f"  | Session ID                  | {session['session_id']:<30} |")
    print(f"  | Session Type                | {session['session_type']:<30} |")
    print(f"  | Quantum Boost Eligible      | {'YES' if can_boost else 'NO':<30} |")
    print("  +-----------------------------+--------------------------------+")
    print(f"  | Power Plan                  | {plan_name:<30} |")
    min_str = f"{min_ac}%" if min_ac is not None else "N/A"
    max_str = f"{max_ac}%" if max_ac is not None else "N/A"
    print(f"  | Min Processor State (AC)    | {min_str:<30} |")
    print(f"  | Max Processor State (AC)    | {max_str:<30} |")
    cp_str = f"{cp_ac}%" if cp_ac is not None else "N/A (not supported)"
    print(f"  | Core Parking Min Cores      | {cp_str:<30} |")
    sleep_ac, _ = query_power_setting(SUB_SLEEP, STANDBYIDLE)
    if sleep_ac is None:
        sleep_str = "N/A"
    elif sleep_ac == 0:
        sleep_str = "never"
    else:
        sleep_str = f"{sleep_ac}s ({sleep_ac / 60:.0f} min)  <-- may cut runs"
    print(f"  | Sleep Timeout (AC)          | {sleep_str:<30} |")
    sb_str = "RUNNING (held)" if is_holder_running(AWAKE_HOLDER_MUTEX) else "not running"
    print(f"  | Modern Standby              | {sb_str:<30} |")
    dstate = get_defender_state()
    d_str = {True: "real-time ON (may skew)", False: "real-time OFF",
             None: "N/A"}[dstate]
    print(f"  | Windows Defender            | {d_str:<30} |")
    print("  +-----------------------------+--------------------------------+")
    print(f"  | Process Priority Class      | {class_name:<30} |")
    print(f"  | Base Priority               | {str(base_priority):<30} |")
    timer_str = f"{cur_res / 10000.0:.3f} ms"
    print(f"  | Timer Resolution (current)  | {timer_str:<30} |")
    timer_default_str = f"{min_res / 10000.0:.3f} ms"
    print(f"  | Timer Resolution (default)  | {timer_default_str:<30} |")
    timer_best_str = f"{max_res / 10000.0:.3f} ms"
    print(f"  | Timer Resolution (best)     | {timer_best_str:<30} |")
    holder_str = "RUNNING (held)" if is_holder_running(TIMER_HOLDER_MUTEX) else "not running"
    print(f"  | Timer Holder                | {holder_str:<30} |")
    print("  +-----------------------------+--------------------------------+")

    # --- GPU Frequency ---
    gpu_info = get_gpu_frequency_info()
    if gpu_info:
        print()
        print("  +-----------------------------+--------------------------------+")
        print("  | GPU Frequency               | Value                          |")
        print("  +-----------------------------+--------------------------------+")
        for info in gpu_info:
            if info["domain"] == "GPU":
                cur_str = f"{info['cur_min']:.0f} - {info['cur_max']:.0f} MHz"
                fixed = info["cur_min"] == info["cur_max"]
                status = f"FIXED at {info['cur_max']:.0f} MHz" if fixed else "dynamic"
                print(f"  | Current Range               | {cur_str:<30} |")
                print(f"  | Status                      | {status:<30} |")
                print(f"  | Can Control                 | {'YES' if info['can_control'] else 'NO':<30} |")
        print("  +-----------------------------+--------------------------------+")

    # --- Win32PrioritySeparation detail ---
    if w32ps is not None:
        decoded = decode_w32ps(w32ps)
        print()
        print(f"  Win32PrioritySeparation = 0x{w32ps:x} (decimal: {w32ps})")
        print(f"    Quantum Length  [5:4]: {decoded['quantum_length']['bits']} = {decoded['quantum_length']['desc']}")
        print(f"    Foreground Boost[3:2]: {decoded['foreground_boost']['bits']} = {decoded['foreground_boost']['desc']}")
        print(f"    Quantum Type    [1:0]: {decoded['quantum_type']['bits']} = {decoded['quantum_type']['desc']}")

    # --- Quantum Calculation ---
    if w32ps is not None:
        base_units, boost_units = calculate_quantum(w32ps)
        fg_units = base_units + boost_units
        bg_units = base_units
        my_units = fg_units if can_boost else bg_units
        print()
        print("  +----------------------+---------+--------------+-------+")
        print("  | Quantum              |  Units  |     Time     | This? |")
        print("  +----------------------+---------+--------------+-------+")
        print(f"  | Base                 |    {base_units:>3}  |  {base_units * clock_ms:>7.1f} ms |       |")
        print(f"  | Foreground Boost     |   +{boost_units:>3}  |  +{boost_units * clock_ms:>6.1f} ms |       |")
        print("  +----------------------+---------+--------------+-------+")
        fg_mark = " <<<" if can_boost else "     "
        bg_mark = " <<<" if not can_boost else "     "
        print(f"  | Foreground (RDP)     |    {fg_units:>3}  |  {fg_units * clock_ms:>7.1f} ms |{fg_mark}|")
        print(f"  | Background (SSH)     |    {bg_units:>3}  |  {bg_units * clock_ms:>7.1f} ms |{bg_mark}|")
        print("  +----------------------+---------+--------------+-------+")

    # --- Priority Reference ---
    print()
    print("  Priority Class Reference:")
    print("  +-----------------------------+---------------+")
    print("  | Class                       | Base Priority |")
    print("  +-----------------------------+---------------+")
    ref = [("IDLE", 4), ("BELOW_NORMAL", 6), ("NORMAL", 8),
           ("ABOVE_NORMAL", 10), ("HIGH", 13), ("REALTIME", 24)]
    for name, bp in ref:
        marker = " <<<" if name == class_name else ""
        print(f"  | {name:<27} |      {bp:>2}       |{marker}")
    print("  +-----------------------------+---------------+")
    print()


def cmd_set(with_timer=False):
    mode_name = "set_max" if with_timer else "set"

    # set이 이미 적용된 상태에서 다시 set하면 "적용된 값"이 원본으로 백업되어 restore가
    # 무의미해진다. 기존 backup을 보존해 이 오염을 막는다.
    if os.path.exists(BACKUP_FILE):
        print(f"[*] Backup already exists: {BACKUP_FILE}")
        print("    Keeping it - re-running set must not overwrite the pristine state.")
        print("    (restore first if you want a fresh baseline)")
        print()
        skip_backup = True
    else:
        skip_backup = False

    # 현재 설정 백업
    print("[*] Saving current settings to backup...")
    guid, name = get_active_scheme()
    min_ac, _ = query_power_setting(SUB_PROCESSOR, PROCTHROTTLEMIN)
    max_ac, _ = query_power_setting(SUB_PROCESSOR, PROCTHROTTLEMAX)
    cp_ac, _ = query_power_setting(SUB_PROCESSOR, CPMINCORES)
    sleep_ac, sleep_dc = query_power_setting(SUB_SLEEP, STANDBYIDLE)
    hib_ac, hib_dc = query_power_setting(SUB_SLEEP, HIBERNATEIDLE)

    if skip_backup:
        print("    Skipped (existing backup preserved)")
    else:
        backup = {
            "power_scheme": guid,
            "power_scheme_name": name,
            "proc_throttle_min": min_ac,
            "proc_throttle_max": max_ac,
            "core_parking_min": cp_ac,
            "defender_realtime": get_defender_state(),
            # 기록용으로만 남긴다. restore는 이 값을 되돌리지 않고 never를 유지한다 -
            # SSH 전용 머신에서 절전은 세션 단절과 제어 불가를 뜻한다.
            "standby_idle_original": [sleep_ac, sleep_dc],
            "hibernate_idle_original": [hib_ac, hib_dc],
        }
        with open(BACKUP_FILE, "w") as f:
            json.dump(backup, f, indent=2)
        print(f"    Backup saved: {BACKUP_FILE}")
    print()

    # 설정 적용
    print(f"[*] Applying benchmark settings ({mode_name})...")

    # High Performance power plan
    print("    - Power Plan: High Performance")
    out, rc = run_powercfg("/setactive", POWER_HIGH_PERFORMANCE)
    if rc != 0:
        print(f"    [WARN] Failed to set High Performance plan (rc={rc})")
        print("    Trying to apply settings to current scheme instead...")

    # Core Parking 비활성화
    print("    - Core Parking: Disabled (100% min cores)")
    run_powercfg("/setacvalueindex", "SCHEME_CURRENT", SUB_PROCESSOR, CPMINCORES, "100")
    run_powercfg("/setactive", "SCHEME_CURRENT")

    # Min Processor State 100%
    print("    - Min Processor State: 100%")
    run_powercfg("/setacvalueindex", "SCHEME_CURRENT", SUB_PROCESSOR, PROCTHROTTLEMIN, "100")
    run_powercfg("/setactive", "SCHEME_CURRENT")

    # Sleep/hibernate timeout을 never로. holder는 프로세스가 살아 있는 동안만 유효하므로,
    # holder가 죽어도 측정 중 절전으로 빠지지 않게 정책 자체를 막는다. AC/DC 모두 설정한다 -
    # 노트북에서 전원이 빠지면 DC 값이 적용된다.
    n_schemes, sleep_failed = disable_sleep_all_schemes()
    print(f"    - Sleep / hibernate timeout: never (AC+DC, all {n_schemes} power schemes)")
    if sleep_failed:
        print("      [WARN] some schemes could not be updated")

    # System sleep 억제. /requestsoverride는 기존 요청을 무시하는 반대 기능이므로 쓰지 않는다.
    print("    - System sleep: starting execution-state holder")
    awake_state = start_awake_holder()
    if awake_state == "already-running":
        print("      [SKIP] holder already running")
    elif awake_state == "started":
        print("      holder started")
    else:
        print("      [WARN] failed to start holder - system sleep NOT suppressed")

    # Defender 실시간 보호 해제. 스캔이 돌면 GPU 커널 시간까지 2-3배 부풀려진다.
    print("    - Defender real-time protection: disabling")
    ok, state = set_defender_realtime(False)
    if state is None:
        print("      [N/A] Defender not present or not queryable")
    elif state:
        print("      [WARN] still enabled - needs Administrator, or policy blocks it")
    else:
        print("      disabled")

    # Timer Resolution (set_max only)
    if with_timer:
        min_res, max_res, cur_res = get_timer_resolution()
        target = 10000  # 1.000 ms in 100ns units
        print(f"    - Timer Resolution: {target / 10000.0:.3f} ms (background holder)")
        state = start_timer_holder(target)
        if state == "already-running":
            print("      [SKIP] holder already running - not starting a duplicate")
        elif state == "started":
            _, _, new_cur = get_timer_resolution()
            print(f"      holder started, system resolution now {new_cur / 10000.0:.3f} ms")
        else:
            print("      [WARN] failed to start holder - timer resolution NOT held")

    # GPU Frequency (set_max only)
    if with_timer:
        gpu_info = get_gpu_frequency_info()
        if gpu_info:
            for info in gpu_info:
                if info["domain"] == "GPU" and info["can_control"]:
                    # 이 드라이버(gfx-driver-ci-master-21445)는 zes_freq_properties_t를
                    # min=3100, max=0으로 채운다. 즉 hw_max는 신뢰할 수 없고 하드웨어
                    # 최대치가 hw_min 쪽에 들어온다. 세 값 중 가장 큰 것을 고르면
                    # 드라이버가 어느 필드를 쓰든 하드웨어 최대로 고정된다.
                    # cur_max만 fallback으로 쓰면, 이전에 낮은 값으로 고정된 상태에서
                    # set_max를 실행할 때 그 낮은 값에 다시 고정되는 함정이 있다.
                    max_freq = max(info["hw_max"], info["hw_min"], info["cur_max"])
                    # 반복 set_max가 이미 고정된 값을 원본으로 덮어쓰지 않게 한다.
                    if not os.path.exists(GPU_BACKUP_FILE):
                        gpu_backup = {"cur_min": info["cur_min"], "cur_max": info["cur_max"]}
                        with open(GPU_BACKUP_FILE, "w") as f:
                            json.dump(gpu_backup, f, indent=2)
                    print(f"    - GPU Frequency: fixed at {max_freq:.0f} MHz")
                    ok, actual = set_gpu_frequency_range(max_freq, max_freq)
                    if ok and actual:
                        print(f"      set to min={actual[0]:.0f}, max={actual[1]:.0f} MHz")
                    else:
                        print("      [WARN] failed to set GPU frequency")
                    break
        else:
            print("    - GPU Frequency: [N/A] Level Zero not available")

    print()
    print("[OK] Benchmark environment ready.")
    print("     Run your benchmark with:")
    print("       start /HIGH benchmark.exe")
    print("     After done, run:")
    print("       python benchmark_env_setup.py restore")


def cmd_restore():
    # Timer holder는 backup 파일과 독립적으로 정리한다. set이 중간에 실패해 backup이
    # 없더라도 holder는 떠 있을 수 있으므로, 먼저 처리하고 나서 backup 유무를 판단한다.
    print("[*] Stopping timer resolution holder...")
    killed = stop_timer_holder()
    if killed > 0:
        _, _, cur = get_timer_resolution()
        print(f"    - holder stopped ({killed} process), "
              f"resolution back to {cur / 10000.0:.3f} ms")
    elif killed == 0:
        print("    - no holder running")
    else:
        print("    - [WARN] holder mutex exists but its process was not found;")
        print("             it may belong to another user. Timer resolution still held.")
    print()

    print("[*] Stopping system sleep holder...")
    awake_killed = stop_awake_holder()
    if awake_killed > 0:
        print(f"    - holder stopped ({awake_killed} process)")
    elif awake_killed == 0:
        print("    - no holder running")
    else:
        print("    - [WARN] holder is still running or could not be found")
    print()

    if not os.path.exists(BACKUP_FILE):
        print(f"[ERROR] Backup file not found: {BACKUP_FILE}")
        print("        Run 'python benchmark_env_setup.py set' first.")
        return 1

    with open(BACKUP_FILE, "r") as f:
        backup = json.load(f)

    print("[*] Restoring settings from backup...")

    restore_failed = awake_killed < 0

    # Power Plan 복원
    scheme = backup.get("power_scheme")
    if scheme:
        print(f"    - Power Plan: {scheme} ({backup.get('power_scheme_name', '')})")
        _, rc = run_powercfg("/setactive", scheme)
        restore_failed |= rc != 0

    # Min Processor State 복원
    proc_min = backup.get("proc_throttle_min")
    if proc_min is not None:
        print(f"    - Min Processor State: {proc_min}%")
        _, rc = run_powercfg("/setacvalueindex", "SCHEME_CURRENT", SUB_PROCESSOR,
                     PROCTHROTTLEMIN, str(proc_min))
        restore_failed |= rc != 0
        _, rc = run_powercfg("/setactive", "SCHEME_CURRENT")
        restore_failed |= rc != 0

    # Core Parking 복원
    cp_min = backup.get("core_parking_min")
    if cp_min is not None:
        print(f"    - Core Parking Min Cores: {cp_min}%")
        _, rc = run_powercfg("/setacvalueindex", "SCHEME_CURRENT", SUB_PROCESSOR,
                             CPMINCORES, str(cp_min))
        restore_failed |= rc != 0
        _, rc = run_powercfg("/setactive", "SCHEME_CURRENT")
        restore_failed |= rc != 0

    # Sleep / hibernate timeout은 복원하지 않고 never로 유지한다.
    #
    # 백업값을 되돌리지 않는 유일한 항목이다. 이 머신은 SSH로만 접근하므로, 절전에
    # 들어가면 세션이 끊기고 머신을 제어할 수 없게 된다 - 다음 측정이 아니라 복구가
    # 불가능해지는 문제다. 전원 계획을 바꿔도 SCHEME_CURRENT에 다시 적용하므로 어느
    # 스킴이 활성이든 never가 유지된다.
    n_schemes, sleep_failed = disable_sleep_all_schemes()
    print(f"    - Sleep / hibernate timeout: keeping 'never' (all {n_schemes} schemes, NOT restored)")
    print("      sleeping would drop the SSH session and lock us out of the machine")
    restore_failed |= sleep_failed

    # Defender는 항상 켜는 쪽으로 복원한다.
    #
    # 백업값을 그대로 따르지 않는다: set이 이미 적용된 상태에서 백업이 떠졌으면
    # defender_realtime=False가 기록되어 restore가 보호를 끈 채로 남긴다. 실시간 보호를
    # 끈 상태로 방치하는 것은 되돌리기 실패 중에서도 위험한 쪽이므로, 안전한 방향으로
    # 편향시킨다. 원래부터 꺼 두고 쓰는 머신이라면 restore 후 다시 끄면 된다.
    if backup.get("defender_realtime") is False:
        print("    - Defender real-time protection: backup says it was already off,")
        print("      but enabling anyway (never leave protection off after restore)")
    else:
        print("    - Defender real-time protection: enabling")
    ok, state = set_defender_realtime(True)
    if state is False:
        print("      [WARN] still disabled - needs Administrator, or policy blocks it")
        restore_failed = True

    # GPU Frequency 복원
    if os.path.exists(GPU_BACKUP_FILE):
        with open(GPU_BACKUP_FILE, "r") as f:
            gpu_backup = json.load(f)
        orig_min = gpu_backup.get("cur_min", 300)
        orig_max = gpu_backup.get("cur_max", 2400)
        print(f"    - GPU Frequency: restoring to {orig_min:.0f} - {orig_max:.0f} MHz")
        ok, actual = set_gpu_frequency_range(orig_min, orig_max)
        if ok and actual:
            print(f"      set to min={actual[0]:.0f}, max={actual[1]:.0f} MHz")
            os.remove(GPU_BACKUP_FILE)
        else:
            print("      [WARN] failed to restore GPU frequency")
            restore_failed = True

    if restore_failed:
        print()
        print("[ERROR] One or more settings could not be restored.")
        print("        Backup files were kept so restore can be retried.")
        return 1

    print()
    print("[OK] Settings restored.")
    os.remove(BACKUP_FILE)
    print("     Backup file removed.")
    return 0


def main():
    # stdout 인코딩을 UTF-8로 강제 (cp1252 등에서 한글 깨짐 방지)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # background holder 진입점. set이 이 스크립트를 이 인자로 재실행한다.
    if len(sys.argv) >= 2 and sys.argv[1] == TIMER_HOLDER_ARG:
        target = int(sys.argv[2]) if len(sys.argv) > 2 else get_timer_resolution()[1]
        return run_timer_holder(target)
    if len(sys.argv) >= 2 and sys.argv[1] == AWAKE_HOLDER_ARG:
        return run_awake_holder()

    if len(sys.argv) < 2 or sys.argv[1] not in ("status", "set", "set_max", "restore"):
        print(__doc__)
        return 0

    cmd = sys.argv[1]
    if cmd == "status":
        cmd_status()
    elif cmd == "set":
        return cmd_set(with_timer=False)
    elif cmd == "set_max":
        return cmd_set(with_timer=True)
    elif cmd == "restore":
        return cmd_restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
