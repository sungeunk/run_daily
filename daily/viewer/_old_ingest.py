
# ingest/ingest_reports.py
import argparse, json, hashlib
import duckdb
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any
from common_utils import is_float
from report import load_result_file, generate_csv_table


logger = logging.getLogger(__name__)

# -------------------------
# 정규식 (전역 컴파일)
# -------------------------

# 1) Purpose: 파이프 테이블 라인: "| Purpose | nightly |"
PURPOSE_TABLE_RE = re.compile(
    r"""(?im)               # 다중라인/대소문자 무시
    ^\s*\|\s*               # 라인 시작의 파이프
    Purpose\s*\|            # 'Purpose' 셀
    \s*(?P<purpose>[^|]+?)  # 다음 셀 내용
    \s*\|                   # 닫는 파이프
    """,
    re.VERBOSE,
)

# 1-2) Purpose: "Purpose: nightly" 같은 자유 텍스트 폴백
PURPOSE_FALLBACK_RE = re.compile(
    r"""(?im)
    ^\s*Purpose\s*:\s*(?P<purpose>.+?)\s*$
    """,
    re.VERBOSE,
)

# 2) Commit ID: OpenVINO 라인에서 '-<build>-<sha>' 추출
# 파이프/자유 텍스트 모두 허용. SHA는 7~40자(짧은 축약에서 전체 SHA까지).
COMMIT_RE = re.compile(
    r"""(?is)               # .가 개행 포함, 대소문자 무시
    OpenVINO[^\n\r]*?       # 'OpenVINO'가 포함된 라인/구간
    -(?P<build>\d+)         # -빌드번호
    -(?P<sha>[0-9a-fA-F]{7,40})  # -SHA
    """,
    re.VERBOSE,
)

# 3) 파일명에서 ".YYYYMMDD_HHMM." 패턴 추출
FILENAME_DT_RE = re.compile(r"""\.(?P<date>\d{8})_(?P<time>\d{3,4})\.""", re.ASCII)


# -------------------------
# 파서 함수
# -------------------------

def parse_report_file(filepath: Path) -> Tuple[str, str]:
    """
    텍스트 .report 파일에서 Purpose와 OpenVINO commit_id('<build>-<sha>')를 추출.
    실패 시 각 항목 "N/A" 반환.

    Returns:
        (purpose, commit_id)
    """
    purpose = "N/A"
    commit_id = "N/A"

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")

        # Purpose (우선: 파이프 테이블)
        m = PURPOSE_TABLE_RE.search(text)
        if m:
            purpose = m.group("purpose").strip()
        else:
            # 폴백: "Purpose: something"
            m2 = PURPOSE_FALLBACK_RE.search(text)
            if m2:
                purpose = m2.group("purpose").strip()

        # Commit ID
        m3 = COMMIT_RE.search(text)
        if m3:
            commit_id = f"{m3.group('build')}-{m3.group('sha').lower()}"

    except OSError as e:
        # Streamlit 사용 중이면 아래 한 줄로 바꿔도 됩니다:
        # import streamlit as st; st.error(f"Error reading {filepath}: {e}")
        logger.error("Error reading %s: %s", filepath, e)

    return purpose, commit_id


def get_report_metadata(report_path: Path) -> Dict[str, str]:
    """
    파일 경로/파일명에서 날짜/시간 키를 추출하고,
    파일 본문에서 purpose/commit_id를 추출해 메타데이터를 반환.

    Returns:
        {
          'filename': str,
          'purpose': str,
          'commit_id': str,  # '<build>-<sha>'
          'workweek': str,   # 'YYYY.WWw.D'
          'datetime': str,   # 'YYYYMMDD_HHMM'
        }
    """
    purpose, commit_id = parse_report_file(report_path)

    workweek = "N/A"
    datetime_key = "N/A"

    m = FILENAME_DT_RE.search(report_path.name)
    if m:
        date_str = m.group("date")
        time_str = m.group("time").zfill(4)  # 3자리면 0 보강 → HHMM
        datetime_key = f"{date_str}_{time_str}"

        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            iso = dt.isocalendar()  # (year, week, weekday)
            workweek = f"{iso.year}.WW{iso.week}.{iso.weekday}"
        except Exception:
            workweek = "N/A"

    return {
        "filename": report_path.name,
        "hostname": report_path.parent.name,
        "purpose": purpose,
        "commit_id": commit_id,
        "workweek": workweek,
        "datetime": datetime_key,
    }

def to_jsonable(obj: Any) -> Any:
    """객체를 JSON 직렬화 가능한 형태로 변환"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return to_jsonable(obj.__dict__)
    elif hasattr(obj, 'tolist'):  # numpy array 등
        return obj.tolist()
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def load_pickle_data(pickle_path: Path) -> Any:
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return to_jsonable(data)

def build_perf_rows_from_pickle(pickle_path: Path, run_id: str) -> list[dict]:
    result_root = load_result_file(str(pickle_path))
    csv_table = generate_csv_table(result_root, False)
    perf_rows = []

    for item in csv_table:
        if len(item) != 6:
            continue
        model, precision, in_token, out_token, exec_mode, value = item

        if model == 'qwen_usage':
            if exec_mode == 'memory percent' and is_float(value):
                value = float(value)
            elif exec_mode == 'memory size' and is_float(value):
                value = float(value) / (1024**3)

        if not is_float(value):
            continue

        perf_rows.append({
            "run_id": run_id,
            "model": model,
            "precision": precision,
            "in_token": int(float(in_token)) if is_float(in_token) else 0,
            "out_token": int(float(out_token)) if is_float(out_token) else 0,
            "exec_mode": exec_mode,
            "value": float(value),
            "unit": None,
        })

    return perf_rows

def pickle_to_json(pickle_path: Path, json_path: Path = None, indent: int = 2, metadata: Dict[str, str] = None) -> str:
    """
    Pickle 파일에 저장된 데이터를 JSON 포맷으로 변환합니다.
    
    Args:
        pickle_path: 변환할 pickle 파일 경로
        json_path: JSON 파일 저장 경로 (None이면 저장하지 않음)
        indent: JSON 출력 들여쓰기 (기본값: 2)
        metadata: 추가할 메타데이터 (기본값: None)
    
    Returns:
        JSON 형식의 문자열
    
    Raises:
        FileNotFoundError: pickle 파일이 존재하지 않을 경우
        ValueError: pickle 데이터를 JSON으로 직렬화할 수 없을 경우
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    data = load_pickle_data(pickle_path)
    serializable_data = to_jsonable(data)
    # 메타데이터 포함
    output = {"metadata": metadata} if metadata else {}
    output["data"] = serializable_data
    json_str = json.dumps(output, indent=indent, ensure_ascii=False)
    
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        logger.info(f"JSON saved to: {json_path}")
    
    return json_str

def parse_ts(s: str) -> datetime:
    # "YYYYMMDD_HHMM" → timestamp
    return datetime.strptime(s, "%Y%m%d_%H%M")

def file_sig(p: Path) -> str:
    st = p.stat()
    raw = f"{p}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def flatten_report(obj: dict, source_path: str, file_hash: str):
    # version / commit_id 구조
    ver = obj.get("version", {}) or {}
    cid = obj.get("commit_id", {}) or {}

    # run 레코드
    ts = parse_ts(obj["datetime"])
    run = {
        "run_id": hashlib.sha1(f'{obj.get("machine","")}|{obj.get("report","")}'.encode()).hexdigest()[:20],
        "report_file": obj.get("report"),
        "machine": obj.get("machine"),
        "purpose": obj.get("purpose"),
        "rawlog": obj.get("rawlog"),
        "ts": ts,
        "ww": obj.get("ww"),
        "ov_version": ver.get("openvino"),
        "genai_version": ver.get("openvino.genai"),
        "ov_commit": cid.get("openvino"),
        "genai_commit": cid.get("openvino.genai"),
        "tok_commit": cid.get("openvino_tokenizers"),
        "source_path": source_path,
        "file_hash": file_hash,
    }

    # system 디바이스
    sys_rows = []
    for i, dev in enumerate(obj.get("system", []) or []):
        sys_rows.append({
            "run_id": run["run_id"],
            "device_index": i,
            "device": dev.get("device"),
            "driver": dev.get("driver"),
            "eu": int(dev.get("eu") or 0),
            "clock_freq_mhz": float(dev.get("clock_freq") or 0),
            "global_mem_size_gb": float(dev.get("global_mem_size") or 0),
        })

    # perf
    perf_rows = []
    for pf in obj.get("perf", []) or []:
        perf_rows.append({
            "run_id": run["run_id"],
            "model": pf.get("model"),
            "precision": pf.get("precision"),
            "in_token": int(pf.get("in_token") or 0),
            "out_token": int(pf.get("out_token") or 0),
            "exec_mode": pf.get("exec"),
            "value": float(pf.get("perf") or 0),
            "unit": pf.get("perf_unit"),
        })

    return run, sys_rows, perf_rows

def upsert(con, run, sys_rows, perf_rows):
    # runs upsert
    con.execute("""
        INSERT INTO runs
        (run_id, report_file, machine, purpose, rawlog, ts, ww,
         ov_version, genai_version, ov_commit, genai_commit, tok_commit,
         source_path, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
          report_file=excluded.report_file,
          machine=excluded.machine,
          purpose=excluded.purpose,
          rawlog=excluded.rawlog,
          ts=excluded.ts,
          ww=excluded.ww,
          ov_version=excluded.ov_version,
          genai_version=excluded.genai_version,
          ov_commit=excluded.ov_commit,
          genai_commit=excluded.genai_commit,
          tok_commit=excluded.tok_commit,
          source_path=excluded.source_path,
          file_hash=excluded.file_hash
    """, [
        run["run_id"], run["report_file"], run["machine"], run["purpose"], run["rawlog"],
        run["ts"], run["ww"], run["ov_version"], run["genai_version"],
        run["ov_commit"], run["genai_commit"], run["tok_commit"],
        run["source_path"], run["file_hash"]
    ])

    if sys_rows:
                con.executemany("""
                    INSERT INTO system_devices
                    (run_id, device_index, device, driver, eu, clock_freq_mhz, global_mem_size_gb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, device_index) DO UPDATE SET
                        device=excluded.device,
                        driver=excluded.driver,
                        eu=excluded.eu,
                        clock_freq_mhz=excluded.clock_freq_mhz,
                        global_mem_size_gb=excluded.global_mem_size_gb
                """, [(
                        r["run_id"], r["device_index"], r["device"], r["driver"],
                        r["eu"], r["clock_freq_mhz"], r["global_mem_size_gb"]
                ) for r in sys_rows])

    if perf_rows:
                con.executemany("""
                    INSERT INTO perf
                    (run_id, model, precision, in_token, out_token, exec_mode, value, unit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, model, precision, in_token, out_token, exec_mode) DO UPDATE SET
                        value=excluded.value,
                        unit=excluded.unit
                """, [(
                        r["run_id"], r["model"], r["precision"], r["in_token"], r["out_token"],
                        r["exec_mode"], r["value"], r["unit"]
                ) for r in perf_rows])

def ingest(root: Path, db: Path, ddl: Path):
    con = duckdb.connect(str(db))
    con.execute(ddl.read_text(encoding="utf-8"))

    # 이미 적재된 파일 해시
    known = set(con.execute("SELECT file_hash FROM runs").fetchdf()["file_hash"].dropna().tolist())

    def print_progress(current: int, total: int, suffix: str = ""):
        if total <= 0:
            return
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        pct = (current / total) * 100
        print(f"\r[{bar}] {current}/{total} ({pct:5.1f}%) {suffix}", end="", flush=True)

    added = 0
    report_files = [p for p in root.iterdir() if p.is_file() and p.suffix == ".report"]
    total = len(report_files)
    print(f"[INFO] scanning {total} report files in {root}")

    for idx, p in enumerate(report_files, start=1):
        h = file_sig(p)
        if h in known:
            continue

        try:
            metadata = get_report_metadata(p)
            pickle_path = p.with_suffix('.pickle')
            if not pickle_path.exists():
                print(f"[WARN] skip {p}: missing pickle {pickle_path}")
                continue

            data = load_pickle_data(pickle_path)
            obj = data if isinstance(data, dict) else {"data": data}

            obj.setdefault("report", metadata.get("filename"))
            obj.setdefault("machine", metadata.get("hostname"))
            obj.setdefault("purpose", metadata.get("purpose"))
            obj.setdefault("ww", metadata.get("workweek"))

            dt_key = metadata.get("datetime")
            if not dt_key or dt_key == "N/A":
                dt_key = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d_%H%M")
            obj.setdefault("datetime", dt_key)

            if "rawlog" not in obj:
                try:
                    obj["rawlog"] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    obj["rawlog"] = None

            cid = obj.get("commit_id")
            if isinstance(cid, dict):
                if metadata.get("commit_id") and metadata["commit_id"] != "N/A":
                    cid.setdefault("openvino", metadata["commit_id"])
            else:
                cid_val = metadata.get("commit_id") if metadata.get("commit_id") != "N/A" else None
                if cid is not None and not cid_val:
                    cid_val = str(cid)
                cid = {"openvino": cid_val} if cid_val else {}
            obj["commit_id"] = cid

            run, sys_rows, perf_rows = flatten_report(obj, str(p), h)
            if not perf_rows:
                perf_rows = build_perf_rows_from_pickle(pickle_path, run["run_id"])
            upsert(con, run, sys_rows, perf_rows)
            known.add(h); added += 1
            print_progress(idx, total, f"added {added} | {p.name}")
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    con.close()
    if total > 0:
        print()
    print(f"[OK] ingested {added} new files")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="리포트(.report) 루트 폴더")
    ap.add_argument("--db", type=Path, default=Path(__file__).with_name("bench.duckdb"))
    ap.add_argument("--ddl", type=Path, default=Path(__file__).with_name("bench_schema.sql"))
    ap.add_argument("-i", "--input", type=Path)
    args = ap.parse_args()
    if args.root:
        ingest(Path(args.root), Path(args.db), Path(args.ddl))
    elif args.input:
        metadata = get_report_metadata(args.input)
        pickle_to_json(Path(args.input).with_suffix('.pickle'),
                       json_path=Path(args.input).with_suffix('.json'),
                       metadata=metadata)
