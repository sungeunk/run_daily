#!/bin/bash
# Usage:
#   run.daily2.sh                      download latest nightly OpenVINO and run
#   run.daily2.sh <COMMIT_SHA>         download that build and run
#   run.daily2.sh <path/setupvars.sh>  use an already installed package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-daily.py312}"
SKILL_ROOT="${SKILL_ROOT:-/home/sungeunk/repo/openvino-gpu-plugin-skills}"
DOWNLOAD_OV="${SKILL_ROOT}/.github/skills/download-openvino/scripts/download-openvino.py"
DOWNLOAD_OUTPUT="${SCRIPT_DIR}/openvino_nightly"
LATEST_SETUP_FILE="${DOWNLOAD_OUTPUT}/latest_ov_setup_file.txt"

# shellcheck disable=SC1091
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

ARG="${1:-}"
if [[ -z "${ARG}" ]]; then
    echo "[1/2] No argument provided. Downloading latest OpenVINO nightly ..."
    uv run --script "${DOWNLOAD_OV}" --output "${DOWNLOAD_OUTPUT}"
    SETUPVARS=""
elif [[ -f "${ARG}" || "${ARG}" == *.sh ]]; then
    SETUPVARS="${ARG}"
else
    echo "[1/2] Downloading OpenVINO for commit ${ARG} ..."
    uv run --script "${DOWNLOAD_OV}" --commit-id "${ARG}" --output "${DOWNLOAD_OUTPUT}"
    SETUPVARS=""
fi

if [[ -z "${SETUPVARS}" ]]; then
    if [[ ! -f "${LATEST_SETUP_FILE}" ]]; then
        echo "[ERROR] ${LATEST_SETUP_FILE} not created by download script" >&2
        exit 1
    fi
    SETUPVARS="$(tr -d '\r\n' < "${LATEST_SETUP_FILE}")"
fi

if [[ ! -f "${SETUPVARS}" ]]; then
    echo "[ERROR] Setup script not found: ${SETUPVARS}" >&2
    exit 1
fi

echo "[2/2] Executing: ${SETUPVARS}"
# setupvars.sh references unset vars, so relax nounset while sourcing
set +u
# shellcheck disable=SC1090
. "${SETUPVARS}"
set -u
# setupvars.sh may prepend its own python paths; re-activate to keep the env first
conda activate "${CONDA_ENV}"

# python -m pytest daily/tests/ -vv --device=GPU.1
python daily/run.py --device GPU.1 -k gemma-2-9b-it --output-dir regression_out --verbose
