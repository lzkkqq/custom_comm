#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_COMM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
CUSTOM_COMM_LIB_DIR="${CUSTOM_COMM_LIB_DIR:-${CUSTOM_COMM_ROOT}/python/custom_comm}"
TARGET="${SCRIPT_DIR}/custom_comm_allgather_batch_testcase"

export LD_LIBRARY_PATH="${CUSTOM_COMM_LIB_DIR}:${ASCEND_CANN_PACKAGE_PATH}/lib64:${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64:${ASCEND_CANN_PACKAGE_PATH}/hcomm/hcomm/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${TARGET}" ]]; then
    make -C "${SCRIPT_DIR}"
fi

mode="default"
if [[ $# -gt 0 ]]; then
    case "$1" in
        decomposed|ccu|ccu-ms)
            mode="$1"
            shift
            ;;
    esac
fi

case "${mode}" in
    decomposed|default)
        unset CUSTOM_COMM_USE_CCU || true
        unset CUSTOM_COMM_CCU_MODE || true
        ;;
    ccu)
        export CUSTOM_COMM_USE_CCU=1
        unset CUSTOM_COMM_CCU_MODE || true
        ;;
    ccu-ms)
        export CUSTOM_COMM_USE_CCU=1
        export CUSTOM_COMM_CCU_MODE=ms
        ;;
esac

exec "${TARGET}" "$@"
