#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="/home/ubuntu/data/exp/proj2410/quant_model/Qwen2.5-0.5B/EfficientQAT/w2g64-int2-kd/checkpoints"
GLOBAL_STEP=""
SRC=""
OUT=""
OUT_DEQUANT=""
BITS="2"
GROUP_SIZE="64"
PACK_DTYPE="int32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT_DEFAULT="$2"; shift 2 ;;
    --global-step) GLOBAL_STEP="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --out-dequant) OUT_DEQUANT="$2"; shift 2 ;;
    --bits) BITS="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    --pack-dtype) PACK_DTYPE="$2"; shift 2 ;;
    *) break ;;
  esac
done

if [[ -z "${SRC}" ]]; then
  if [[ -z "${GLOBAL_STEP}" ]]; then
    GLOBAL_STEP="$(ls -d "${ROOT_DEFAULT}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
  else
    GLOBAL_STEP="${ROOT_DEFAULT}/global_step_${GLOBAL_STEP}"
  fi
  if [[ -z "${GLOBAL_STEP}" || ! -d "${GLOBAL_STEP}" ]]; then
    echo "ERROR: cannot resolve global_step under ${ROOT_DEFAULT}" >&2
    exit 1
  fi
  SRC="${GLOBAL_STEP}/hf_ckpt"
fi

OUT="${OUT:-${ROOT_DEFAULT}/out}"
OUT_DEQUANT="${OUT_DEQUANT:-${ROOT_DEFAULT}/out_dequant}"

python3 VeOmni/tasks/quantize/export_tritonv2_quant.py \
  --src "${SRC}" \
  --dst "${OUT}" \
  --dst-dequant "${OUT_DEQUANT}" \
  --bits "${BITS}" \
  --group-size "${GROUP_SIZE}" \
  --pack-dtype "${PACK_DTYPE}" \
  --compare \
  "$@"
