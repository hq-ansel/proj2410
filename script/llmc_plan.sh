#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLMC_ROOT="${LLMC_ROOT:-$REPO_ROOT/baseline/llmc}"
CONFIG_DIR="$LLMC_ROOT/configs/quantization/exp"

RUN=false
FILTER_RE=""
METHOD_RE=""
MODEL_RE=""
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"

usage() {
  cat <<USAGE
Usage: script/llmc_plan.sh [options]

Options:
  --run             Execute quantization for matched configs.
  --filter REGEX    Regex to filter config filenames.
  --method REGEX    Regex for method (e.g., awq|quik).
  --model REGEX     Regex for model (e.g., Llama3-8B).
  --config-dir DIR  Override config directory (default: baseline/llmc/configs/quantization/exp).
  -h, --help        Show this help.

Examples:
  script/llmc_plan.sh --filter 'awq_Llama3-8B'
  script/llmc_plan.sh --method awq --model Qwen2.5-7B
  script/llmc_plan.sh --run --method quik --model Llama2-7B
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN=true
      shift
      ;;
    --filter)
      FILTER_RE="$2"
      shift 2
      ;;
    --method)
      METHOD_RE="$2"
      shift 2
      ;;
    --model)
      MODEL_RE="$2"
      shift 2
      ;;
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config dir not found: $CONFIG_DIR" >&2
  exit 1
fi

mapfile -t configs < <(ls "$CONFIG_DIR"/*.yml 2>/dev/null | sort)

selected=()
for cfg in "${configs[@]}"; do
  base=$(basename "$cfg")
  if [[ -n "$FILTER_RE" ]] && [[ ! "$base" =~ $FILTER_RE ]]; then
    continue
  fi
  if [[ -n "$METHOD_RE" ]] && [[ ! "$base" =~ $METHOD_RE ]]; then
    continue
  fi
  if [[ -n "$MODEL_RE" ]] && [[ ! "$base" =~ $MODEL_RE ]]; then
    continue
  fi
  selected+=("$cfg")
done

if [[ ${#selected[@]} -eq 0 ]]; then
  echo "No configs matched." >&2
  exit 1
fi

echo "Matched configs (${#selected[@]}):"
for c in "${selected[@]}"; do
  echo "  $c"
done

if ! $RUN; then
  echo "Dry run only. Use --run to execute."
  exit 0
fi

export PYTHONPATH="$LLMC_ROOT:$PYTHONPATH"

find_unused_port() {
  while true; do
    port=$(shuf -i 10000-60000 -n 1)
    if ! ss -tuln | grep -q ":$port "; then
      echo "$port"
      return 0
    fi
  done
}

for cfg in "${selected[@]}"; do
  UNUSED_PORT=$(find_unused_port)
  MASTER_ADDR=127.0.0.1
  MASTER_PORT=$UNUSED_PORT
  task_id=$UNUSED_PORT

  echo "Running: $cfg"
  torchrun \
    --nnodes "$NNODES" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --rdzv_id "$task_id" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    "$LLMC_ROOT/llmc/__main__.py" --config "$cfg" --task_id "$task_id"

done
