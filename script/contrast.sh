#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 所有输出同时显示在屏幕并写入 exp.log
exec > >(tee -a "$REPO_ROOT/exp.log") 2>&1
EVAL_SH="$REPO_ROOT/script/eval.sh"

quant_root="$REPO_ROOT/quant_model"
llmc_root="$REPO_ROOT/baseline/llmc"
llmc_out="$llmc_root/output"
llmc_config_dir="$llmc_root/configs/quantization/exp"

model_re=""
method_re=""
dry_run=false
llmc_run=false
llmc_filter_re=""
llmc_method_re=""
llmc_model_re=""
hold=false
include_efficientqat=false

include_paths=()
glob_paths=()

llmc_effective_save_dirs=()

usage() {
  cat <<USAGE
Usage: script/contrast.sh [options]

Options:
  --include PATH     Add a specific model path to evaluate (repeatable).
  --glob GLOB        Add paths via a glob (repeatable).
  --model-re REGEX   Keep only paths matching REGEX (applies after discovery).
  --method-re REGEX  Keep only paths matching REGEX (applies after discovery).
  --quant-root DIR   Root for EfficientQAT/VeOmni outputs. Default: quant_model
  --llmc-root DIR    Root for llmc outputs. Default: baseline/llmc
  --llmc-config-dir DIR  llmc config directory. Default: baseline/llmc/configs/quantization/exp
  --llmc-run         Run llmc configs before evaluation.
  --llmc-filter REGEX    Regex to filter llmc config filenames.
  --llmc-method REGEX    Regex for llmc method in filename (e.g., awq|quik).
  --llmc-model REGEX     Regex for llmc model in filename (e.g., Llama3-8B).
  --include-efficientqat Include EfficientQAT paths even when llmc filters are set.
  --dry-run          Print resolved paths only; do not run eval.
  --hold             Wait for GPUs to be >90% free before running llmc/eval.
  -h, --help         Show this help.

Examples:
  script/contrast.sh --model-re 'Qwen2.5-0.5B' --method-re 'w3g64'
  script/contrast.sh --include /abs/path/to/model --include ./quant_model/Qwen2.5-0.5B/EfficientQAT/w3g64-int3-kd/checkpoints/out
  script/contrast.sh --llmc-root baseline/llmc --model-re 'Llama' --method-re 'awq|gptq'
  script/contrast.sh --llmc-run --llmc-method awq --llmc-model Llama3-8B
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include)
      include_paths+=("$2")
      shift 2
      ;;
    --glob)
      glob_paths+=("$2")
      shift 2
      ;;
    --model-re)
      model_re="$2"
      shift 2
      ;;
    --method-re)
      method_re="$2"
      shift 2
      ;;
    --quant-root)
      quant_root="$2"
      shift 2
      ;;
    --llmc-root)
      llmc_root="$2"
      llmc_out="$llmc_root/output"
      llmc_config_dir="$llmc_root/configs/quantization/exp"
      shift 2
      ;;
    --llmc-config-dir)
      llmc_config_dir="$2"
      shift 2
      ;;
    --llmc-run)
      llmc_run=true
      shift
      ;;
    --llmc-filter)
      llmc_filter_re="$2"
      shift 2
      ;;
    --llmc-method)
      llmc_method_re="$2"
      shift 2
      ;;
    --llmc-model)
      llmc_model_re="$2"
      shift 2
      ;;
    --include-efficientqat)
      include_efficientqat=true
      shift
      ;;
    --hold)
      hold=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
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

# If llmc filters are provided and global filters are not, propagate them
if [[ -n "$llmc_method_re" && -z "$method_re" ]]; then
  method_re="$llmc_method_re"
fi
if [[ -n "$llmc_model_re" && -z "$model_re" ]]; then
  model_re="$llmc_model_re"
fi
if [[ -n "$llmc_method_re" || -n "$llmc_model_re" || -n "$llmc_filter_re" ]]; then
  if [[ "$dry_run" != "true" && "$llmc_run" != "true" ]]; then
    llmc_run=true
  fi
fi

normalize_dir() {
  local p="$1"
  if [[ -d "$p" ]]; then
    (cd "$p" && pwd)
  fi
}

add_candidate() {
  local p="$1"
  local allow_missing="${2:-false}"
  if [[ -d "$p" ]]; then
    candidates+=("$(cd "$p" && pwd)")
  elif [[ "$allow_missing" == "true" ]]; then
    candidates+=("$p")
  fi
}

llmc_expected_save_dirs_from_config() {
  local cfg="$1"
  python3 - <<PY
import yaml

cfg = "$cfg"
with open(cfg, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

save = data.get("save") or {}
save_path = save.get("save_path")
if not save_path:
    raise SystemExit(0)

# Keep this mapping in sync with baseline/llmc/llmc/__main__.py
mapping = [
    ("save_trans", "transformed_model"),
    ("save_vllm", "vllm_quant_model"),
    ("save_lightllm", "lightllm_quant_model"),
    ("save_sgl", "sgl_quant_model"),
    ("save_autoawq", "autoawq_quant_model"),
    ("save_mlcllm", "mlcllm_quant_model"),
    ("save_lightx2v", "lightx2v_quant_model"),
    ("save_fake", "fake_quant_model"),
]

for key, subdir in mapping:
    if save.get(key, False):
        print(f"{save_path}/{subdir}")
PY
}

filter_path() {
  local p="$1"
  if [[ "$include_efficientqat" != "true" ]]; then
    if [[ -n "$llmc_method_re" || -n "$llmc_model_re" ]]; then
      if [[ "$p" == *"/EfficientQAT/"* ]]; then
        return 1
      fi
    fi
  fi
  if [[ -n "$model_re" ]] && [[ ! "$p" =~ $model_re ]]; then
    return 1
  fi
  if [[ -n "$method_re" ]] && [[ ! "$p" =~ $method_re ]]; then
    return 1
  fi
  return 0
}

check_gpu_memory() {
  local threshold=90
  local gpu_ids="$1"

  IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    local free_mem_ratio
    free_mem_ratio=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | awk -F',' '{print ($1/$2)*100}')
    if [ -z "$free_mem_ratio" ]; then
      return 1
    fi
    if (( $(echo "$free_mem_ratio < $threshold" | bc -l) )); then
      return 1
    fi
  done
  return 0
}

show_progress_bar() {
  local duration=600
  local cols=$(tput cols 2>/dev/null || echo 80)
  local bar_width=$((cols - 25))

  for ((i=0; i<=duration; i++)); do
    local percent=$((i * 100 / duration))
    local filled=$((i * bar_width / duration))
    local empty=$((bar_width - filled))

    printf "\r\033[K["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%% | Next check in %2ds" "$percent" "$((duration - i))"
    sleep 1
  done
  printf "\r\033[K"
}

wait_for_gpus_if_needed() {
  if ! "$hold"; then
    return 0
  fi
  local gpu_ids="${CUDA_VISIBLE_DEVICES:-0}"
  echo "Hold mode enabled. Waiting for all GPUs ($gpu_ids) to have >90% free memory..."
  while true; do
    if check_gpu_memory "$gpu_ids"; then
      echo "✓ GPU memory check passed. All GPUs ($gpu_ids) have sufficient memory."
      break
    else
      show_progress_bar
    fi
  done
}

find_unused_port() {
  while true; do
    port=$(shuf -i 10000-60000 -n 1)
    if ! ss -tuln | grep -q ":$port "; then
      echo "$port"
      return 0
    fi
  done
}

run_llmc_configs() {
  local no_save_cfgs
  local cfg_dirs

  if [[ ! -d "$llmc_config_dir" ]]; then
    echo "Error: llmc config dir not found: $llmc_config_dir" >&2
    exit 1
  fi

  mapfile -t configs < <(ls "$llmc_config_dir"/*.yml 2>/dev/null | sort)
  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "Error: no llmc configs found in $llmc_config_dir" >&2
    exit 1
  fi

  selected=()
  for cfg in "${configs[@]}"; do
    base=$(basename "$cfg")
    if [[ -n "$llmc_filter_re" ]] && [[ ! "$base" =~ $llmc_filter_re ]]; then
      continue
    fi
    if [[ -n "$llmc_method_re" ]] && [[ ! "$base" =~ $llmc_method_re ]]; then
      continue
    fi
    if [[ -n "$llmc_model_re" ]] && [[ ! "$base" =~ $llmc_model_re ]]; then
      continue
    fi
    selected+=("$cfg")
  done

  if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No llmc configs matched." >&2
    exit 1
  fi

  echo "Matched llmc configs (${#selected[@]}):"
  for c in "${selected[@]}"; do
    echo "  $c"
  done

  llmc_effective_save_dirs=()
  no_save_cfgs=()
  for cfg in "${selected[@]}"; do
    mapfile -t cfg_dirs < <(llmc_expected_save_dirs_from_config "$cfg")
    if [[ ${#cfg_dirs[@]} -eq 0 ]]; then
      no_save_cfgs+=("$cfg")
      continue
    fi
    for d in "${cfg_dirs[@]}"; do
      llmc_effective_save_dirs+=("$d")
    done
  done

  if [[ ${#no_save_cfgs[@]} -gt 0 ]]; then
    echo "Error: selected llmc configs have no enabled model save target (save_trans/save_fake/save_vllm/...):" >&2
    for c in "${no_save_cfgs[@]}"; do
      echo "  $c" >&2
    done
    echo "Update llmc configs under $llmc_config_dir to enable at least one save_* flag." >&2
    exit 1
  fi

  if "$dry_run"; then
    return 0
  fi

  wait_for_gpus_if_needed
  export PYTHONPATH="$llmc_root:$llmc_root/lmms-eval:$PYTHONPATH"
  local nnodes="${NNODES:-1}"
  local nproc_per_node="${NPROC_PER_NODE:-1}"
  local log_root="${llmc_out}/logs/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$log_root"
  echo "llmc logs will be written to: $log_root"

  for cfg in "${selected[@]}"; do
    UNUSED_PORT=$(find_unused_port)
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=$UNUSED_PORT
    task_id=$UNUSED_PORT

    echo "Running llmc: $cfg"
    export TORCHELASTIC_ERROR_FILE="$log_root/elastic_error_${task_id}.log"
    torchrun \
      --nnodes "$nnodes" \
      --nproc_per_node "$nproc_per_node" \
      --rdzv_id "$task_id" \
      --rdzv_backend c10d \
      --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
      --log_dir "$log_root" \
      "$llmc_root/llmc/__main__.py" --config "$cfg" --task_id "$task_id"
  done
}

candidates=()

# Planned llmc outputs from configs (useful for --dry-run)
if [[ -d "$llmc_config_dir" ]]; then
  mapfile -t llmc_cfgs < <(ls "$llmc_config_dir"/*.yml 2>/dev/null | sort)
  for cfg in "${llmc_cfgs[@]}"; do
    base=$(basename "$cfg")
    if [[ -n "$llmc_filter_re" ]] && [[ ! "$base" =~ $llmc_filter_re ]]; then
      continue
    fi
    if [[ -n "$llmc_method_re" ]] && [[ ! "$base" =~ $llmc_method_re ]]; then
      continue
    fi
    if [[ -n "$llmc_model_re" ]] && [[ ! "$base" =~ $llmc_model_re ]]; then
      continue
    fi
    mapfile -t cfg_dirs < <(llmc_expected_save_dirs_from_config "$cfg")
    for d in "${cfg_dirs[@]}"; do
      add_candidate "$d" "true"
    done
  done
fi

# Explicit includes
for p in "${include_paths[@]}"; do
  if n=$(normalize_dir "$p"); then
    candidates+=("$n")
  else
    echo "Warning: include path not found: $p" >&2
  fi
done

# Globs
for g in "${glob_paths[@]}"; do
  for p in $g; do
    add_candidate "$p" "false"
  done
done

# Discover from quant_root (EfficientQAT/VeOmni outputs)
if [[ -d "$quant_root" ]]; then
  if [[ -n "$llmc_method_re" || -n "$llmc_model_re" ]]; then
    if "$include_efficientqat"; then
      while IFS= read -r -d '' p; do
        add_candidate "$p" "false"
      done < <(find "$quant_root" -type d -path "*/checkpoints/out" -print0 2>/dev/null)

      while IFS= read -r -d '' f; do
        add_candidate "$(dirname "$f")" "false"
      done < <(find "$quant_root" -type f -name "quantize_config.json" -print0 2>/dev/null)
    fi
  else
    while IFS= read -r -d '' p; do
      add_candidate "$p" "false"
    done < <(find "$quant_root" -type d -path "*/checkpoints/out" -print0 2>/dev/null)

    while IFS= read -r -d '' f; do
      add_candidate "$(dirname "$f")" "false"
    done < <(find "$quant_root" -type f -name "quantize_config.json" -print0 2>/dev/null)
  fi
fi

# Discover from llmc output (if present)
if [[ -d "$llmc_out" ]]; then
  while IFS= read -r -d '' f; do
    dir=$(dirname "$f")
    add_candidate "$dir" "false"
  done < <(find "$llmc_out" -type f \( -name "config.json" -o -name "quantize_config.json" \) -print0 2>/dev/null)
fi

# Optional: run llmc configs first
if "$llmc_run"; then
  run_llmc_configs
  for d in "${llmc_effective_save_dirs[@]}"; do
    add_candidate "$d" "true"
  done
fi

# De-duplicate + filter
mapfile -t unique_paths < <(
  printf "%s\n" "${candidates[@]}" | awk 'NF' | sort -u
)

final_paths=()
for p in "${unique_paths[@]}"; do
  if filter_path "$p"; then
    final_paths+=("$p")
  fi
done

if [[ ${#final_paths[@]} -eq 0 ]]; then
  echo "No evaluation paths found. Use --include/--glob or adjust filters." >&2
  exit 1
fi

echo "Resolved evaluation paths (${#final_paths[@]}):"
for p in "${final_paths[@]}"; do
  echo "  $p"
done

if "$dry_run"; then
  exit 0
fi

if [[ ! -f "$EVAL_SH" ]]; then
  echo "Error: script/eval.sh not found at $EVAL_SH" >&2
  exit 1
fi

missing_paths=()
for p in "${final_paths[@]}"; do
  if [[ ! -d "$p" ]]; then
    missing_paths+=("$p")
  fi
done

if [[ ${#missing_paths[@]} -gt 0 ]]; then
  echo "Missing evaluation paths (${#missing_paths[@]}):" >&2
  for p in "${missing_paths[@]}"; do
    echo "  $p" >&2
  done
  if "$llmc_run"; then
    echo "Some llmc outputs are still missing after --llmc-run. Check llmc logs." >&2
  else
    echo "Run with --llmc-run to generate them, or use --dry-run to inspect." >&2
  fi
  exit 1
fi

wait_for_gpus_if_needed
bash "$EVAL_SH" "${final_paths[@]}" 2>&1 | tee exp.log
