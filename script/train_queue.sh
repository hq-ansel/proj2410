#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_DIR="$ROOT_DIR/script/.queue"
JOBS_FILE="$QUEUE_DIR/jobs.jsonl"
LOCK_FILE="$QUEUE_DIR/queue.lock"
PID_FILE="$QUEUE_DIR/worker.pid"
WORKER_LOG="$QUEUE_DIR/worker.log"
STOP_FILE="$QUEUE_DIR/worker.stop"
LOG_ROOT="$ROOT_DIR/exp/queue_logs"

TRAIN_SH="$ROOT_DIR/VeOmni/train.sh"
TRAIN_PY="VeOmni/tasks/quantize/train.py"
EVAL_SH="$ROOT_DIR/script/eval.sh"

HOLD_FREE_MEM_THRESHOLD="${HOLD_FREE_MEM_THRESHOLD:-90}"
HOLD_CHECK_INTERVAL="${HOLD_CHECK_INTERVAL:-600}"
QUEUE_POLL_INTERVAL="${QUEUE_POLL_INTERVAL:-15}"

STOP_REQUESTED=0

usage() {
  cat <<USAGE
Usage:
  bash script/train_queue.sh init
  bash script/train_queue.sh start
  bash script/train_queue.sh stop
  bash script/train_queue.sh submit <config.yaml> [more.yaml ...]
  bash script/train_queue.sh remove <job_id>
  bash script/train_queue.sh list
  bash script/train_queue.sh status
  bash script/train_queue.sh tail [job_id]
  bash script/train_queue.sh retry <job_id>
USAGE
}

now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

ensure_layout() {
  mkdir -p "$QUEUE_DIR" "$LOG_ROOT"
  touch "$JOBS_FILE" "$LOCK_FILE" "$WORKER_LOG"
}

log_worker() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$WORKER_LOG"
}

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

get_worker_pid() {
  if [[ -f "$PID_FILE" ]]; then
    cat "$PID_FILE"
  fi
}

is_worker_running() {
  local pid
  pid="$(get_worker_pid || true)"
  is_pid_running "$pid"
}

parse_output_dir() {
  local yaml_file="$1"
  python3 - "$yaml_file" <<'PY'
import sys
try:
    import yaml
except Exception:
    print("")
    raise SystemExit(0)

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    train = data.get("train") or {}
    out = train.get("output_dir")
    if out is None:
        out = ""
    print(str(out))
except Exception:
    print("")
PY
}

resolve_eval_path() {
  local out_dir="$1"
  local checkpoints_dir="$out_dir/checkpoints"
  local quant_cfg="$checkpoints_dir/out/quantize_config.json"
  local quant_type=""

  if [[ -f "$quant_cfg" ]]; then
    quant_type=$(python3 - "$quant_cfg" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        print((json.load(f) or {}).get("quant_type", ""))
except Exception:
    print("")
PY
)
  fi

  if [[ "$quant_type" == "mixed" && -d "$checkpoints_dir/out_dequant" ]]; then
    echo "$checkpoints_dir/out_dequant"
    return 0
  fi
  if [[ -d "$checkpoints_dir/out" ]]; then
    echo "$checkpoints_dir/out"
    return 0
  fi
  if [[ -d "$checkpoints_dir/out_dequant" ]]; then
    echo "$checkpoints_dir/out_dequant"
    return 0
  fi

  local latest_hf=""
  local latest_step=-1
  if [[ -d "$checkpoints_dir" ]]; then
    for d in "$checkpoints_dir"/global_step_*; do
      [[ -d "$d" ]] || continue
      local step="${d##*_}"
      [[ "$step" =~ ^[0-9]+$ ]] || continue
      if [[ -d "$d/hf_ckpt" ]] && (( step > latest_step )); then
        latest_step="$step"
        latest_hf="$d/hf_ckpt"
      fi
    done
  fi
  if [[ -n "$latest_hf" ]]; then
    echo "$latest_hf"
    return 0
  fi

  return 1
}

check_gpu_memory() {
  local threshold="$1"
  local gpu_ids="$2"

  IFS=',' read -ra gpu_array <<< "$gpu_ids"
  for gpu_id in "${gpu_array[@]}"; do
    local free_mem_ratio
    free_mem_ratio=$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | awk -F',' '{print ($1/$2)*100}')
    if [[ -z "$free_mem_ratio" ]]; then
      return 1
    fi
    if (( $(echo "$free_mem_ratio < $threshold" | bc -l) )); then
      return 1
    fi
  done
  return 0
}

show_progress_bar() {
  local duration="$1"
  local cols bar_width i percent filled empty
  cols=$(tput cols 2>/dev/null || echo 80)
  bar_width=$((cols - 36))
  if (( bar_width < 10 )); then
    bar_width=10
  elif (( bar_width > 60 )); then
    bar_width=60
  fi

  for ((i=0; i<=duration; i++)); do
    percent=$((i * 100 / duration))
    filled=$((i * bar_width / duration))
    empty=$((bar_width - filled))

    printf "\r\033[2K["
    printf "%${filled}s" "" | tr ' ' '#'
    printf "%${empty}s" "" | tr ' ' '-'
    printf "] %3d%% | Next check in %3ds" "$percent" "$((duration - i))"
    sleep 1
  done
  printf "\r\033[2K"
}

wait_for_gpus_if_needed() {
  local gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  local threshold="$HOLD_FREE_MEM_THRESHOLD"
  local interval="$HOLD_CHECK_INTERVAL"

  log_worker "Hold check enabled. Waiting for GPUs ($gpu_ids) free_mem>${threshold}%"
  while true; do
    if check_gpu_memory "$threshold" "$gpu_ids"; then
      log_worker "GPU memory check passed for $gpu_ids"
      return 0
    fi
    show_progress_bar "$interval"
  done
}

submit_job_locked() {
  local config_path="$1"
  local output_dir="$2"
  local submit_ts="$3"
  local job_id
  job_id=$(python3 - <<'PY'
import datetime
import uuid

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
print(f"{now}_{uuid.uuid4().hex[:6]}")
PY
)

  python3 - "$JOBS_FILE" "$job_id" "$config_path" "$output_dir" "$submit_ts" <<'PY'
import json
import os
import sys

jobs_file, job_id, config_path, output_dir, submit_ts = sys.argv[1:6]
existing_job_id = None
if os.path.exists(jobs_file):
    with open(jobs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except Exception:
                continue
            if (
                job.get("config_path") == config_path
                and job.get("status") in {"pending", "running"}
            ):
                existing_job_id = job.get("job_id")
                break

if existing_job_id:
    print(f"DUP:{existing_job_id}")
    raise SystemExit(0)

job = {
    "job_id": job_id,
    "config_path": config_path,
    "output_dir": output_dir,
    "status": "pending",
    "submit_time": submit_ts,
    "start_time": None,
    "end_time": None,
    "attempt": 1,
    "max_attempts": 1,
    "exit_code": None,
    "train_log": None,
    "eval_log": None,
    "error_type": None,
}
os.makedirs(os.path.dirname(jobs_file), exist_ok=True)
with open(jobs_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(job, ensure_ascii=False) + "\n")
print(f"NEW:{job_id}")
PY
}

submit_job() {
  local config_path="$1"
  if [[ ! -f "$config_path" ]]; then
    echo "Error: config not found: $config_path" >&2
    exit 1
  fi

  ensure_layout
  local output_dir submit_ts job_id abs_config result
  abs_config="$(python3 - "$config_path" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"
  output_dir="$(parse_output_dir "$config_path" || true)"
  submit_ts="$(now_iso)"

  result=$( (
    flock -x 9
    submit_job_locked "$abs_config" "$output_dir" "$submit_ts"
  ) 9>"$LOCK_FILE")

  if [[ "$result" == DUP:* ]]; then
    job_id="${result#DUP:}"
    echo "Skipped duplicate: existing active job_id=$job_id"
    echo "  config=$abs_config"
    return 0
  fi
  job_id="${result#NEW:}"

  echo "Submitted: job_id=$job_id"
  echo "  config=$abs_config"
  if [[ -n "$output_dir" ]]; then
    echo "  output_dir=$output_dir"
  else
    echo "  output_dir=<empty>"
  fi
}

submit_jobs() {
  local cfg
  for cfg in "$@"; do
    submit_job "$cfg"
  done
}

dequeue_pending_job_locked() {
  python3 - "$JOBS_FILE" "$(now_iso)" <<'PY'
import json
import os
import sys

jobs_file, now_ts = sys.argv[1:3]
jobs = []
if os.path.exists(jobs_file):
    with open(jobs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))

picked = None
for job in jobs:
    if job.get("status") == "pending":
        job["status"] = "running"
        job["start_time"] = now_ts
        job["end_time"] = None
        job["exit_code"] = None
        job["error_type"] = None
        picked = job
        break

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")

if picked is not None:
    print(json.dumps(picked, ensure_ascii=False))
PY
}

update_job_logs_locked() {
  local job_id="$1"
  local train_log="$2"
  local eval_log="$3"
  python3 - "$JOBS_FILE" "$job_id" "$train_log" "$eval_log" <<'PY'
import json
import sys

jobs_file, job_id, train_log, eval_log = sys.argv[1:5]
jobs = []
with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

for job in jobs:
    if job.get("job_id") == job_id:
        job["train_log"] = train_log
        job["eval_log"] = eval_log
        break

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
PY
}

finalize_job_locked() {
  local job_id="$1"
  local status="$2"
  local exit_code="$3"
  local error_type="$4"
  local end_ts="$5"

  python3 - "$JOBS_FILE" "$job_id" "$status" "$exit_code" "$error_type" "$end_ts" <<'PY'
import json
import sys

jobs_file, job_id, status, exit_code, error_type, end_ts = sys.argv[1:7]

jobs = []
with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

for job in jobs:
    if job.get("job_id") == job_id:
        job["status"] = status
        job["end_time"] = end_ts
        if exit_code == "null":
            job["exit_code"] = None
        else:
            try:
                job["exit_code"] = int(exit_code)
            except Exception:
                job["exit_code"] = exit_code
        job["error_type"] = None if error_type == "null" else error_type
        break

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")
PY
}

recover_running_jobs_locked() {
  python3 - "$JOBS_FILE" "$(now_iso)" <<'PY'
import json
import os
import sys

jobs_file, now_ts = sys.argv[1:3]
if not os.path.exists(jobs_file):
    print(0)
    raise SystemExit(0)

jobs = []
recovered = 0
with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

for job in jobs:
    if job.get("status") == "running":
        job["status"] = "pending"
        job["start_time"] = None
        job["end_time"] = now_ts
        job["error_type"] = "recovered_from_crash"
        job["exit_code"] = None
        try:
            job["attempt"] = int(job.get("attempt", 1)) + 1
        except Exception:
            job["attempt"] = 2
        recovered += 1

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")

print(recovered)
PY
}

retry_job_locked() {
  local job_id="$1"
  python3 - "$JOBS_FILE" "$job_id" <<'PY'
import json
import sys

jobs_file, job_id = sys.argv[1:3]
jobs = []
found = False

with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

for job in jobs:
    if job.get("job_id") != job_id:
        continue
    found = True
    if job.get("status") == "running":
        print("ERROR:running")
        raise SystemExit(0)
    try:
        job["attempt"] = int(job.get("attempt", 1)) + 1
    except Exception:
        job["attempt"] = 2
    job["status"] = "pending"
    job["start_time"] = None
    job["end_time"] = None
    job["exit_code"] = None
    job["error_type"] = None
    break

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in jobs:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")

if not found:
    print("ERROR:not_found")
else:
    print("OK")
PY
}

list_jobs() {
  ensure_layout
  python3 - "$JOBS_FILE" <<'PY'
import json
import sys

jobs_file = sys.argv[1]
jobs = []
with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

if not jobs:
    print("No jobs.")
    raise SystemExit(0)

print(f"{'JOB_ID':<24} {'STATUS':<10} {'ATTEMPT':<7} {'SUBMIT_TIME':<20} CONFIG")
for job in jobs:
    print(
        f"{job.get('job_id',''):<24} "
        f"{job.get('status',''):<10} "
        f"{str(job.get('attempt','')):<7} "
        f"{str(job.get('submit_time','')):<20} "
        f"{job.get('config_path','')}"
    )
PY
}

status_jobs() {
  ensure_layout
  local worker_state="stopped"
  local worker_pid=""
  if is_worker_running; then
    worker_state="running"
    worker_pid="$(get_worker_pid)"
  elif [[ -f "$PID_FILE" ]]; then
    worker_state="stale_pid"
    worker_pid="$(get_worker_pid)"
  fi

  echo "Worker: $worker_state${worker_pid:+ (pid=$worker_pid)}"

  python3 - "$JOBS_FILE" <<'PY'
import json
import sys
from collections import Counter

jobs_file = sys.argv[1]
count = Counter()
running_ids = []

with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        st = job.get("status", "unknown")
        count[st] += 1
        if st == "running":
            running_ids.append(job.get("job_id", ""))

print(
    "Jobs: "
    f"pending={count.get('pending',0)}, "
    f"running={count.get('running',0)}, "
    f"success={count.get('success',0)}, "
    f"failed={count.get('failed',0)}"
)
if running_ids:
    print("Running job_ids:", ", ".join(running_ids))
PY
}

tail_logs() {
  ensure_layout
  if [[ $# -eq 0 ]]; then
    tail -n 200 -F "$WORKER_LOG"
    return
  fi

  local job_id="$1"
  local paths
  paths=$(python3 - "$JOBS_FILE" "$job_id" "$LOG_ROOT" <<'PY'
import json
import os
import sys

jobs_file, job_id, log_root = sys.argv[1:4]
job = None
with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        if j.get("job_id") == job_id:
            job = j
            break

if not job:
    print("ERROR:not_found")
    raise SystemExit(0)

train_log = job.get("train_log") or os.path.join(log_root, job_id, "train.log")
eval_log = job.get("eval_log") or os.path.join(log_root, job_id, "eval.log")
print(train_log)
print(eval_log)
PY
)

  if [[ "$paths" == ERROR:not_found* ]]; then
    echo "Error: job not found: $job_id" >&2
    exit 1
  fi

  mapfile -t path_array <<< "$paths"
  local existing=()
  for p in "${path_array[@]}"; do
    if [[ -n "$p" ]]; then
      existing+=("$p")
    fi
  done
  if [[ ${#existing[@]} -eq 0 ]]; then
    echo "No log files available for job $job_id" >&2
    exit 1
  fi
  tail -n 200 -F "${existing[@]}"
}

start_worker() {
  ensure_layout
  rm -f "$STOP_FILE"

  if is_worker_running; then
    echo "Worker already running (pid=$(get_worker_pid))."
    return
  fi

  if [[ -f "$PID_FILE" ]]; then
    rm -f "$PID_FILE"
  fi

  nohup bash "$0" worker >>"$WORKER_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Worker started (pid=$pid)."
}

stop_worker() {
  ensure_layout
  if ! is_worker_running; then
    echo "Worker is not running."
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    return
  fi

  local pid
  pid="$(get_worker_pid)"
  touch "$STOP_FILE"
  kill -TERM "$pid"
  echo "Stop signal sent to worker pid=$pid (graceful shutdown after current job)."
}

retry_job() {
  local job_id="$1"
  ensure_layout

  local result
  result=$( (
    flock -x 9
    retry_job_locked "$job_id"
  ) 9>"$LOCK_FILE")

  case "$result" in
    OK)
      echo "Job requeued: $job_id"
      ;;
    ERROR:running)
      echo "Error: job is currently running, cannot retry: $job_id" >&2
      exit 1
      ;;
    ERROR:not_found)
      echo "Error: job not found: $job_id" >&2
      exit 1
      ;;
    *)
      echo "Error: unexpected retry result: $result" >&2
      exit 1
      ;;
  esac
}

remove_job_locked() {
  local job_id="$1"
  python3 - "$JOBS_FILE" "$job_id" <<'PY'
import json
import sys

jobs_file, job_id = sys.argv[1:3]
jobs = []
found = False
removed = False
invalid_state = None

with open(jobs_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            jobs.append(json.loads(line))

kept = []
for job in jobs:
    if job.get("job_id") != job_id:
        kept.append(job)
        continue
    found = True
    st = job.get("status")
    if st == "pending":
        removed = True
        continue
    invalid_state = st
    kept.append(job)

with open(jobs_file, "w", encoding="utf-8") as f:
    for job in kept:
        f.write(json.dumps(job, ensure_ascii=False) + "\n")

if not found:
    print("ERROR:not_found")
elif removed:
    print("OK")
else:
    print(f"ERROR:not_pending:{invalid_state}")
PY
}

remove_job() {
  local job_id="$1"
  ensure_layout

  local result
  result=$( (
    flock -x 9
    remove_job_locked "$job_id"
  ) 9>"$LOCK_FILE")

  case "$result" in
    OK)
      echo "Removed pending job: $job_id"
      ;;
    ERROR:not_found)
      echo "Error: job not found: $job_id" >&2
      exit 1
      ;;
    ERROR:not_pending:*)
      echo "Error: only pending jobs can be removed. job_id=$job_id status=${result#ERROR:not_pending:}" >&2
      exit 1
      ;;
    *)
      echo "Error: unexpected remove result: $result" >&2
      exit 1
      ;;
  esac
}

worker_handle_term() {
  STOP_REQUESTED=1
  log_worker "TERM received. Worker will stop after current iteration."
}

run_worker_loop() {
  ensure_layout

  trap worker_handle_term TERM INT

  echo "$$" > "$PID_FILE"

  local recovered
  recovered=$( (
    flock -x 9
    recover_running_jobs_locked
  ) 9>"$LOCK_FILE")

  if [[ "$recovered" != "0" ]]; then
    log_worker "Recovered $recovered running job(s) to pending."
  fi

  log_worker "Worker loop started."

  while true; do
    if [[ -f "$STOP_FILE" ]]; then
      STOP_REQUESTED=1
    fi
    if [[ "$STOP_REQUESTED" -eq 1 ]]; then
      log_worker "Stop requested. Exiting worker loop."
      break
    fi

    local job_json
    job_json=$( (
      flock -x 9
      dequeue_pending_job_locked
    ) 9>"$LOCK_FILE")

    if [[ -z "$job_json" ]]; then
      sleep "$QUEUE_POLL_INTERVAL"
      continue
    fi

    local job_id config_path output_dir attempt
    mapfile -t job_fields < <(
      python3 - <<'PY' "$job_json"
import json
import sys

j = json.loads(sys.argv[1])
print(j.get("job_id", ""))
print(j.get("config_path", ""))
print(j.get("output_dir", ""))
print(j.get("attempt", ""))
PY
    )
    job_id="${job_fields[0]:-}"
    config_path="${job_fields[1]:-}"
    output_dir="${job_fields[2]:-}"
    attempt="${job_fields[3]:-}"

    local job_log_dir train_log eval_log
    job_log_dir="$LOG_ROOT/$job_id"
    mkdir -p "$job_log_dir"
    train_log="$job_log_dir/train.log"
    eval_log="$job_log_dir/eval.log"

    (
      flock -x 9
      update_job_logs_locked "$job_id" "$train_log" "$eval_log"
    ) 9>"$LOCK_FILE"

    log_worker "Picked job=$job_id attempt=$attempt config=$config_path"

    wait_for_gpus_if_needed

    if [[ ! -f "$config_path" ]]; then
      log_worker "Job $job_id failed: config_not_found ($config_path)"
      (
        flock -x 9
        finalize_job_locked "$job_id" "failed" "2" "config_not_found" "$(now_iso)"
      ) 9>"$LOCK_FILE"
      continue
    fi

    set +e
    (
      cd "$ROOT_DIR"
      bash "$TRAIN_SH" "$TRAIN_PY" "$config_path"
    ) 2>&1 | tee "$train_log"
    train_exit=${PIPESTATUS[0]}
    set -e

    if [[ "$train_exit" -ne 0 ]]; then
      log_worker "Job $job_id failed in training (exit=$train_exit)"
      (
        flock -x 9
        finalize_job_locked "$job_id" "failed" "$train_exit" "train_failed" "$(now_iso)"
      ) 9>"$LOCK_FILE"
      continue
    fi

    local eval_path=""
    if [[ -n "$output_dir" ]]; then
      eval_path="$(resolve_eval_path "$output_dir" || true)"
    fi

    if [[ -z "$eval_path" ]]; then
      log_worker "Job $job_id failed: eval_path_not_found"
      (
        flock -x 9
        finalize_job_locked "$job_id" "failed" "3" "eval_path_not_found" "$(now_iso)"
      ) 9>"$LOCK_FILE"
      continue
    fi

    if [[ ! -f "$EVAL_SH" ]]; then
      log_worker "Job $job_id failed: eval_script_missing"
      (
        flock -x 9
        finalize_job_locked "$job_id" "failed" "4" "eval_script_missing" "$(now_iso)"
      ) 9>"$LOCK_FILE"
      continue
    fi

    set +e
    (
      cd "$ROOT_DIR"
      bash "$EVAL_SH" "$eval_path"
    ) 2>&1 | tee "$eval_log"
    eval_exit=${PIPESTATUS[0]}
    set -e

    if [[ "$eval_exit" -ne 0 ]]; then
      log_worker "Job $job_id failed in evaluation (exit=$eval_exit)"
      (
        flock -x 9
        finalize_job_locked "$job_id" "failed" "$eval_exit" "eval_failed" "$(now_iso)"
      ) 9>"$LOCK_FILE"
      continue
    fi

    log_worker "Job $job_id success."
    (
      flock -x 9
      finalize_job_locked "$job_id" "success" "0" "null" "$(now_iso)"
    ) 9>"$LOCK_FILE"
  done

  rm -f "$PID_FILE" "$STOP_FILE"
  log_worker "Worker exited."
}

cmd="${1:-}" || true

case "$cmd" in
  init)
    ensure_layout
    echo "Queue initialized at $QUEUE_DIR"
    ;;
  start)
    start_worker
    ;;
  stop)
    stop_worker
    ;;
  submit)
    if [[ $# -lt 2 ]]; then
      echo "Error: submit requires at least one argument: <config.yaml> [more.yaml ...]" >&2
      usage
      exit 1
    fi
    submit_jobs "${@:2}"
    ;;
  remove)
    if [[ $# -ne 2 ]]; then
      echo "Error: remove requires one argument: <job_id>" >&2
      usage
      exit 1
    fi
    remove_job "$2"
    ;;
  list)
    list_jobs
    ;;
  status)
    status_jobs
    ;;
  tail)
    if [[ $# -eq 1 ]]; then
      tail_logs
    elif [[ $# -eq 2 ]]; then
      tail_logs "$2"
    else
      echo "Error: tail accepts zero or one argument [job_id]" >&2
      usage
      exit 1
    fi
    ;;
  retry)
    if [[ $# -ne 2 ]]; then
      echo "Error: retry requires one argument: <job_id>" >&2
      usage
      exit 1
    fi
    retry_job "$2"
    ;;
  worker)
    run_worker_loop
    ;;
  *)
    usage
    exit 1
    ;;
esac
