#!/bin/bash
# Periodic GPU cooldown for long runs on a laptop GPU.
#
# Pauses the training process with SIGSTOP for a cooldown window, then resumes
# with SIGCONT. The process keeps its CUDA context, warp kernel cache and
# optimiser state -- nothing is reloaded and no progress is lost, unlike
# killing and resuming from a checkpoint.
#
# Reports training metrics at pause, during cooling, and at resume, so the run
# stays observable while it is frozen.
#
# Usage: gpu_cooldown.sh <TAG> [WORK_SECONDS] [COOL_SECONDS] [SAMPLE_SECONDS]
set -u

TAG=${1:?tag required}
WORK=${2:-10800}   # 3 hours
COOL=${3:-900}     # 15 minutes
SAMPLE=${4:-300}   # report every 5 min while cooling
LOG=$HOME/runs/$TAG.log
DONE=$HOME/runs/$TAG.done

gpu() { nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
iters() { grep -cE 'Learning iteration' "$LOG" 2>/dev/null || echo 0; }
metric() { grep -E "$1" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}'; }

report() {
  echo "[cooldown/$1] iter=$(iters) ep_len=$(metric 'Mean episode length:') reward=$(metric 'Mean reward:') gpu=$(gpu)C/%"
}

while true; do
  waited=0
  while [ "$waited" -lt "$WORK" ]; do
    [ -f "$DONE" ] && { echo "[cooldown] run finished; stopping"; exit 0; }
    sleep 30
    waited=$((waited + 30))
  done

  pid=$(pgrep -f 'train[.]py' | head -1)
  if [ -z "$pid" ]; then
    echo "[cooldown] no training process found; stopping"
    exit 0
  fi

  if ! kill -STOP "$pid" 2>/dev/null; then
    echo "[cooldown] failed to pause pid=$pid; continuing without cooldown"
    continue
  fi
  report "paused"

  slept=0
  while [ "$slept" -lt "$COOL" ]; do
    sleep "$SAMPLE"
    slept=$((slept + SAMPLE))
    [ "$slept" -lt "$COOL" ] && report "cooling ${slept}s/${COOL}s"
  done

  # Resume, and keep trying: a pause that never lifts would stall the run.
  for attempt in 1 2 3 4 5; do
    kill -CONT "$pid" 2>/dev/null
    sleep 3
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
    case "$state" in
      T*) echo "[cooldown] WARNING resume attempt $attempt failed (state=$state)"; sleep 5 ;;
      "") echo "[cooldown] process gone during cooldown"; exit 1 ;;
      *)  report "resumed"; break ;;
    esac
  done
done
