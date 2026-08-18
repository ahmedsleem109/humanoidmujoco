#!/bin/bash
# Periodic progress report for a long training run, with early-abort milestones.
#
# The point is to find out a run is failing at iteration 500, not at hour 13.
# Emits ONE compact line per check so it can be watched without flooding.
#
# Usage: watch_run.sh <TAG> [INTERVAL_SECONDS]
set -u

TAG=${1:?tag required}
INTERVAL=${2:-1800}
LOG=$HOME/runs/$TAG.log
DONE=$HOME/runs/$TAG.done

# Milestones: iteration -> minimum acceptable mean episode length (steps).
# Target is 900 steps (18 s of the 20 s limit) for Gate G1.
# Derived from the T1.2a smoke run, which reached ~450 by iteration 150.
milestone_floor() {
  local it=$1
  if   [ "$it" -ge 4000 ]; then echo 850
  elif [ "$it" -ge 2000 ]; then echo 800
  elif [ "$it" -ge 1000 ]; then echo 700
  elif [ "$it" -ge 500 ];  then echo 550
  elif [ "$it" -ge 200 ];  then echo 400
  else echo 0
  fi
}

prev_len=0
while true; do
  if [ ! -f "$LOG" ]; then sleep 30; continue; fi

  it=$(grep -cE 'Learning iteration' "$LOG" 2>/dev/null || echo 0)
  len=$(grep -E 'Mean episode length:' "$LOG" | tail -1 | awk '{print $NF}')
  rew=$(grep -E 'Mean reward:' "$LOG" | tail -1 | awk '{print $NF}')
  eta=$(grep -E 'ETA:' "$LOG" | tail -1 | awk '{print $NF}')
  # Average of the last 20 reports, to judge trend past per-iteration noise.
  avg=$(grep -E 'Mean episode length:' "$LOG" | tail -20 | awk '{s+=$NF} END {if(NR>0) printf "%.0f", s/NR; else print 0}')

  [ -z "$len" ] && len=0
  [ -z "$rew" ] && rew=0
  [ -z "$avg" ] && avg=0
  [ -z "$eta" ] && eta="?"

  floor=$(milestone_floor "$it")
  verdict="ok"
  if [ "$floor" -gt 0 ] && [ "${avg%.*}" -lt "$floor" ]; then
    verdict="BELOW_MILESTONE(need>=${floor})"
  elif [ "$prev_len" -gt 0 ] && [ "${avg%.*}" -lt $((prev_len * 80 / 100)) ]; then
    # Ignore the expected dip when commands_vel widens the range at iter 5000.
    if [ "$it" -lt 4800 ] || [ "$it" -gt 5600 ]; then
      verdict="REGRESSING(was ${prev_len})"
    fi
  fi

  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "[$TAG] iter=$it ep_len=$len avg20=$avg reward=$rew gpu=${gpu}C/% eta=$eta -> $verdict"
  prev_len=${avg%.*}

  if [ -f "$DONE" ]; then
    echo "[$TAG] FINISHED exit=$(cat "$DONE")"
    break
  fi
  sleep "$INTERVAL"
done
