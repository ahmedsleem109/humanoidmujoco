#!/usr/bin/env bash
# Run a disturbance sweep with the right environment, from a file.
#
# A file rather than an inline command because env vars cannot be set through
# `detach.sh` (which runs `setsid nohup "$@"`), and because passing this through
# PowerShell mangles quotes and eats backslashes (gotchas 18/19).
#
# Usage: run_sweep.sh <CHECKPOINT> <LABEL> [extra eval_push args...]
#
# Override the grid with env vars, e.g.
#   FORCES=750,800,900,1000 run_sweep.sh ckpt.pt label_hi
# P2 saturated the original 0-700 N grid (its lateral F* was never reached), so
# the high end has to be extended for every policy to keep the ablation on one
# comparable axis.
set -u

CKPT=${1:?checkpoint required}
LABEL=${2:?label required}
shift 2

FORCES=${FORCES:-0,50,100,150,200,250,300,350,400,450,500,550,600,650,700}
DIRECTIONS=${DIRECTIONS:-0,90,180,270}
NUM_ENVS=${NUM_ENVS:-64}

REPO=/mnt/d/humanoid/unitree_rl_mjlab
PY=$HOME/venvs/mjlab/bin/python

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib:$HOME/.local/bin
export MUJOCO_GL=egl
export WANDB_MODE=disabled

cd "$REPO" || exit 1

"$PY" -u scripts/eval_push.py \
  --checkpoint "$CKPT" \
  --label "$LABEL" \
  --forces "$FORCES" \
  --directions "$DIRECTIONS" \
  --num-envs "$NUM_ENVS" \
  "$@"

echo "SWEEP_EXIT=$?"
