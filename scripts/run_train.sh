#!/bin/bash
# Launch a training run headlessly and record a completion sentinel.
#
# Video is deliberately NOT recorded during training. mjlab's VideoRecorder
# keeps only env[0] but renders the whole batch, and with MUJOCO_GL unset MuJoCo
# falls back to CPU rasterisation -- at 4096 envs that stalls the run (GPU idle,
# CPU pegged). Videos come from scripts/eval_gate.py, which renders a dedicated
# 1-env pass.
#
# Usage: run_train.sh <TAG> <TASK> <NUM_ENVS> <ITERS> [extra args...]
set -u

TAG=${1:?tag required}
TASK=${2:?task required}
NUM_ENVS=${3:?num_envs required}
ITERS=${4:?iterations required}
shift 4

REPO=/mnt/d/humanoid/unitree_rl_mjlab
PY=$HOME/venvs/mjlab/bin/python
LOG=$HOME/runs/$TAG.log
DONE=$HOME/runs/$TAG.done

mkdir -p "$HOME/runs"
rm -f "$LOG" "$DONE"
cd "$REPO" || exit 1

export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled

"$PY" scripts/train.py "$TASK" \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.max-iterations="$ITERS" \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  "$@" > "$LOG" 2>&1

echo "$?" > "$DONE"
