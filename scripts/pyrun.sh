#!/usr/bin/env bash
# Generic runner: pyrun.sh <script-relative-to-repo> [args...]
# Sets the same environment every other launcher uses. Run from a file, never
# inline via `wsl -- bash -c` (see HANDOFF.md gotcha 1).
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib:$HOME/.local/bin
export WANDB_MODE=disabled
export MUJOCO_GL=${MUJOCO_GL:-egl}
REPO=/mnt/d/humanoid/unitree_rl_mjlab
cd "$REPO" || exit 1
"$HOME/venvs/mjlab/bin/python" -u "$@"
echo "PYRUN_EXIT=$?"
