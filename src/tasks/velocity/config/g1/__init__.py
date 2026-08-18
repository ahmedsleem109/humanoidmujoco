from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .obstacle_env_cfg import (
  unitree_g1_obstacle_env_cfg,
  unitree_g1_obstacle_fixed_env_cfg,
  unitree_g1_obstacle_random_env_cfg,
)
from .robust_env_cfg import (
  unitree_g1_robust_p1_nominal,
  unitree_g1_robust_p2_push,
  unitree_g1_robust_p3_robust,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# --- Obstacle traversal (feature/g1-obstacle-traversal) ---

register_mjlab_task(
  task_id="Unitree-G1-Obstacle-Fixed",
  env_cfg=unitree_g1_obstacle_fixed_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_fixed_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Obstacle",
  env_cfg=unitree_g1_obstacle_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Obstacle-Random",
  env_cfg=unitree_g1_obstacle_random_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_random_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# --- Robustness ablation (Robust Humanoid Locomotion Under Disturbances) ---
#
# Three arms of one controlled experiment. Identical reward, observations,
# commands, network and horizon; only the event dict differs. See
# robust_env_cfg.py for why each departure from the mjlab default was made.

register_mjlab_task(
  task_id="Unitree-G1-Robust-P1-Nominal",
  env_cfg=unitree_g1_robust_p1_nominal(),
  play_env_cfg=unitree_g1_robust_p1_nominal(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Robust-P2-Push",
  env_cfg=unitree_g1_robust_p2_push(),
  play_env_cfg=unitree_g1_robust_p2_push(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Robust-P3-Robust",
  env_cfg=unitree_g1_robust_p3_robust(),
  play_env_cfg=unitree_g1_robust_p3_robust(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
