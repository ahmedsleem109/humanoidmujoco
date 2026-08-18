"""Unitree G1 single-obstacle traversal environment (Phase 2 / T2.1).

Derived from the **flat** config, not the rough one. That is deliberate: the
rough config attaches a pelvis-mounted `terrain_scan` raycast height-scanner,
and this project's scope explicitly excludes height-map perception. Obstacle
geometry reaches the policy only through the privileged 5-D vector in
`obstacle_mdp.py`, which keeps the research question ("can RL learn the physical
behaviour?") separate from perception.

Three task variants, all sharing one terrain class:

  Unitree-G1-Obstacle-Fixed   -- one height everywhere (T2.4)
  Unitree-G1-Obstacle         -- height ladder via terrain rows (T2.5)
  Unitree-G1-Obstacle-Random  -- randomised geometry (T3.2)
"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from src.tasks.velocity.mdp import obstacle_mdp
from src.terrains.single_obstacle import (
  make_curriculum_obstacle_cfg,
  make_fixed_obstacle_cfg,
  make_randomized_obstacle_cfg,
)

# One tile per difficulty row. 10 rows spans 10 cm -> 40 cm in 3.3 cm steps.
NUM_ROWS = 10
NUM_COLS = 4
TILE_SIZE = (8.0, 4.0)


def _obstacle_terrain(sub_terrain, num_rows: int = NUM_ROWS) -> TerrainEntityCfg:
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=TILE_SIZE,
      num_rows=num_rows,
      num_cols=NUM_COLS,
      # Allocation mode: one terrain type per column. NOT the learning
      # curriculum -- that is the terrain_levels term below.
      curriculum=True,
      sub_terrains={"obstacle": sub_terrain},
      difficulty_range=(0.0, 1.0),
      # Run-out apron around the whole grid, flush with the tile floor
      # (border_height=1.0 puts the border's top face at z=0).
      #
      # Measured at Gate G3 with border_width=0.0: 50% of all terminations
      # were the robot walking off the edge of the generated floor and falling
      # into the void -- 40% off an x edge, 9% off a y edge -- against only
      # 31% that were genuine falls at the obstacle. The twist command is
      # sampled uniformly, so envs are routinely told to walk backwards or
      # sideways, and an 8 x 4 m tile with a spawn 1.5 m in is easily crossed
      # inside the 20 s episode.
      #
      # Left as it was, both the Baseline 2 number and the G4 reward signal
      # would have been dominated by "fell off the world" rather than by
      # obstacle traversal.
      border_width=5.0,
      # "none" repaints every terrain geom a uniform (0.5, 0.5, 0.5)
      # (terrain_generator.py:327), which threw away the green->red difficulty
      # colouring `single_obstacle.py` computes and left a grey 10 cm box on a
      # grey floor. Gate G4's video was then unreviewable: a human cannot
      # confirm the robot stepped OVER the obstacle rather than through it
      # when the obstacle is not visible. Since traversal_success reads only
      # root position and uprightness, that made the human sign-off the gate
      # demands impossible to actually perform.
      #
      # "height" applies the per-geometry colour instead, so the box is green
      # at 10 cm and red at 40 cm.
      color_scheme="height",
    ),
    max_init_terrain_level=0,
  )


def _face_the_obstacle(cfg: ManagerBasedRlEnvCfg) -> None:
  """Point the commanded heading at the obstacle instead of anywhere.

  Measured on the Gate G4 policy over 192 episodes
  (`scripts/conditional_success.py`):

      raw traversal success        0.193
      episodes that APPROACHED     0.380   (within 0.5 m of the near face)
      success GIVEN it approached  0.507
      median closest approach      1.20 m

  With `heading_command=True` and `rel_heading_envs=1.0`, the heading is drawn
  from the full circle, so **62% of episodes never come near the box at all** --
  the robot is told to walk parallel to it or away from it and the episode ends
  without the obstacle ever being tested. Those score as failures.

  Two consequences, both bad:

    * the raw metric is capped at ~0.38 no matter how good the policy is, so
      Gate G4's 0.70 criterion was unreachable by construction;
    * the obstacle rewards carry signal in only 38% of episodes, so most of the
      training compute is spent where the obstacle is irrelevant.

  Narrowing the heading to roughly +/-23 degrees of +x, and keeping forward
  speed positive, makes the task actually about traversal and multiplies the
  useful training signal ~2.5x.

  This is a deliberate narrowing of scope, not a metric fix: the resulting
  policy is a traversal specialist that is no longer asked to walk backwards or
  strafe. Phase 3 should widen it again once traversal itself is solved.
  """
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (0.3, 1.0)   # always advancing, never reversing
  twist.ranges.lin_vel_y = (-0.3, 0.3)  # mild strafing only
  twist.ranges.ang_vel_z = (-0.5, 0.5)
  twist.ranges.heading = (-0.4, 0.4)    # radians, ~+/-23 deg about +x
  twist.rel_standing_envs = 0.0         # a standing robot never meets the box


def _apply_obstacle_mdp(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add privileged observation, success metric, and traversal rewards."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["obstacle"] = ObservationTermCfg(
      func=obstacle_mdp.obstacle_geometry
    )

  # Measured independently of reward -- this is what the gates read.
  cfg.metrics["traversal_success"] = MetricsTermCfg(
    func=obstacle_mdp.traversal_success
  )

  # Minimum viable reward set. plan.md's rule: add terms only when a diagnosed
  # failure justifies one. Foot-clearance and energy terms stay out until the
  # failure videos say they are needed.
  #
  # Both weights start at 0 and are raised by the curriculum below. Gate G4
  # attempt 1 started them at full strength against a policy that had never
  # seen an obstacle, and the gait was gone before it could adapt. Letting the
  # locomotion terms re-stabilise on the new terrain first costs ~500
  # iterations and is the cheapest insurance available.
  cfg.rewards["obstacle_progress"] = RewardTermCfg(
    func=obstacle_mdp.obstacle_progress, weight=0.0
  )
  cfg.rewards["traversal_bonus"] = RewardTermCfg(
    func=obstacle_mdp.traversal_bonus, weight=0.0
  )


# num_steps_per_env=24, so an iteration is 24 env steps and the curriculum's
# `step` thresholds are in env steps.
_ITER = 24


def _ramp_disabled() -> bool:
  """True when the run continues a policy that has already adapted.

  The ramp is keyed on `common_step_counter`, which restarts at 0 for every
  freshly built env. On a *continuation* run that is actively harmful: a policy
  already trained to full obstacle weight would have its crossing reward cut to
  zero for 400 iterations and then faded back in, discarding the behaviour the
  previous run paid for. Set G1_OBSTACLE_RAMP=off when resuming such a policy.
  """
  return os.environ.get("G1_OBSTACLE_RAMP", "").lower() in ("off", "0", "false")


def _ramp_obstacle_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Fade the obstacle rewards in over the first ~1500 iterations.

  `mdp.reward_weight` applies a stage when `common_step_counter > stage.step`,
  so the value declared on the RewardTermCfg is what is used until the first
  threshold is crossed -- which is why both terms are declared at 0.0 above
  rather than at their final values.
  """
  # traversal_bonus 20 -> 100, from measurements at iteration ~2700 of attempt 2.
  #
  # `Episode_Reward/<term>` is `episode_sum / max_episode_length_s` (20 s), and
  # `episode_sum = weight * dt * sum(func)` with dt = 0.02. That makes the
  # reported numbers convertible into per-episode currency:
  #
  #   one fall        is_terminated -0.0083 -> 200 * 0.02      = 4.0  per fall
  #   one crossing    traversal_bonus                 W * 0.02 = 0.02W
  #   total positives (foot_gait + pose + track_*)     ~24.4    per episode
  #
  # At W = 20 a crossing was worth 0.4, i.e. **1.6%** of an episode's positive
  # reward, against `obstacle_progress` netting about -0.4 because commands
  # point away from the box as often as toward it. The net incentive to cross
  # was approximately zero, and the measured result was a policy with an
  # excellent gait (6.3% falls, 972/1000 step episodes) that crossed only 15.6%
  # of the time -- barely above the 10.9% baseline.
  #
  # The ceiling is set by the fall penalty, not by taste: if a crossing pays
  # more than the 4.0 a fall costs, then "lunge across, then fall" is net
  # positive and attempt 1's exploit comes straight back in a new form.
  #
  #   0.02 * W < 4.0   ->   W < 200
  #
  # W = 100 gives a crossing 2.0, which is 8.2% of the positive reward (5x the
  # previous incentive) while leaving cross-then-fall at 2.0 - 4.0 = -2.0, so it
  # is still strictly punished. Deliberately not the ~300 first considered,
  # which sits above the ceiling and would have re-created the lunge.
  for name, final in (("obstacle_progress", 1.0), ("traversal_bonus", 100.0)):
    cfg.curriculum[f"ramp_{name}"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": name,
        "weight_stages": [
          {"step": 400 * _ITER, "weight": 0.25 * final},
          {"step": 800 * _ITER, "weight": 0.50 * final},
          {"step": 1500 * _ITER, "weight": final},
        ],
      },
    )


def _tune_sim_for_boxes(cfg: ManagerBasedRlEnvCfg) -> None:
  """Flat-terrain contact budgets are too small for box collisions.

  The flat config sets njmax=300 / nconmax=None / ccd_iterations=50 because a
  plane generates few contacts. A box edge strike generates many more.

  Measured at Gate G2 (`scripts/g2_contact_probe.py`, 256 envs, 400 steps of a
  zero-action collapse -- the worst case for simultaneous body-ground
  contacts):

      peak contacts   7475 of the 40 x 256 = 10240 pool  (73%, 29.2 per world)
      peak nefc       116 per world against njmax = 800  (14%)

  73% peak leaves too little headroom once box-edge contacts are added on top,
  so nconmax goes to 56 (-> ~52% on the same measurement). njmax is already
  ~7x the measured peak and stays as is; VRAM was never the binding constraint
  here (T1.1), so the over-allocation is cheaper than a re-measurement.
  """
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 56
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.mujoco.ccd_iterations = 200


def _fix_play_event_order(cfg: ManagerBasedRlEnvCfg) -> None:
  """Move `randomize_terrain` ahead of the root-state resets in play mode.

  The G1 play config appends `randomize_terrain` to `cfg.events`, so it runs
  *after* `reset_base`. `reset_base` reads `env.scene.env_origins` at call
  time, while `randomize_terrain` reassigns `terrain_levels` / `terrain_types`
  and rewrites those origins. The robot therefore lands on the tile it was
  assigned *before* the shuffle, while the privileged obstacle observation --
  which indexes `flat_patches[terrain_levels, terrain_types]` -- describes the
  tile it was assigned *after*. Measured at Gate G2: lateral error up to a full
  tile width (4 m) instead of the ~0.5 m reset noise.

  Harmless on the flat baseline (plane terrain has no `terrain_origins`, so
  `randomize_env_origins` returns immediately), which is why Gate G1 was
  unaffected. It is not harmless here: every traversal number would have been
  measured against the wrong box.
  """
  if "randomize_terrain" not in cfg.events:
    return
  term = cfg.events.pop("randomize_terrain")
  cfg.events = {"randomize_terrain": term, **cfg.events}


def _base_obstacle_cfg(play: bool, sub_terrain, num_rows: int):
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.terrain = _obstacle_terrain(sub_terrain, num_rows=num_rows)
  _apply_obstacle_mdp(cfg)
  _tune_sim_for_boxes(cfg)
  _fix_play_event_order(cfg)
  # Applied to BOTH train and play. If evaluation kept the wide heading, the
  # gate would measure a different task from the one being trained, and the
  # reported success rate would still be dominated by episodes that never meet
  # the obstacle.
  _face_the_obstacle(cfg)
  # The command curriculum re-widens lin_vel_x/y at its stages and would undo
  # the narrowing above partway through training.
  cfg.curriculum.pop("command_vel", None)
  if play:
    # Play mode clears cfg.curriculum, so the ramp never runs and the weights
    # would stay at their declared 0.0. Evaluation does not use rewards for
    # anything the gates read, but leaving them at zero would make the reward
    # columns of a gate report silently meaningless.
    # Must match the ramp's final weights in `_ramp_obstacle_rewards`, or a
    # gate report's reward columns describe a different function from the one
    # that was trained.
    cfg.rewards["obstacle_progress"].weight = 1.0
    cfg.rewards["traversal_bonus"].weight = 100.0
  elif _ramp_disabled():
    cfg.rewards["obstacle_progress"].weight = 1.0
    cfg.rewards["traversal_bonus"].weight = 100.0
  else:
    _ramp_obstacle_rewards(cfg)
  return cfg


def unitree_g1_obstacle_fixed_env_cfg(
  play: bool = False, height_m: float = 0.10
) -> ManagerBasedRlEnvCfg:
  """T2.4 -- one fixed height, no curriculum, no randomisation."""
  cfg = _base_obstacle_cfg(
    play, make_fixed_obstacle_cfg(height_m, size=TILE_SIZE), num_rows=1
  )
  cfg.curriculum.pop("terrain_levels", None)
  return cfg


def unitree_g1_obstacle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """T2.5 -- height ladder driven by terrain rows.

  Re-enables `terrain_levels_vel`, which the flat config popped. It promotes an
  env to a harder row when it walks far enough and demotes it when it does not.
  Because envs stay distributed across rows and demotion is possible, easy
  obstacles are never removed from the batch -- which is what plan.md's M4.3
  anti-forgetting requirement asks for, at no extra cost.
  """
  cfg = _base_obstacle_cfg(
    play, make_curriculum_obstacle_cfg(size=TILE_SIZE), num_rows=NUM_ROWS
  )
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_vel, params={"command_name": "twist"}
  )
  return cfg


def unitree_g1_obstacle_random_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """T3.2 -- randomised width/depth/placement, height still on difficulty."""
  cfg = _base_obstacle_cfg(
    play, make_randomized_obstacle_cfg(size=TILE_SIZE), num_rows=NUM_ROWS
  )
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_vel, params={"command_name": "twist"}
  )
  return cfg


__all__ = [
  "unitree_g1_obstacle_env_cfg",
  "unitree_g1_obstacle_fixed_env_cfg",
  "unitree_g1_obstacle_random_env_cfg",
  "replace",
]
