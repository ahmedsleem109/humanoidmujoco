"""Privileged obstacle observations, traversal success metric, and rewards.

Design note -- how geometry reaches the policy
----------------------------------------------
The terrain generator runs at build time; the policy needs per-env obstacle
geometry at run time. mjlab's ``TerrainEntity`` exposes ``flat_patches`` as
device tensors shaped ``(num_rows, num_cols, num_patches, 3)``, which is the
only per-tile channel from generation to runtime.

The catch (``mjlab/terrains/terrain_generator.py:335``): collected patch values
have the tile's ``world_position`` **added** to them. They are therefore
positions, not free-form storage -- stashing raw scalars like height or depth
would be silently corrupted by the tile offset.

So ``BoxSingleObstacleTerrainCfg`` emits three points that genuinely *are*
world positions, and geometry is recovered from their differences:

  p0 = near-face bottom centre
  p1 = far-face  top    centre
  p2 = side edge at the near face

  depth  = p1.x - p0.x
  height = p1.z - p0.z
  width  = 2 * (p2.y - p0.y)

This survives the world offset exactly, and works unchanged under Phase 3
randomisation -- whereas deriving height from ``terrain_levels`` alone would
only hold while geometry stays deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

OBSTACLE_PATCH = "obstacle"

# Normalisation constants. Chosen so every quantity lands roughly in [-1, 1]
# across the Phase 3 randomisation ranges.
_HEIGHT_SCALE = 0.40  # max obstacle height, m
_WIDTH_SCALE = 4.0  # max tile width, m
_DEPTH_SCALE = 0.50  # max obstacle depth, m
_DIST_SCALE = 4.0  # max meaningful approach distance, m


def _obstacle_points(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Per-env obstacle key points, shape (num_envs, 3, 3). None if unavailable."""
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None:
    return None
  patches = getattr(terrain, "flat_patches", None)
  if not patches or OBSTACLE_PATCH not in patches:
    return None

  arr = patches[OBSTACLE_PATCH]  # (rows, cols, num_patches, 3)
  levels = terrain.terrain_levels  # (num_envs,)
  types = terrain.terrain_types  # (num_envs,)
  return arr[levels, types][:, :3, :]


def obstacle_geometry(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged 5-D obstacle observation, normalised.

  Returns ``(num_envs, 5)``: height, width, depth, relative_x, relative_y.

  ``relative_x`` is signed: positive while the obstacle is ahead, negative once
  the robot has passed it. That sign flip is the cue the policy needs to stop
  preparing to climb and resume normal walking.
  """
  n = env.num_envs
  pts = _obstacle_points(env)
  if pts is None:
    # No obstacle terrain (e.g. flat baseline): report a null obstacle rather
    # than failing, so the same policy architecture runs on both.
    return torch.zeros((n, 5), device=env.device)

  p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]

  depth = (p1[:, 0] - p0[:, 0]).clamp(min=0.0)
  height = (p1[:, 2] - p0[:, 2]).clamp(min=0.0)
  width = (2.0 * (p2[:, 1] - p0[:, 1])).abs()

  robot = env.scene["robot"]
  root = robot.data.root_link_pos_w
  rel_x = p0[:, 0] - root[:, 0]
  rel_y = p0[:, 1] - root[:, 1]

  return torch.stack(
    (
      height / _HEIGHT_SCALE,
      width / _WIDTH_SCALE,
      depth / _DEPTH_SCALE,
      (rel_x / _DIST_SCALE).clamp(-1.0, 1.0),
      (rel_y / _DIST_SCALE).clamp(-1.0, 1.0),
    ),
    dim=-1,
  )


def _cleared(env: ManagerBasedRlEnv, margin: float) -> torch.Tensor:
  """True where the root is past the obstacle's far face by ``margin``."""
  pts = _obstacle_points(env)
  if pts is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  far_x = pts[:, 1, 0]
  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  return root_x > (far_x + margin)


def traversal_success(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  min_base_height: float = 0.5,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Traversal success, measured independently of the reward.

  This is what the gates read. It is deliberately *not* derived from reward
  terms: a reward can be maximised by behaviour that never clears the obstacle,
  and the whole point of the metric is to be able to detect that.

  Success requires the root past the far face AND the robot still upright. It
  says nothing about *how* the obstacle was cleared -- penetration or
  contact tunnelling would still register here, which is exactly why every
  gate also demands human video review.
  """
  del asset_cfg  # Robot is fixed for this task; kept for cfg symmetry.
  upright = env.scene["robot"].data.root_link_pos_w[:, 2] > min_base_height
  return (_cleared(env, margin) & upright).float()


def _episode_start(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mask of envs that have just been reset.

  `episode_length_buf` is 0 or 1 on the first step depending on whether the
  manager increments it before or after rewards are computed, so both are
  treated as "fresh". Falls back to "nothing is fresh" if the attribute is
  missing, which degrades to the old always-latched behaviour rather than
  crashing mid-run.
  """
  buf = getattr(env, "episode_length_buf", None)
  if buf is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return buf <= 1


def _env_buffer(env: ManagerBasedRlEnv, name: str,
                init: torch.Tensor) -> tuple[torch.Tensor, bool]:
  """Lazily allocate per-env reward state on the env object.

  Returns ``(buffer, was_created)``. The initial value is supplied by the
  caller rather than being zeros, which matters more than it looks:
  `RewardManager.compute` **skips any term whose weight is 0.0**, so a term
  faded in by a curriculum is not called at all until its weight leaves zero.
  A gap buffer initialised to zeros would then compute its first delta against
  a gap of 0 while the true gap is ~2.3 m, firing a single reward spike of
  about -115 at the exact moment the ramp switched on.
  """
  buf = getattr(env, name, None)
  if buf is None or buf.shape[0] != env.num_envs:
    buf = init.clone()
    setattr(env, name, buf)
    return buf, True
  return buf, False


def obstacle_progress(env: ManagerBasedRlEnv, std: float = 1.0) -> torch.Tensor:
  """Reward *closing* the gap to the obstacle, per unit time.

  Attempt 1 used ``exp(-gap/std)``, which is a function of *position*: once the
  robot is past the far face the gap is 0 and the term pays its maximum on
  every remaining timestep, forever. Combined with `traversal_bonus` that made
  "cross once, then stand still past the box" the optimal policy at any weight,
  and the measured result was a gait that dived over the obstacle and collapsed
  (Gate G4 attempt 1: final episode length 14.7 steps).

  This is the telescoping form instead: the per-step change in the gap, in m/s.
  Summed over an episode it collapses to ``(gap_start - gap_end) / dt``, so
  crossing pays exactly once no matter how it is done, and standing still pays
  nothing.

  Deliberately **not** clamped to positive values. An asymmetric version that
  rewarded approach but never penalised retreat would be farmable by oscillating
  back and forth across the same stretch of ground, which is a worse exploit
  than the one being fixed.
  """
  del std  # No longer a kernel width; kept so existing cfgs stay loadable.
  pts = _obstacle_points(env)
  if pts is None:
    return torch.zeros(env.num_envs, device=env.device)

  root_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  gap = (pts[:, 1, 0] - root_x).clamp(min=0.0)

  prev, created = _env_buffer(env, "_obstacle_prev_gap", gap)
  if created:
    return torch.zeros_like(gap)  # first call: no previous gap to compare to

  fresh = _episode_start(env)
  prev[fresh] = gap[fresh]  # no phantom progress across a reset

  delta = prev - gap
  prev.copy_(gap)

  dt = float(getattr(env, "step_dt", 0.02)) or 0.02
  # A reset teleports the robot backwards; without the guard above that shows
  # up as a large negative delta. Clamp the magnitude as a second line of
  # defence against any transition that still slips through -- at 0.02 s a
  # legitimate step cannot move the root more than a few cm.
  return (delta / dt).clamp(-5.0, 5.0)


def traversal_bonus(
  env: ManagerBasedRlEnv, margin: float = 0.3, min_base_height: float = 0.5
) -> torch.Tensor:
  """One-shot bonus, paid on the FIRST step the obstacle is cleared.

  Attempt 1 returned `traversal_success` directly, which is 1.0 on *every*
  timestep the root is past the far face and upright. At weight 5.0 that paid
  ~5 per step indefinitely against ~1 for tracking the commanded velocity, and
  by the end of the run `traversal_bonus` was 19x `track_linear_velocity` --
  the largest positive term in the whole function.

  Latching it to the first crossing keeps the incentive to get over the box
  while removing any reason to stay there. The latch is cleared on reset.
  """
  success = traversal_success(
    env, margin=margin, min_base_height=min_base_height
  ).bool()

  paid, _ = _env_buffer(env, "_traversal_paid",
                        torch.zeros_like(success, dtype=torch.bool))
  paid[_episode_start(env)] = False

  first_time = success & ~paid
  paid |= success
  return first_time.float()
