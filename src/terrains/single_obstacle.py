"""Single-box obstacle terrain for the G1 traversal project.

The G1 spawns on a flat platform and walks in +x toward one box that spans the
full width of the tile, so the obstacle cannot be walked around -- traversal is
the only way forward.

Obstacle height interpolates on ``difficulty``, which is what lets mjlab's
existing ``terrain_levels_vel`` curriculum drive the 10 cm -> 40 cm progression
without a bespoke curriculum: rows of the terrain grid are difficulty levels,
and envs are promoted/demoted between them by how far they walked.

Verified against the mjlab 1.2.0 terrain API (see
``doc/verification/g0_3_terrain_api.md``). Modelled on
``mjlab.terrains.primitive_terrains.BoxPyramidStairsTerrainCfg``.

Note: ``add_geom(size=...)`` takes HALF-extents. A 10 cm box resting on the
ground is ``size=(hx, hy, 0.05)`` at ``pos=(x, y, 0.05)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane

# Name of the flat-patch channel carrying obstacle key points to runtime.
# Must match src/tasks/velocity/mdp/obstacle_mdp.py.
OBSTACLE_PATCH = "obstacle"

# The generator pre-allocates patch storage from flat_patch_sampling, so the
# channel must be declared even though we supply the points explicitly rather
# than sampling them off a heightfield.
OBSTACLE_PATCH_SAMPLING = {
  OBSTACLE_PATCH: FlatPatchSamplingCfg(num_patches=3, patch_radius=0.05)
}

_FLOOR_RGBA = (0.45, 0.45, 0.48, 1.0)
_OBSTACLE_EASY = np.array([0.25, 0.80, 0.45, 1.0])  # green, low
_OBSTACLE_HARD = np.array([0.90, 0.30, 0.30, 1.0])  # red, tall


def _lerp(lo: float, hi: float, t: float) -> float:
  return lo + t * (hi - lo)


@dataclass(kw_only=True)
class BoxSingleObstacleTerrainCfg(SubTerrainCfg):
  """One full-width box obstacle on a flat tile.

  Phase 2 (fixed / curriculum) leaves the ``*_range`` pairs collapsed or driven
  purely by ``difficulty``. Phase 3 (randomised) widens them so ``rng`` samples
  width, depth, and placement per tile while ``difficulty`` still drives height.
  """

  obstacle_height_range: tuple[float, float] = (0.10, 0.40)
  """Min/max obstacle height in metres. Interpolated by difficulty."""

  obstacle_depth_range: tuple[float, float] = (0.30, 0.30)
  """Min/max obstacle depth (extent along x), in metres. Sampled from rng."""

  obstacle_width_range: tuple[float, float] | None = None
  """Min/max obstacle width (extent along y), in metres. ``None`` (default)
  spans the full tile width so the robot cannot walk around it. Phase 3 may set
  a finite range, but note a narrower box reintroduces the walk-around
  shortcut -- keep it wide enough that going around is not the easy option."""

  spawn_distance_range: tuple[float, float] = (2.0, 2.0)
  """Min/max distance from the spawn point to the near face of the obstacle."""

  lateral_offset_range: tuple[float, float] = (0.0, 0.0)
  """Min/max lateral (y) offset of the obstacle centre from the tile centreline."""

  platform_length: float = 1.5
  """Flat run-up between the tile edge and the spawn point, in metres."""

  randomize: bool = False
  """If False, every sampled range collapses to its midpoint for reproducible
  fixed-obstacle training (Phase 2). If True, ``rng`` samples within the ranges
  (Phase 3). Height always follows ``difficulty``, never ``rng``."""

  def _sample(self, rng: np.random.Generator, rng_pair: tuple[float, float]) -> float:
    lo, hi = rng_pair
    if not self.randomize or hi <= lo:
      return 0.5 * (lo + hi)
    return float(rng.uniform(lo, hi))

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    size_x, size_y = self.size

    # Height is the curriculum axis: difficulty, never rng.
    height = _lerp(*self.obstacle_height_range, difficulty)
    depth = self._sample(rng, self.obstacle_depth_range)
    distance = self._sample(rng, self.spawn_distance_range)
    lateral = self._sample(rng, self.lateral_offset_range)

    spawn_x = self.platform_length
    obstacle_near_face = spawn_x + distance
    obstacle_cx = obstacle_near_face + 0.5 * depth
    obstacle_cy = 0.5 * size_y + lateral

    # Keep the box (and a landing run-out) inside the tile.
    max_cx = size_x - 0.5 * depth - 0.5
    if obstacle_cx > max_cx:
      obstacle_cx = max_cx
    if obstacle_cx <= spawn_x + 0.5 * depth:
      raise ValueError(
        f"Obstacle would overlap the spawn point: tile size_x={size_x} is too "
        f"small for platform_length={self.platform_length} + distance={distance} "
        f"+ depth={depth}. Increase size or reduce the ranges."
      )

    if self.obstacle_width_range is None:
      half_width = 0.5 * size_y
    else:
      half_width = 0.5 * self._sample(rng, self.obstacle_width_range)

    geometries: list[TerrainGeometry] = []

    floor = make_plane(body, self.size, 0.0, center_zero=False)
    geometries.append(TerrainGeometry(geom=floor[0], color=_FLOOR_RGBA))

    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        max(1e-6, 0.5 * depth),
        max(1e-6, half_width),
        max(1e-6, 0.5 * height),
      ),
      pos=(obstacle_cx, obstacle_cy, 0.5 * height),
    )
    rgba = tuple(
      _OBSTACLE_EASY + difficulty * (_OBSTACLE_HARD - _OBSTACLE_EASY)
    )
    geometries.append(TerrainGeometry(geom=box, color=rgba))

    # Publish obstacle geometry to runtime via flat_patches.
    #
    # The generator ADDS the tile's world_position to every patch value
    # (terrain_generator.py:335), so these must be genuine world positions.
    # Encoding the box by three of its points means geometry is recovered from
    # differences, which is invariant to that offset. See obstacle_mdp.py.
    near_x = obstacle_cx - 0.5 * depth
    far_x = obstacle_cx + 0.5 * depth
    obstacle_points = np.array(
      [
        [near_x, obstacle_cy, 0.0],  # p0: near face, bottom, centre
        [far_x, obstacle_cy, height],  # p1: far face, top, centre
        [near_x, obstacle_cy + half_width, 0.0],  # p2: near face, side edge
      ]
    )

    # Spawn on the flat run-up, facing +x toward the obstacle.
    origin = np.array([spawn_x, 0.5 * size_y, 0.0])
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches={OBSTACLE_PATCH: obstacle_points},
    )


def make_fixed_obstacle_cfg(height_m: float, size=(8.0, 4.0)):
  """A single fixed height at every difficulty level (Phase 2 / T2.4)."""
  return BoxSingleObstacleTerrainCfg(
    size=size,
    obstacle_height_range=(height_m, height_m),
    randomize=False,
    flat_patch_sampling=OBSTACLE_PATCH_SAMPLING,
  )


def make_curriculum_obstacle_cfg(
  min_height=0.10, max_height=0.40, size=(8.0, 4.0)
):
  """Height ladder driven by terrain difficulty rows (Phase 2 / T2.5)."""
  return BoxSingleObstacleTerrainCfg(
    size=size,
    obstacle_height_range=(min_height, max_height),
    randomize=False,
    flat_patch_sampling=OBSTACLE_PATCH_SAMPLING,
  )


def make_randomized_obstacle_cfg(size=(8.0, 4.0)):
  """Randomised geometry for the final policy (Phase 3 / T3.1)."""
  return BoxSingleObstacleTerrainCfg(
    size=size,
    obstacle_height_range=(0.10, 0.40),
    obstacle_depth_range=(0.20, 0.50),
    spawn_distance_range=(1.5, 3.0),
    lateral_offset_range=(-0.2, 0.2),
    randomize=True,
    flat_patch_sampling=OBSTACLE_PATCH_SAMPLING,
  )
