"""Deterministic external disturbances for the robustness study.

`mjlab.envs.mdp.apply_body_impulse` is the right tool for *training*: it samples
forces per component, per environment, with independent timers, which is exactly
the decorrelated disturbance distribution a robust policy should be trained under.

It is the wrong tool for *evaluation*, for two reasons:

1. **Magnitude is not controllable.** ``force_range`` is sampled independently
   per component, so a nominal "50 N push" is really ``U(-50, 50)^3``: the
   magnitude follows a corner-biased distribution over the cube with an expected
   value nowhere near 50, and the direction is not uniform on the sphere. A
   force-vs-recovery curve built on that is not a curve of anything.
2. **Timing and direction are not repeatable.** The headline comparison requires
   the *same* push, at the *same* gait-relative instant, on every policy. Only
   the policy may vary.

`deterministic_push` therefore applies an exact magnitude, in an exact direction,
at an exact time after reset, once per episode, identically across the batch.

Direction convention
--------------------
``direction_deg`` is the azimuth of the applied force **in the robot's yaw
frame**, so it is invariant to which way the robot happens to be facing:

    0 deg   -> +x, pushed from behind (shoved forward)
    90 deg  -> +y, pushed from its right (shoved to its left)
    180 deg -> -x, pushed from the front (shoved backward)
    270 deg -> -y, pushed from its left (shoved to its right)

Roll and pitch are ignored deliberately (``quat_apply_yaw``): a push from the
left is a push from the left regardless of how far the torso has already tipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_yaw

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


PUSH_STATE_ATTR = "_det_push_state"


class DeterministicPushState:
  """Per-env bookkeeping for the deterministic push.

  Lives on the env object under `PUSH_STATE_ATTR` so the recovery metrics in
  `recovery_metrics.py` can read it without being wired through the manager
  config. Follows the same "state on the env" pattern as `obstacle_mdp._env_buffer`.
  """

  def __init__(self, num_envs: int, device: str) -> None:
    z_f = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
    z_b = lambda: torch.zeros(num_envs, device=device, dtype=torch.bool)  # noqa: E731

    self.active = z_b()
    """True while the impulse is being applied."""
    self.fired = z_b()
    """True once this episode's push has been triggered (latched until reset)."""
    self.trigger_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    """Episode step index at which the push fired, -1 if it has not."""
    self.time_since_push = z_f()
    """Seconds since the push fired; 0 while it has not."""
    self.force_w = torch.zeros(num_envs, 3, device=device)
    """The applied force vector in world frame (N), zero when inactive."""
    self.magnitude = z_f()
    """Commanded magnitude in N, held for the whole episode after firing."""
    self._steps_remaining = torch.zeros(num_envs, device=device, dtype=torch.long)


def _state(env: ManagerBasedRlEnv) -> DeterministicPushState:
  st = getattr(env, PUSH_STATE_ATTR, None)
  if st is None or st.active.shape[0] != env.num_envs:
    st = DeterministicPushState(env.num_envs, env.device)
    setattr(env, PUSH_STATE_ATTR, st)
  return st


class deterministic_push:  # noqa: N801 - matches mjlab's event-term naming
  """Apply one exact impulse per episode, identically across the batch.

  Use with ``mode="step"``. Intended for evaluation only; training uses
  `mjlab.envs.mdp.apply_body_impulse`.

  A zero ``force_n`` is a valid and useful configuration: it produces the
  undisturbed control condition through exactly the same code path, so the
  "no push" arm of an experiment cannot differ from the "push" arm by anything
  other than the force itself.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._device = env.device
    self._step_dt = env.step_dt
    self._num_bodies = (
      len(self._body_ids) if isinstance(self._body_ids, list) else self._asset.num_bodies
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_n: float,
    direction_deg: float,
    trigger_time_s: float,
    duration_s: float,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del env_ids, asset_cfg  # Step events always cover all envs.
    st = _state(env)

    # Episode step index. `episode_length_buf` counts env steps since reset.
    ep_step = env.episode_length_buf
    trigger_step = max(1, int(round(trigger_time_s / self._step_dt)))
    duration_steps = max(1, int(round(duration_s / self._step_dt)))

    # --- Reset bookkeeping for freshly-restarted envs -------------------------
    # Cheaper and more robust than hooking the reset event: any env whose step
    # counter has gone backwards has been reset.
    fresh = ep_step <= 1
    if fresh.any():
      st.fired[fresh] = False
      st.active[fresh] = False
      st.trigger_step[fresh] = -1
      st.time_since_push[fresh] = 0.0
      st.force_w[fresh] = 0.0
      st.magnitude[fresh] = 0.0
      st._steps_remaining[fresh] = 0

    # --- Expire impulses that have run their course --------------------------
    if st.active.any():
      st._steps_remaining[st.active] -= 1
      expired = st.active & (st._steps_remaining <= 0)
      if expired.any():
        ids = expired.nonzero(as_tuple=False).squeeze(-1)
        zeros = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
        self._asset.write_external_wrench_to_sim(
          zeros, zeros, env_ids=ids, body_ids=self._body_ids
        )
        st.active[ids] = False
        st.force_w[ids] = 0.0

    # Advance the post-push clock for every env that has already been pushed.
    st.time_since_push[st.fired] += self._step_dt

    # --- Trigger --------------------------------------------------------------
    due = (~st.fired) & (ep_step >= trigger_step)
    if not due.any():
      return

    ids = due.nonzero(as_tuple=False).squeeze(-1)
    n = len(ids)

    # Exact magnitude, exact azimuth, in the robot's yaw frame.
    theta = math.radians(direction_deg)
    dir_b = torch.tensor(
      [math.cos(theta), math.sin(theta), 0.0], device=self._device, dtype=torch.float32
    ).expand(n, 3)
    quat = self._asset.data.root_link_quat_w[ids]
    force_vec = quat_apply_yaw(quat, dir_b) * float(force_n)

    forces = force_vec.unsqueeze(1).expand(n, self._num_bodies, 3).contiguous()
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=ids, body_ids=self._body_ids
    )

    st.fired[ids] = True
    st.active[ids] = force_n != 0.0
    st.trigger_step[ids] = ep_step[ids]
    st.time_since_push[ids] = 0.0
    st.force_w[ids] = force_vec
    st.magnitude[ids] = float(force_n)
    st._steps_remaining[ids] = duration_steps


def push_magnitude(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Commanded push magnitude (N) for this episode. Observation/metric helper."""
  return _state(env).magnitude


def push_fired(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 once this episode's push has been applied."""
  return _state(env).fired.float()


def time_since_push(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Seconds since the push fired; 0 before it does."""
  return _state(env).time_since_push
