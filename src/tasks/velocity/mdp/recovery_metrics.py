"""Recovery metrics for the disturbance study.

Protocol shape follows the published G1 push-recovery benchmark (RecoverFormer;
also used by the variable-stiffness locomotion work): torso impulses scored by
Recovery Success Rate under a 45 degree tilt bound, with a bounded settling
window. Adopting their thresholds rather than inventing our own is deliberate --
it makes our numbers directly comparable to a published baseline on the same
robot.

Why this is a driven tracker and not an `Episode_Metrics` term
--------------------------------------------------------------
`Episode_Metrics/<name>` is a TIME-AVERAGE over the episode -- `mean(episode_sum
/ step_count)` -- not a terminal outcome. On the previous project the logged
`traversal_success` read 0.0117 while the true latched rate was 0.109, a factor
of ~9, and the ratio was not even stable across policies. A recovery rate read
from the training log would be wrong in exactly the same way and would look
entirely plausible.

So recovery is latched per environment, by this class, driven step by step from
the evaluation harness.

Definitions
-----------
fell        terminated, or torso tilt from vertical exceeds `fall_tilt_deg`
recovered   within `window_s` of the push: never fell, AND tilt returned below
            `stable_tilt_deg`, AND velocity-tracking error returned to within
            `vel_tol_factor` of its pre-push mean -- both held continuously for
            `settle_s`
recovery_time   push instant -> start of that sustained-stable interval
velocity_loss   integral of tracking error above the pre-push mean over the window

A policy that survives by freezing in a crouch fails the velocity criterion,
which is the point: the brief's section 12 rule is "reward task completion, not
just survival", and the metric has to agree with the reward about that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from .disturbance import PUSH_STATE_ATTR


class RecoveryTracker:
  """Latches per-episode recovery outcomes. One instance per evaluation run."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    window_s: float = 3.0,
    settle_s: float = 0.5,
    fall_tilt_deg: float = 45.0,
    stable_tilt_deg: float = 15.0,
    vel_tol_factor: float = 1.5,
    vel_tol_floor: float = 0.30,
    pre_push_s: float = 1.0,
  ) -> None:
    self.env = env
    self.command_name = command_name
    self.dt = env.step_dt
    self.window_steps = max(1, int(round(window_s / self.dt)))
    self.settle_steps = max(1, int(round(settle_s / self.dt)))
    self.pre_push_steps = max(1, int(round(pre_push_s / self.dt)))
    self.fall_cos = math.cos(math.radians(fall_tilt_deg))
    self.stable_cos = math.cos(math.radians(stable_tilt_deg))
    self.vel_tol_factor = vel_tol_factor
    self.vel_tol_floor = vel_tol_floor

    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    zb = lambda: torch.zeros(n, device=dev, dtype=torch.bool)  # noqa: E731

    # Rolling pre-push error estimate.
    self._pre_sum, self._pre_cnt = z(), z()
    self._pre_mean = z()

    # Post-push accumulators.
    self._fell, self._recovered, self._scored = zb(), zb(), zb()
    self._steps_since = torch.zeros(n, device=dev, dtype=torch.long)
    self._settle_run = torch.zeros(n, device=dev, dtype=torch.long)
    self._recovery_time, self._vel_loss, self._max_tilt = z(), z(), z()
    self._phase_at_push, self._contacts_at_push = z(), z()
    self._magnitude = z()

    # Push-delivery verification. A force that is configured but never reaches
    # the solver would leave every recovery rate at 1.0 and look like a great
    # result. `peak_dv` must scale with the commanded magnitude; if it does not,
    # the disturbance is not landing and no downstream number means anything.
    self._vel_at_push = torch.zeros(n, 3, device=dev)
    self._peak_dv = z()
    self._dv_steps = max(1, int(round(0.30 / self.dt)))

    self.records: list[dict] = []
    """One dict per completed episode. The Phase 3 CSV is built from this."""

  # ---------------------------------------------------------------- helpers --
  def _tilt_cos(self) -> torch.Tensor:
    """cos(tilt from vertical). +1 upright, 0 horizontal, negative inverted."""
    return -self.env.scene["robot"].data.projected_gravity_b[:, 2]

  def _vel_error(self) -> torch.Tensor:
    cmd = self.env.command_manager.get_command(self.command_name)
    vel = self.env.scene["robot"].data.root_link_lin_vel_b
    return torch.linalg.norm(cmd[:, :2] - vel[:, :2], dim=1)

  def _foot_contacts(self) -> torch.Tensor:
    """Number of feet in contact, or NaN when the sensor is unavailable."""
    try:
      data = self.env.scene.sensors["feet_ground_contact"].data
      found = getattr(data, "found", None)
      if found is None:
        raise AttributeError
      return found.reshape(self.env.num_envs, -1).sum(dim=1).float()
    except Exception:  # noqa: BLE001 - metric is diagnostic, never fatal
      return torch.full((self.env.num_envs,), float("nan"), device=self.env.device)

  # ------------------------------------------------------------------ update --
  def update(self, dones: torch.Tensor | None = None) -> None:
    """Advance one environment step. Call after `env.step`."""
    env = self.env
    st = getattr(env, PUSH_STATE_ATTR, None)
    if st is None:
      return

    tilt_cos = self._tilt_cos()
    vel_err = self._vel_error()
    pushed = st.fired

    # --- Before the push: accumulate the tracking-error baseline -------------
    pre = ~pushed
    if pre.any():
      self._pre_sum[pre] += vel_err[pre]
      self._pre_cnt[pre] += 1.0

    # --- At the push instant: freeze the baseline and snapshot the gait -------
    just = pushed & (st.trigger_step >= 0) & (self._steps_since == 0) & ~self._scored
    fresh_push = just & (self._pre_cnt > 0)
    if fresh_push.any():
      cnt = self._pre_cnt[fresh_push].clamp(min=1.0, max=float(self.pre_push_steps))
      self._pre_mean[fresh_push] = self._pre_sum[fresh_push] / cnt
      self._contacts_at_push[fresh_push] = self._foot_contacts()[fresh_push]
      self._magnitude[fresh_push] = st.magnitude[fresh_push]
      self._vel_at_push[fresh_push] = (
        self.env.scene["robot"].data.root_link_lin_vel_b[fresh_push]
      )

    # --- Inside the recovery window ------------------------------------------
    active = pushed & ~self._scored & (self._steps_since < self.window_steps)
    if active.any():
      self._steps_since[active] += 1
      self._max_tilt[active] = torch.maximum(
        self._max_tilt[active],
        torch.rad2deg(torch.arccos(tilt_cos[active].clamp(-1.0, 1.0))),
      )

      # Fall detection.
      newly_fell = active & (tilt_cos < self.fall_cos) & ~self._fell
      if dones is not None:
        newly_fell = newly_fell | (active & dones.bool() & ~self._recovered)
      self._fell[newly_fell] = True

      # Velocity loss: error above the pre-push baseline, integrated.
      excess = (vel_err - self._pre_mean).clamp(min=0.0) * self.dt
      self._vel_loss[active] += excess[active]

      # Push-delivery check: largest speed deviation shortly after the impulse.
      near = active & (self._steps_since <= self._dv_steps)
      if near.any():
        dv = torch.linalg.norm(
          self.env.scene["robot"].data.root_link_lin_vel_b - self._vel_at_push, dim=1
        )
        self._peak_dv[near] = torch.maximum(self._peak_dv[near], dv[near])

      # Sustained-stable run.
      tol = torch.maximum(
        self._pre_mean * self.vel_tol_factor,
        torch.full_like(self._pre_mean, self.vel_tol_floor),
      )
      stable = (tilt_cos > self.stable_cos) & (vel_err < tol) & ~self._fell
      self._settle_run = torch.where(
        stable & active, self._settle_run + 1, torch.zeros_like(self._settle_run)
      )

      done_settling = active & (self._settle_run >= self.settle_steps) & ~self._recovered
      if done_settling.any():
        self._recovered[done_settling] = True
        # Recovery began settle_steps ago.
        self._recovery_time[done_settling] = (
          self._steps_since[done_settling] - self.settle_steps
        ).float() * self.dt

    # --- Score episodes whose outcome is already determined ------------------
    #
    # A fall is committed IMMEDIATELY, not at reset. The push event clears its
    # own per-env state when `episode_length_buf` wraps, and that happens inside
    # `env.step` -- i.e. BEFORE this tracker next runs. Waiting for the reset to
    # score a fallen episode therefore loses it: `st.fired` is already False and
    # the episode silently never enters `records`.
    #
    # The symptom is brutal and quiet: at 600 N, 32 environments fell and only 2
    # were counted; at 1000 N, none were. Recovery rate is computed over scored
    # episodes, so dropping exactly the failures biases it toward 1.0 -- the
    # metric would have reported a policy as invincible precisely where it was
    # collapsing. Once fallen the outcome cannot change, so commit on the spot.
    window_done = pushed & ~self._scored & (self._steps_since >= self.window_steps)
    fell_now = pushed & ~self._scored & self._fell
    ended = torch.zeros_like(window_done)
    if dones is not None:
      ended = pushed & ~self._scored & dones.bool()
    to_score = window_done | fell_now | ended
    if to_score.any():
      self._commit(to_score)

    # --- Clear state for envs that reset -------------------------------------
    if dones is not None and dones.any():
      self._reset(dones.bool())

  # ------------------------------------------------------------------ commit --
  def _commit(self, mask: torch.Tensor) -> None:
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    for i in ids.tolist():
      recovered = bool(self._recovered[i]) and not bool(self._fell[i])
      self.records.append(
        {
          "push_n": float(self._magnitude[i]),
          "fell": bool(self._fell[i]),
          "recovered": recovered,
          "recovery_time_s": float(self._recovery_time[i]) if recovered else float("nan"),
          "max_tilt_deg": float(self._max_tilt[i]),
          "velocity_loss": float(self._vel_loss[i]),
          "pre_push_vel_err": float(self._pre_mean[i]),
          "feet_in_contact_at_push": float(self._contacts_at_push[i]),
          "peak_dv": float(self._peak_dv[i]),
        }
      )
    self._scored[mask] = True

  def _reset(self, mask: torch.Tensor) -> None:
    for buf in (
      self._pre_sum, self._pre_cnt, self._pre_mean, self._recovery_time,
      self._vel_loss, self._max_tilt, self._phase_at_push,
      self._contacts_at_push, self._magnitude, self._peak_dv,
    ):
      buf[mask] = 0.0
    self._vel_at_push[mask] = 0.0
    for buf in (self._fell, self._recovered, self._scored):
      buf[mask] = False
    self._steps_since[mask] = 0
    self._settle_run[mask] = 0

  # ----------------------------------------------------------------- summary --
  def summary(self) -> dict:
    """Aggregate over every completed episode."""
    n = len(self.records)
    if n == 0:
      return {"episodes": 0}

    rec = [r for r in self.records if r["recovered"]]
    times = [r["recovery_time_s"] for r in rec if not math.isnan(r["recovery_time_s"])]
    return {
      "episodes": n,
      "recovery_rate": len(rec) / n,
      "fall_rate": sum(r["fell"] for r in self.records) / n,
      "mean_recovery_time_s": (sum(times) / len(times)) if times else float("nan"),
      "mean_max_tilt_deg": sum(r["max_tilt_deg"] for r in self.records) / n,
      "mean_velocity_loss": sum(r["velocity_loss"] for r in self.records) / n,
      "mean_pre_push_vel_err": sum(r["pre_push_vel_err"] for r in self.records) / n,
      "mean_peak_dv": sum(r["peak_dv"] for r in self.records) / n,
    }
