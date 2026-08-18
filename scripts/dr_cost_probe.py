"""Which domain-randomization term is making P3 9x slower?

P3 ran at 43 s/iteration against P1/P2's 4.7 s -- from iteration 0, so it is the
configuration, not contention. At that rate 6600 iterations is 79 hours instead
of 10, which makes the ablation's identical-budget requirement unaffordable.

Rather than guess, add the DR terms one at a time and time a fixed number of
environment steps for each. The suspect named in its own docstring is
`dr.pseudo_inertia`, which "triggers set_const recomputation" -- but the whole
point of measuring is that the obvious suspect is not always the guilty one.

Usage:  python scripts/dr_cost_probe.py [--num-envs 1024] [--steps 60]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg  # noqa: E402

TORSO = ("torso_link",)


def _impulse() -> EventTermCfg:
  return EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": (-200.0, 200.0),
      "torque_range": (0.0, 0.0),
      "duration_s": (0.05, 0.20),
      "cooldown_s": (2.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=TORSO),
    },
  )


def _pseudo_inertia() -> EventTermCfg:
  return EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.07, 0.07),
      "d_range": (-0.03, 0.03),
    },
  )


def _joint_damping() -> EventTermCfg:
  return EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )


def _joint_friction() -> EventTermCfg:
  return EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )


def _pd_gains() -> EventTermCfg:
  return EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )


# Cumulative: each variant is the previous one plus one term, so the jump
# between adjacent rows attributes the cost to the term that was added.
VARIANTS: list[tuple[str, list[str]]] = [
  ("baseline (P1: no pushes, no DR)", []),
  ("+ impulses (= P2)", ["impulse"]),
  ("+ friction range", ["impulse", "friction"]),
  ("+ joint_damping", ["impulse", "friction", "damping"]),
  ("+ joint_friction", ["impulse", "friction", "damping", "jfriction"]),
  ("+ actuator pd_gains", ["impulse", "friction", "damping", "jfriction", "gains"]),
  ("+ pseudo_inertia (= P3)",
   ["impulse", "friction", "damping", "jfriction", "gains", "inertia"]),
  ("+ body_mass instead of pseudo_inertia",
   ["impulse", "friction", "damping", "jfriction", "gains", "mass"]),
]


def build(terms: list[str], num_envs: int, device: str):
  cfg = unitree_g1_flat_env_cfg(play=False)
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)
  cfg.scene.num_envs = num_envs

  # P1/P2 pin the dynamics.
  if "friction" in terms:
    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.3)
  else:
    cfg.events["foot_friction"].params["ranges"] = (0.9, 0.9)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)

  if "impulse" in terms:
    cfg.events["body_impulse"] = _impulse()
  if "damping" in terms:
    cfg.events["joint_damping"] = _joint_damping()
  if "jfriction" in terms:
    cfg.events["joint_friction_dr"] = _joint_friction()
  if "gains" in terms:
    cfg.events["actuator_gains"] = _pd_gains()
  if "inertia" in terms:
    cfg.events["link_inertia"] = _pseudo_inertia()
  if "mass" in terms:
    cfg.events["link_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "scale",
        "ranges": (0.85, 1.15),
      },
    )

  return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)


def main(num_envs: int = 1024, steps: int = 60, device: str = "cuda:0",
         only: int = -1) -> None:
  """`only=N` runs a single variant by index.

  The sequential sweep leaks: warp/mujoco allocations are not freed by
  `env.close()` + `torch.cuda.empty_cache()`, so later rows report a partly
  cumulative footprint. Any VRAM number used to make a decision must come from a
  fresh process running one variant -- otherwise the measurement has the same
  defect it is trying to diagnose.
  """
  print(f"\n{num_envs} envs, {steps} timed steps each\n")
  variants = VARIANTS if only < 0 else [VARIANTS[only]]
  print(f"{'variant':38s} {'build (s)':>10s} {'ms/step':>10s} {'rel':>7s} "
        f"{'VRAM MB':>9s}")
  print("-" * 80)

  base_ms = None
  for label, terms in variants:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    env = build(terms, num_envs, device)
    build_s = time.time() - t0

    action = torch.zeros(
      (num_envs, env.action_manager.total_action_dim), device=device
    )
    env.reset()
    # Some reward terms write diagnostics into extras["log"], which the RL
    # wrapper normally creates. Stepping the raw env skips that, so seed it here.
    env.extras.setdefault("log", {})
    with torch.inference_mode():
      for _ in range(10):  # warm up kernels
        env.step(action)
      torch.cuda.synchronize()
      t0 = time.time()
      for _ in range(steps):
        env.step(action)
      torch.cuda.synchronize()
      elapsed = time.time() - t0

    ms = elapsed / steps * 1000.0
    if base_ms is None:
      base_ms = ms
    # torch's allocator does not see warp/mujoco allocations, so read the
    # driver's view of the whole device -- that is what actually thrashes.
    free_b, total_b = torch.cuda.mem_get_info()
    used_mb = (total_b - free_b) / 1024 / 1024
    print(f"{label:38s} {build_s:10.1f} {ms:10.2f} {ms / base_ms:6.2f}x "
          f"{used_mb:9.0f}")
    env.close()
    del env
    torch.cuda.empty_cache()

  print("\nA large jump between adjacent rows attributes the cost to the term")
  print("added on that row. Build time matters too: a startup term that forces a")
  print("model recompute shows up there rather than in ms/step.")


if __name__ == "__main__":
  tyro.cli(main)
