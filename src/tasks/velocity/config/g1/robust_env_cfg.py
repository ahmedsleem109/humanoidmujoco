"""The three policies of the robustness ablation.

    P1  nominal   no pushes, no dynamics randomization
    P2  push      + disturbance randomization
    P3  robust    + disturbance randomization + domain randomization

All three share an identical reward function, observation space, command
distribution, seed, horizon and network. **The event dict is the only thing that
differs.** That is what makes the ablation causal: any gap between P1, P2 and P3
is attributable to the training distribution and nothing else.

Three deliberate departures from the mjlab default, each recorded here because
each would otherwise look like an accident:

1. **`push_robot` is removed from all three.** mjlab's default disturbance is
   `push_by_setting_velocity`, which teleports the root velocity. It cannot be
   expressed in Newtons, so a policy trained under it cannot be evaluated on a
   force axis without changing units mid-experiment. P2/P3 use
   `apply_body_impulse`, which writes a real wrench, so training and evaluation
   share a unit.

2. **Observation noise stays ON in all three.** It models the *sensor*, not the
   *dynamics*. Removing it from P1 would confound the ablation with an
   observability change, and the question being asked is about dynamics.

3. **The `command_vel` curriculum is popped.** It doubles the velocity command
   range at iteration 5000, which (a) makes the command distribution
   non-stationary in the last quarter of a 6600-iteration run, and (b) cost the
   previous project ~40% of its reward when it fired. Evaluation is at a fixed
   1.0 m/s, so all three policies train on the full range from step 0 instead.
   This removes a confound rather than adding one.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

Variant = Literal["nominal", "push", "robust"]

# Torso body, matching the published G1 push-recovery protocol (torso impulses).
TORSO = ("torso_link",)

# Per-component force range (N). `apply_body_impulse` samples each component
# independently, so the achievable magnitude reaches |F|*sqrt(3) ~ 346 N at 200.
# Sized against the measured failure boundary of the legacy push-trained policy
# (F* ~ 450 N at 0.1 s), so the training distribution straddles the interesting
# region without being uniformly unrecoverable.
TRAIN_FORCE_RANGE = (-200.0, 200.0)
TRAIN_DURATION_S = (0.05, 0.20)
TRAIN_COOLDOWN_S = (2.0, 5.0)

# Domain randomization ranges. Evaluation deliberately goes WIDER than these
# (see configs/protocol.yaml) -- that gap is the generalization experiment.
FRICTION_TRAIN = (0.5, 1.3)  # ~ +/-30% about 0.9
FRICTION_NOMINAL = (0.9, 0.9)  # P1: fixed, no randomization


def _strip_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """P1: pin the dynamics. One fixed robot, one fixed floor.

  `foot_friction` is pinned rather than deleted so the friction *value* is the
  same in all three variants; only its variance changes. Deleting the term would
  leave P1 on the model default and confound "no randomization" with "different
  mean friction".
  """
  cfg.events["foot_friction"].params["ranges"] = FRICTION_NOMINAL
  cfg.events.pop("encoder_bias", None)
  cfg.events.pop("base_com", None)


def _add_full_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """P3: randomize what the brief's section 14 asks for.

  Startup-mode terms resample per environment when the scene is built, so with
  4096 environments the policy sees 4096 different robots concurrently.
  """
  cfg.events["foot_friction"].params["ranges"] = FRICTION_TRAIN

  # Mass randomization via `dr.body_mass`, NOT `dr.pseudo_inertia`.
  #
  # `pseudo_inertia` is the physically superior term -- it perturbs mass, COM and
  # inertia jointly and stays positive-definite -- and it was the original
  # choice. It is unusable here: measured with `scripts/dr_cost_probe.py`, it
  # imposes a **fixed ~4 GB overhead** (its `set_const` recomputation), which is
  # nearly independent of environment count:
  #
  #     P3 config, 1024 envs -> 5649 MB
  #     P3 config, 2048 envs -> 5914 MB
  #     P3 config, 4096 envs -> saturates a 6 GB card -> 43 s/iteration
  #
  # 43 s/iteration against P1/P2's 4.7 s is 79 h for one run. The alternatives
  # were to drop P3 to a smaller env count -- which breaks the ablation's
  # identical-budget requirement and would handicap exactly the arm under test --
  # or to drop the term. Every other DR term is free: the full set without
  # pseudo_inertia costs 2502 MB at 4096 envs against P2's 2470 MB.
  #
  # What is lost: `body_mass` scales mass without scaling the inertia tensor, so
  # it does NOT model "this link is denser than modelled". What it DOES model
  # exactly is a **point mass added at the COM** -- an unmodelled payload, an
  # added battery, mounted equipment. That is a real and relevant source of mass
  # uncertainty for a humanoid, so the term is physically meaningful rather than
  # a fudge; it is simply narrower than intended. Recorded in ROBUSTNESS_LOG.md.
  cfg.events["link_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "scale",
      "ranges": (0.85, 1.15),
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["joint_friction_dr"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": (0.8, 1.2),
    },
  )
  # pd_gains takes separate kp/kd ranges, not a single `ranges` tuple.
  cfg.events["actuator_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )

  # Initial-condition randomization (brief section 14). Wider than the default
  # reset so the policy meets disturbances from more starting states.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.2, 0.2)


def _add_impulses(cfg: ManagerBasedRlEnvCfg) -> None:
  """P2/P3: real external forces, in Newtons, with independent per-env timers."""
  cfg.events["body_impulse"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": TRAIN_FORCE_RANGE,
      "torque_range": (0.0, 0.0),
      "duration_s": TRAIN_DURATION_S,
      "cooldown_s": TRAIN_COOLDOWN_S,
      "asset_cfg": SceneEntityCfg("robot", body_names=TORSO),
    },
  )


def unitree_g1_robust_env_cfg(
  variant: Variant, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build one arm of the ablation."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Shared across all three arms -- see the module docstring.
  cfg.events.pop("push_robot", None)
  cfg.curriculum.pop("command_vel", None)

  if variant == "nominal":
    _strip_domain_randomization(cfg)
  elif variant == "push":
    _strip_domain_randomization(cfg)
    if not play:
      _add_impulses(cfg)
  elif variant == "robust":
    _add_full_domain_randomization(cfg)
    if not play:
      _add_impulses(cfg)
  else:  # pragma: no cover - guarded by Literal
    raise ValueError(f"unknown variant {variant!r}")

  return cfg


def unitree_g1_robust_p1_nominal(play: bool = False) -> ManagerBasedRlEnvCfg:
  return unitree_g1_robust_env_cfg("nominal", play=play)


def unitree_g1_robust_p2_push(play: bool = False) -> ManagerBasedRlEnvCfg:
  return unitree_g1_robust_env_cfg("push", play=play)


def unitree_g1_robust_p3_robust(play: bool = False) -> ManagerBasedRlEnvCfg:
  return unitree_g1_robust_env_cfg("robust", play=play)
