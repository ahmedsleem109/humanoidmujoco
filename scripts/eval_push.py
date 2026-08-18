"""Push-recovery evaluation and disturbance sweep.

Reuses `eval_gate.py`'s env/policy construction, ffmpeg resolution and provenance
capture rather than forking them -- those functions carry three fixed bugs that
must not be reintroduced (infinite play episodes, unbounded max_steps, and log
metrics diluted by sampling `extras["log"]` outside the episode-end block).

What this adds on top:
  * a fixed, controlled scenario -- constant forward command, no heading
    randomisation, no standing envs, so every episode is the same experiment;
  * `deterministic_push` at an exact magnitude / direction / instant;
  * `RecoveryTracker`, latched per environment;
  * a sweep over (force x direction) written to JSON + CSV.

The env is built ONCE and the push parameters are mutated between cells. Env
construction dominates runtime, and rebuilding per cell would turn a 2-minute
sweep into an hour.

Usage
-----
  ~/pyrun.sh scripts/eval_push.py \
      --checkpoint checkpoints/baseline_flat/model_6600.pt \
      --forces 0,10,20,30,40,50,60,80,100,120,150 \
      --directions 0,90,180,270 \
      --num-envs 128 --label r2_baseline
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

import eval_gate  # noqa: E402  (sibling script; sys.path set above)
from src.tasks.velocity.mdp.disturbance import deterministic_push  # noqa: E402
from src.tasks.velocity.mdp.recovery_metrics import RecoveryTracker  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@dataclass
class PushEvalConfig:
  checkpoint: str
  """Path to the policy checkpoint (repo-relative or absolute)."""
  task: str = "Unitree-G1-Flat"
  """Registered task id to evaluate in."""
  label: str = "push"
  """Short tag used in the output directory name."""

  forces: str = "0,10,20,30,40,50,60,80,100,120,150"
  """Comma-separated push magnitudes in Newtons."""
  directions: str = "0,90,180,270"
  """Comma-separated azimuths in degrees, in the robot's yaw frame.
  0=from behind, 90=from its right, 180=from the front, 270=from its left."""

  num_envs: int = 128
  """Episodes per (force, direction) cell -- the whole batch runs one cell."""
  repeats: int = 1
  """Batches per cell. Total episodes per cell = num_envs * repeats."""

  push_time_s: float = 3.0
  """Seconds after reset at which the push fires. Long enough to reach a steady gait."""
  push_duration_s: float = 0.1
  """Impulse width. 0.1 s matches the published G1 benchmark protocol."""
  episode_length_s: float = 7.0
  """push_time + recovery window + margin. Short episodes keep the sweep fast."""

  command_x: float = 1.0
  """Fixed forward velocity command (m/s) held for the whole episode."""

  window_s: float = 3.0
  settle_s: float = 0.5
  fall_tilt_deg: float = 45.0
  stable_tilt_deg: float = 15.0

  seed: int = 0
  device: str = "cuda:0"
  out_root: str = "results/push"


def _fix_scenario(env_cfg, cfg: PushEvalConfig) -> None:
  """Pin the command so the only thing that varies is the push.

  The play config samples a heading over the full circle and stands still in a
  fraction of envs. Both are correct for a general locomotion evaluation and
  fatal for a controlled disturbance experiment: a robot commanded to stand or
  to walk sideways is not the same experiment as one walking forward at 1 m/s.
  """
  twist = env_cfg.commands["twist"]
  twist.ranges.lin_vel_x = (cfg.command_x, cfg.command_x)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  # heading must be cleared, not zeroed: the command term rejects a heading range
  # when heading_command is False.
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.heading_command = False
  # Never resample mid-episode: a command change would be a second disturbance.
  twist.resampling_time_range = (1e6, 1e6)

  # Reset facing +x with no yaw noise, so "push from the left" is unambiguous.
  reset = env_cfg.events.get("reset_base")
  if reset is not None:
    reset.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                  "yaw": (0.0, 0.0)}

  # mjlab's own random push would contaminate the measurement.
  env_cfg.events.pop("push_robot", None)

  env_cfg.events["eval_push"] = EventTermCfg(
    func=deterministic_push,
    mode="step",
    params={
      "force_n": 0.0,
      "direction_deg": 0.0,
      "trigger_time_s": cfg.push_time_s,
      "duration_s": cfg.push_duration_s,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )


def _push_params(env):
  """The live params dict of the eval_push term, mutated between sweep cells."""
  term_cfgs = env.event_manager._mode_term_cfgs["step"]  # noqa: SLF001
  names = env.event_manager._mode_term_names["step"]  # noqa: SLF001
  return term_cfgs[names.index("eval_push")].params


def run_cell(env, wrapped, policy, cfg: PushEvalConfig, force_n: float,
             direction_deg: float) -> tuple[dict, list[dict]]:
  """One (force, direction) cell: reset the batch, roll out, score."""
  params = _push_params(env)
  params["force_n"] = float(force_n)
  params["direction_deg"] = float(direction_deg)

  tracker = RecoveryTracker(
    env,
    window_s=cfg.window_s,
    settle_s=cfg.settle_s,
    fall_tilt_deg=cfg.fall_tilt_deg,
    stable_tilt_deg=cfg.stable_tilt_deg,
  )

  steps_per_ep = int(math.ceil(cfg.episode_length_s / env.step_dt))
  for _ in range(cfg.repeats):
    obs, _ = wrapped.reset()
    for _ in range(steps_per_ep):
      with torch.inference_mode():
        actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      tracker.update(dones)
    # Score any episode still open when the batch ends.
    tracker.update(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

  summary = tracker.summary()
  summary["push_n"] = float(force_n)
  summary["direction_deg"] = float(direction_deg)
  for r in tracker.records:
    r["direction_deg"] = float(direction_deg)
  return summary, tracker.records


def main() -> None:
  cfg = tyro.cli(PushEvalConfig)
  forces = [float(x) for x in cfg.forces.split(",") if x.strip()]
  directions = [float(x) for x in cfg.directions.split(",") if x.strip()]

  stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  out_dir = REPO / cfg.out_root / f"{cfg.label}_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"[eval_push] output -> {out_dir}", flush=True)

  # Build the env directly rather than via eval_gate.build_env_and_policy: the
  # scenario edits must be applied to the cfg BEFORE the managers are constructed,
  # so the event term exists from the first step. The episode-length override that
  # build_env_and_policy exists to apply is replicated here explicitly -- play
  # configs set episode_length_s to 1e9 and would never time out.
  env_cfg = eval_gate.load_env_cfg(cfg.task, play=True)
  agent_cfg = eval_gate.load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = cfg.episode_length_s
  _fix_scenario(env_cfg, cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)

  wrapped, policy = eval_gate.load_policy(
    env, agent_cfg, cfg.task, cfg.checkpoint, cfg.device
  )
  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  wrapped.seed(cfg.seed)

  cells: list[dict] = []
  rows: list[dict] = []
  total = len(forces) * len(directions)
  i = 0
  for f in forces:
    for d in directions:
      i += 1
      s, recs = run_cell(env, wrapped, policy, cfg, f, d)
      cells.append(s)
      rows.extend(recs)
      print(
        f"[{i:3d}/{total}] {f:6.1f} N  dir {d:5.1f} deg  "
        f"recovery={s.get('recovery_rate', float('nan')):.3f}  "
        f"fall={s.get('fall_rate', float('nan')):.3f}  "
        f"peak_dv={s.get('mean_peak_dv', float('nan')):.3f} m/s  "
        f"tilt={s.get('mean_max_tilt_deg', float('nan')):.1f}deg  "
        f"n={s.get('episodes', 0)}",
        flush=True,
      )

  # Aggregate per force across directions -- the section 19 curve.
  curve = []
  for f in forces:
    sel = [c for c in cells if c["push_n"] == f and c.get("episodes")]
    if not sel:
      continue
    n = sum(c["episodes"] for c in sel)
    curve.append(
      {
        "push_n": f,
        "episodes": n,
        "recovery_rate": sum(c["recovery_rate"] * c["episodes"] for c in sel) / n,
        "fall_rate": sum(c["fall_rate"] * c["episodes"] for c in sel) / n,
      }
    )

  f_star = None
  for a, b in zip(curve, curve[1:]):
    if a["recovery_rate"] >= 0.5 > b["recovery_rate"]:
      span = a["recovery_rate"] - b["recovery_rate"]
      t = (a["recovery_rate"] - 0.5) / span if span > 0 else 0.0
      f_star = a["push_n"] + t * (b["push_n"] - a["push_n"])
      break

  result = {
    "label": cfg.label,
    "task": cfg.task,
    "checkpoint": cfg.checkpoint,
    "utc": stamp,
    "config": cfg.__dict__,
    "curve": curve,
    "cells": cells,
    "f_star_n": f_star,
  }
  (out_dir / "results.json").write_text(json.dumps(result, indent=2))

  if rows:
    with (out_dir / "episodes.csv").open("w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
      w.writeheader()
      w.writerows(rows)

  print("\n" + "=" * 62)
  print(f"{'force (N)':>10} {'recovery':>10} {'fall':>8} {'n':>6}")
  print("=" * 62)
  for c in curve:
    print(f"{c['push_n']:10.1f} {c['recovery_rate']:10.3f} "
          f"{c['fall_rate']:8.3f} {c['episodes']:6d}")
  print("=" * 62)
  print(f"F* (50% recovery) = {f_star if f_star is not None else 'NOT REACHED'}")
  print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
  main()
