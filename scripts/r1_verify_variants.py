"""Prove the three ablation arms differ ONLY in their event dict.

The ablation's entire claim is causal: P1 vs P2 vs P3 differ in the training
distribution and nothing else. That claim is worth exactly as much as this check.
A stray difference in reward weights, observation terms, command ranges or
episode length would silently turn "domain randomization helps" into "something
else helped", and the result would look completely normal.

Prints a diff and exits non-zero if anything outside `events` differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks.velocity.config.g1.robust_env_cfg import (  # noqa: E402
  unitree_g1_robust_env_cfg,
)

VARIANTS = ("nominal", "push", "robust")


def reward_sig(cfg) -> dict:
  return {k: float(getattr(t, "weight", float("nan"))) for k, t in cfg.rewards.items()}


def obs_sig(cfg) -> dict:
  return {
    g: (list(o.terms.keys()), bool(getattr(o, "enable_corruption", False)))
    for g, o in cfg.observations.items()
  }


def cmd_sig(cfg) -> dict:
  out = {}
  for name, c in cfg.commands.items():
    r = getattr(c, "ranges", None)
    out[name] = {
      "lin_vel_x": getattr(r, "lin_vel_x", None),
      "lin_vel_y": getattr(r, "lin_vel_y", None),
      "ang_vel_z": getattr(r, "ang_vel_z", None),
      "heading": getattr(r, "heading", None),
      "heading_command": getattr(c, "heading_command", None),
      "rel_standing_envs": getattr(c, "rel_standing_envs", None),
    }
  return out


def event_sig(cfg) -> dict:
  out = {}
  for name, t in cfg.events.items():
    fn = getattr(t.func, "__name__", type(t.func).__name__)
    keep = {}
    for k in ("force_range", "duration_s", "cooldown_s", "ranges",
              "position_range", "velocity_range", "operation"):
      if k in getattr(t, "params", {}):
        keep[k] = t.params[k]
    out[name] = {"mode": getattr(t, "mode", "?"), "func": fn, "params": keep}
  return out


cfgs = {v: unitree_g1_robust_env_cfg(v, play=False) for v in VARIANTS}

fails: list[str] = []

print("=" * 74)
print("INVARIANTS (must be identical across all three arms)")
print("=" * 74)
for label, fn in (
  ("rewards", reward_sig),
  ("observations", obs_sig),
  ("commands", cmd_sig),
):
  sigs = {v: fn(cfgs[v]) for v in VARIANTS}
  ref = sigs["nominal"]
  same = all(sigs[v] == ref for v in VARIANTS)
  print(f"  {'OK  ' if same else 'DIFF'}  {label}")
  if not same:
    fails.append(label)
    for v in VARIANTS:
      if sigs[v] != ref:
        for k in set(ref) | set(sigs[v]):
          if ref.get(k) != sigs[v].get(k):
            print(f"          {v}.{k}: {sigs[v].get(k)!r} != nominal {ref.get(k)!r}")

for label, get in (
  ("episode_length_s", lambda c: c.episode_length_s),
  ("decimation", lambda c: c.decimation),
  ("num reward terms", lambda c: len(c.rewards)),
  ("curriculum", lambda c: sorted(c.curriculum.keys()) if c.curriculum else []),
):
  vals = {v: get(cfgs[v]) for v in VARIANTS}
  same = len(set(map(str, vals.values()))) == 1
  print(f"  {'OK  ' if same else 'DIFF'}  {label}: {vals['nominal']!r}")
  if not same:
    fails.append(label)
    print(f"          {vals}")

print()
print("=" * 74)
print("THE INTERVENTION (events -- these SHOULD differ)")
print("=" * 74)
evs = {v: event_sig(cfgs[v]) for v in VARIANTS}
all_names = sorted(set().union(*[set(e) for e in evs.values()]))
print(f"  {'event':22s} {'P1 nominal':>14s} {'P2 push':>14s} {'P3 robust':>14s}")
for name in all_names:
  cells = []
  for v in VARIANTS:
    cells.append("yes" if name in evs[v] else "-")
  print(f"  {name:22s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}")

print("\n  friction ranges:")
for v in VARIANTS:
  ff = evs[v].get("foot_friction", {}).get("params", {}).get("ranges")
  print(f"    {v:8s} {ff}")

print("\n  impulse config:")
for v in VARIANTS:
  bi = evs[v].get("body_impulse")
  print(f"    {v:8s} {bi['params'] if bi else '(none)'}")

# --- Assertions the design depends on ---------------------------------------
print()
print("=" * 74)
print("DESIGN ASSERTIONS")
print("=" * 74)
checks = [
  ("P1 has no impulses", "body_impulse" not in evs["nominal"]),
  ("P2 has impulses", "body_impulse" in evs["push"]),
  ("P3 has impulses", "body_impulse" in evs["robust"]),
  ("P1 has no encoder_bias", "encoder_bias" not in evs["nominal"]),
  ("P2 has no encoder_bias", "encoder_bias" not in evs["push"]),
  ("P3 has encoder_bias", "encoder_bias" in evs["robust"]),
  ("P3 has mass DR", "link_mass" in evs["robust"]),
  ("P3 has actuator-gain DR", "actuator_gains" in evs["robust"]),
  ("P1 friction is pinned",
   evs["nominal"]["foot_friction"]["params"]["ranges"][0]
   == evs["nominal"]["foot_friction"]["params"]["ranges"][1]),
  ("P3 friction is randomized",
   evs["robust"]["foot_friction"]["params"]["ranges"][0]
   != evs["robust"]["foot_friction"]["params"]["ranges"][1]),
  ("push_robot removed everywhere",
   all("push_robot" not in evs[v] for v in VARIANTS)),
  ("command_vel curriculum removed everywhere",
   all("command_vel" not in (cfgs[v].curriculum or {}) for v in VARIANTS)),
  ("observation corruption ON for actor in all three",
   all(cfgs[v].observations["actor"].enable_corruption for v in VARIANTS)),
]
for label, ok in checks:
  print(f"  {'OK  ' if ok else 'FAIL'}  {label}")
  if not ok:
    fails.append(label)

print()
print(f"  {len(fails)} failures")
print("  R1 VARIANT VERDICT:", "PASS" if not fails else "FAIL")
raise SystemExit(1 if fails else 0)
