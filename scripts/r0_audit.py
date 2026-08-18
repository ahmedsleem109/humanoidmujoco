"""R0 — environment revalidation for the robustness project.

Checks, in order:
  1. The four load-bearing version pins (HANDOFF.md section 2).
  2. `apply_body_impulse` exists and is exported from mjlab.envs.mdp.
  3. The `mjlab.envs.mdp.dr` surface covers the brief's section 14 checklist.
  4. The flat G1 velocity env builds in TRAINING mode, and its reward / event
     dicts are dumped so the section 12 energy + joint-motion terms can be audited.

Run via:  ~/pyrun.sh scripts/r0_audit.py
Read-only. Builds no simulation, allocates no GPU memory.
"""

from __future__ import annotations

import importlib.metadata as md

PINS = {
  "torch": "2.9.1",
  "mjlab": "1.2.0",
  "mujoco": "3.5.0",
  "warp-lang": "1.12.0",
}

# Brief section 14 -> the dr function that satisfies it.
DR_REQUIRED = {
  "link masses": ("body", "body_mass"),
  "com offset": ("body", "body_com_offset"),
  "full inertia": ("body", "pseudo_inertia"),
  "joint damping": ("joint", "joint_damping"),
  "joint friction": ("joint", "joint_friction"),
  "joint armature": ("joint", "joint_armature"),
  "joint stiffness": ("joint", "joint_stiffness"),
  "encoder bias": ("joint", "encoder_bias"),
  "ground friction": ("geom", "geom_friction"),
  "actuator gains": ("actuator", "pd_gains"),
  "effort limits": ("actuator", "effort_limits"),
  "actuator delay": ("actuator", "sync_actuator_delays"),
}

# Brief section 12 asks for energy and joint-motion penalties. Match loosely on
# name, since the exact term names are what we are trying to discover.
ENERGY_HINTS = ("action_rate", "energy", "torque", "power", "joint_acc", "joint_vel",
                "action_acc", "dof_acc", "dof_vel", "limit")

fails: list[str] = []
warns: list[str] = []


def head(t: str) -> None:
  print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


head("1. VERSION PINS")
for pkg, want in PINS.items():
  try:
    got = md.version(pkg)
  except md.PackageNotFoundError:
    print(f"  MISSING  {pkg}")
    fails.append(f"{pkg} not installed")
    continue
  ok = got.split("+")[0] == want
  print(f"  {'OK  ' if ok else 'DRIFT'}  {pkg:12s} {got}  (want {want})")
  if not ok:
    fails.append(f"{pkg} is {got}, pin is {want}")

head("2. DISTURBANCE GENERATOR")
try:
  from mjlab.envs import mdp as envs_mdp

  impulse = getattr(envs_mdp, "apply_body_impulse", None)
  if impulse is None:
    fails.append("apply_body_impulse not exported from mjlab.envs.mdp")
    print("  MISSING  apply_body_impulse")
  else:
    print(f"  OK       apply_body_impulse -> {impulse}")
    viz = getattr(impulse, "VizCfg", None)
    print(f"  {'OK      ' if viz else 'MISSING '} apply_body_impulse.VizCfg (force arrows)")
    if viz is None:
      warns.append("VizCfg missing - the video force indicator needs another route")
  for name in ("apply_external_force_torque", "push_by_setting_velocity"):
    print(f"  {'OK      ' if hasattr(envs_mdp, name) else 'MISSING '} {name}")
except Exception as exc:  # noqa: BLE001
  fails.append(f"cannot import mjlab.envs.mdp: {exc}")
  print(f"  IMPORT FAILED: {exc}")

head("3. DOMAIN RANDOMIZATION SURFACE (brief section 14)")
try:
  from mjlab.envs.mdp import dr

  for label, (_mod, fn) in DR_REQUIRED.items():
    present = hasattr(dr, fn)
    print(f"  {'OK      ' if present else 'MISSING '} {label:18s} dr.{fn}")
    if not present:
      fails.append(f"dr.{fn} missing ({label})")
except Exception as exc:  # noqa: BLE001
  fails.append(f"cannot import mjlab.envs.mdp.dr: {exc}")
  print(f"  IMPORT FAILED: {exc}")

head("4. FLAT G1 ENV, TRAINING MODE (play=False)")
try:
  from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  cfg = unitree_g1_flat_env_cfg(play=False)
  print(f"  built OK   episode_length_s={cfg.episode_length_s}")

  print("\n  --- EVENTS (the intervention surface) ---")
  for name, term in cfg.events.items():
    fn = getattr(term.func, "__name__", type(term.func).__name__)
    extra = ""
    if getattr(term, "mode", None) == "interval":
      extra = f"  interval={getattr(term, 'interval_range_s', None)}"
    print(f"    {name:22s} mode={getattr(term, 'mode', '?'):8s} {fn}{extra}")

  print("\n  --- REWARDS (brief section 12 audit) ---")
  found_energy: list[str] = []
  for name, term in cfg.rewards.items():
    w = getattr(term, "weight", None)
    fn = getattr(term.func, "__name__", type(term.func).__name__)
    print(f"    {name:26s} weight={w!s:>8s}  {fn}")
    if any(h in name.lower() or h in fn.lower() for h in ENERGY_HINTS):
      found_energy.append(name)

  print("\n  section 12 'excessive energy / joint motion' terms found:")
  if found_energy:
    for n in found_energy:
      print(f"    OK  {n}")
  else:
    print("    NONE - section 12 wants energy + joint-motion penalties")
    warns.append("no energy/joint-motion penalty found; section 12 may need one added")

  print("\n  --- OBSERVATIONS (brief section 11 compliance) ---")
  for group, obs in cfg.observations.items():
    terms = list(obs.terms.keys())
    print(f"    {group:8s} corruption={getattr(obs, 'enable_corruption', '?')!s:5s} "
          f"n={len(terms)}")
    print(f"             {terms}")

  print("\n  --- CURRICULUM ---")
  print(f"    {list(cfg.curriculum.keys()) if cfg.curriculum else '(none)'}")

  print("\n  --- COMMANDS ---")
  for name, c in cfg.commands.items():
    print(f"    {name}: ranges={getattr(c, 'ranges', None)}")
except Exception as exc:  # noqa: BLE001
  import traceback

  fails.append(f"flat env cfg failed to build: {exc}")
  traceback.print_exc()

head("R0 SUMMARY")
for w in warns:
  print(f"  WARN  {w}")
for f in fails:
  print(f"  FAIL  {f}")
print(f"\n  {len(fails)} failures, {len(warns)} warnings")
print("  R0 VERDICT:", "PASS" if not fails else "FAIL")
raise SystemExit(1 if fails else 0)
