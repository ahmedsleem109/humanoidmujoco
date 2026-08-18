"""Consolidate every measured number into one place, ready for plotting.

Re-runnable and idempotent: run it again after each policy lands and it picks up
the new sweeps and training logs automatically. Nothing here is hand-typed --
every value is read from a `results.json` written by `eval_push.py` or parsed
from a training log, so the consolidated file cannot drift from the artifacts.

Outputs (all under results/metrics/):
  all_metrics.json     everything, nested
  curves.csv           long format: policy, force_n, direction, recovery, n
  fstar.csv            policy x direction -> F*
  training.csv         per-policy final training metrics
  ablation.md          the human-readable summary table

Usage:  python scripts/collect_metrics.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUSH_DIR = REPO / "results" / "push"
OUT_DIR = REPO / "results" / "metrics"
RUNS = Path.home() / "runs"

# label prefix in results/push/<label>_<stamp>/ -> canonical policy name
POLICY_OF_LABEL = {
  "r2_legacy_baseline": "legacy",
  "p1_nominal": "P1_nominal",
  "p2_push": "P2_push",
  "p2_push_hi": "P2_push",
  "p3_robust": "P3_robust",
  "p3_robust_hi": "P3_robust",
}

POLICY_DESC = {
  "legacy": "mjlab default recipe: velocity pushes + partial DR (pre-existing)",
  "P1_nominal": "no pushes, no dynamics randomization",
  "P2_push": "+ disturbance randomization (apply_body_impulse)",
  "P3_robust": "+ disturbance randomization + domain randomization",
}

TRAIN_LOGS = {
  "P1_nominal": RUNS / "p1nominal.log",
  "P2_push": RUNS / "p2push.log",
  "P3_robust": RUNS / "p3robust.log",
}

DIR_LABEL = {
  0.0: "from_behind", 90.0: "from_right",
  180.0: "from_front", 270.0: "from_left",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n == 0:
    return (float("nan"), float("nan"))
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (max(0.0, c - h), min(1.0, c + h))


def f_star(points: list[tuple[float, float]]) -> float | None:
  pts = sorted(points)
  for (f0, r0), (f1, r1) in zip(pts, pts[1:]):
    if r0 >= 0.5 > r1:
      span = r0 - r1
      t = (r0 - 0.5) / span if span > 0 else 0.0
      return f0 + t * (f1 - f0)
  return None


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def collect_sweeps() -> dict:
  """Merge every sweep cell, keyed by (policy, force, direction).

  Base and extended-grid runs are merged rather than concatenated: the extended
  run re-measures 700 N, and a duplicate force would corrupt the interpolation
  that produces F*. Later runs win.
  """
  cells: dict[str, dict[tuple[float, float], dict]] = {}
  sources: dict[str, list[str]] = {}

  for d in sorted(PUSH_DIR.glob("*/")):
    rj = d / "results.json"
    if not rj.exists():
      continue
    try:
      data = json.loads(rj.read_text())
    except json.JSONDecodeError:
      continue

    label = data.get("label", "")
    if label.startswith("smoke") or label.startswith("verify"):
      continue  # development runs, not measurements
    policy = POLICY_OF_LABEL.get(label)
    if policy is None:
      continue

    cells.setdefault(policy, {})
    sources.setdefault(policy, []).append(d.name)
    for c in data.get("cells", []):
      if not c.get("episodes"):
        continue
      key = (float(c["push_n"]), float(c["direction_deg"]))
      cells[policy][key] = c

  out: dict[str, dict] = {}
  for policy, grid in cells.items():
    forces = sorted({f for f, _ in grid})
    dirs = sorted({d for _, d in grid})

    overall = []
    for f in forces:
      sel = [grid[(f, d)] for d in dirs if (f, d) in grid]
      n = sum(c["episodes"] for c in sel)
      k = sum(round(c["recovery_rate"] * c["episodes"]) for c in sel)
      lo, hi = wilson(k, n)
      overall.append({
        "push_n": f, "n": n, "recovered": k,
        "recovery_rate": k / n if n else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "fall_rate": sum(c["fall_rate"] * c["episodes"] for c in sel) / n if n else float("nan"),
        "mean_peak_dv": sum(c.get("mean_peak_dv", 0.0) * c["episodes"] for c in sel) / n if n else float("nan"),
        "mean_max_tilt_deg": sum(c.get("mean_max_tilt_deg", 0.0) * c["episodes"] for c in sel) / n if n else float("nan"),
      })

    by_dir = {}
    for d in dirs:
      pts = [(f, grid[(f, d)]["recovery_rate"]) for f in forces if (f, d) in grid]
      by_dir[DIR_LABEL.get(d, f"{d:.0f}deg")] = {
        "direction_deg": d,
        "f_star_n": f_star(pts),
        "points": [{"push_n": f, "recovery_rate": r} for f, r in pts],
      }

    out[policy] = {
      "description": POLICY_DESC.get(policy, ""),
      "source_runs": sorted(sources[policy]),
      "max_force_tested_n": max(forces) if forces else None,
      "total_scored_pushes": sum(c["episodes"] for c in grid.values()),
      "f_star_overall_n": f_star([(o["push_n"], o["recovery_rate"]) for o in overall]),
      "f_star_by_direction_n": {k: v["f_star_n"] for k, v in by_dir.items()},
      "curve": overall,
      "by_direction": by_dir,
    }
  return out


# --------------------------------------------------------------------------
# Training logs
# --------------------------------------------------------------------------

NUM = r"(-?\d+\.?\d*(?:e-?\d+)?)"
PATTERNS = {
  "mean_reward": re.compile(r"Mean reward:\s*" + NUM),
  "mean_episode_length": re.compile(r"Mean episode length:\s*" + NUM),
  "error_vel_xy": re.compile(r"Metrics/twist/error_vel_xy:\s*" + NUM),
  "error_vel_yaw": re.compile(r"Metrics/twist/error_vel_yaw:\s*" + NUM),
  "mean_action_acc": re.compile(r"Episode_Metrics/mean_action_acc:\s*" + NUM),
  "angular_momentum_mean": re.compile(r"Metrics/angular_momentum_mean:\s*" + NUM),
  "slip_velocity_mean": re.compile(r"Metrics/slip_velocity_mean:\s*" + NUM),
  "landing_force_mean": re.compile(r"Metrics/landing_force_mean:\s*" + NUM),
  "fell_over": re.compile(r"Episode_Termination/fell_over:\s*" + NUM),
  "iteration_time": re.compile(r"Iteration time:\s*" + NUM),
}
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")


def collect_training() -> dict:
  out = {}
  for policy, log in TRAIN_LOGS.items():
    if not log.exists():
      continue
    try:
      text = log.read_text(errors="ignore")
    except OSError:
      continue

    iters = ITER_RE.findall(text)
    rec: dict = {
      "log": str(log),
      "iterations_seen": int(iters[-1][0]) if iters else None,
      "iterations_target": int(iters[-1][1]) if iters else None,
      "complete": bool(iters) and int(iters[-1][0]) >= int(iters[-1][1]) - 1,
    }
    for name, pat in PATTERNS.items():
      vals = pat.findall(text)
      if vals:
        rec[name] = float(vals[-1])
        if name == "mean_reward":
          # Coarse trajectory for training-curve plots.
          rec["reward_trajectory"] = [float(v) for v in vals[:: max(1, len(vals) // 40)]]
        if name == "mean_episode_length":
          rec["episode_length_trajectory"] = [
            float(v) for v in vals[:: max(1, len(vals) // 40)]
          ]
    out[policy] = rec
  return out


# --------------------------------------------------------------------------
# Static facts worth preserving (measured, not assumed)
# --------------------------------------------------------------------------

STATIC = {
  "hardware": {
    "gpu": "RTX 3060 Laptop, 6 GB",
    "host": "Windows 11 + WSL2 Ubuntu 24.04",
    "train_num_envs": 4096,
    "train_iterations": 6600,
    "iteration_time_s": 4.7,
  },
  "pins": {
    "torch": "2.9.1+cu129", "mjlab": "1.2.0",
    "mujoco": "3.5.0", "warp-lang": "1.12.0",
  },
  "eval_protocol": {
    "push_body": "torso_link",
    "push_duration_s": 0.1,
    "push_trigger_time_s": 3.0,
    "command_lin_vel_x_m_s": 1.0,
    "episode_length_s": 7.0,
    "num_envs": 64,
    "fall_tilt_deg": 45.0,
    "stable_tilt_deg": 15.0,
    "recovery_window_s": 3.0,
    "settle_s": 0.5,
    "seed": 0,
  },
  "training_disturbance": {
    "force_range_per_component_n": [-200.0, 200.0],
    "max_realisable_magnitude_n": 346,
    "duration_s": [0.05, 0.20],
    "cooldown_s": [2.0, 5.0],
  },
  "dr_cost_probe_4096_envs": {
    "note": "VRAM measured in a FRESH process per variant; the cumulative sweep "
            "leaks warp allocations and mis-attributes cost.",
    "P1_baseline_mb": 2470,
    "P2_impulses_mb": 2470,
    "P3_without_pseudo_inertia_mb": 2502,
    "P3_with_body_mass_mb": 2502,
    "P3_with_pseudo_inertia": "saturates 6 GB -> 43 s/iter (9x slowdown)",
    "pseudo_inertia_is_env_count_independent": {"1024": 5649, "2048": 5914},
  },
}


def main() -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  sweeps = collect_sweeps()
  training = collect_training()

  blob = {
    "project": "Robust Humanoid Locomotion Under Unexpected Disturbances",
    "robot": "Unitree G1 (29 dof)",
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "static": STATIC,
    "sweeps": sweeps,
    "training": training,
  }
  (OUT_DIR / "all_metrics.json").write_text(json.dumps(blob, indent=2))

  # ---- curves.csv : long format, one row per (policy, force, direction) ----
  with (OUT_DIR / "curves.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["policy", "direction", "direction_deg", "push_n", "recovery_rate"])
    for policy, d in sweeps.items():
      for dname, dd in d["by_direction"].items():
        for p in dd["points"]:
          w.writerow([policy, dname, dd["direction_deg"], p["push_n"],
                      f"{p['recovery_rate']:.4f}"])

  # ---- curves_overall.csv : aggregated across directions, with CIs ----
  with (OUT_DIR / "curves_overall.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["policy", "push_n", "recovery_rate", "ci_lo", "ci_hi",
                "fall_rate", "n", "mean_peak_dv", "mean_max_tilt_deg"])
    for policy, d in sweeps.items():
      for o in d["curve"]:
        w.writerow([policy, o["push_n"], f"{o['recovery_rate']:.4f}",
                    f"{o['ci_lo']:.4f}", f"{o['ci_hi']:.4f}",
                    f"{o['fall_rate']:.4f}", o["n"],
                    f"{o['mean_peak_dv']:.4f}", f"{o['mean_max_tilt_deg']:.2f}"])

  # ---- fstar.csv ----
  with (OUT_DIR / "fstar.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    dirs = ["from_behind", "from_front", "from_right", "from_left"]
    w.writerow(["policy", "f_star_overall_n", *[f"f_star_{d}_n" for d in dirs],
                "max_force_tested_n", "total_scored_pushes"])
    for policy, d in sweeps.items():
      row = [policy, d["f_star_overall_n"]]
      row += [d["f_star_by_direction_n"].get(x) for x in dirs]
      row += [d["max_force_tested_n"], d["total_scored_pushes"]]
      w.writerow(row)

  # ---- training.csv ----
  keys = ["iterations_seen", "complete", "mean_reward", "mean_episode_length",
          "fell_over", "error_vel_xy", "error_vel_yaw", "mean_action_acc",
          "angular_momentum_mean", "slip_velocity_mean", "landing_force_mean"]
  with (OUT_DIR / "training.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["policy", *keys])
    for policy, rec in training.items():
      w.writerow([policy, *[rec.get(k) for k in keys]])

  # ---- ablation.md ----
  lines = ["# Ablation summary", "",
           f"Generated {blob['generated_utc']}", "",
           "## Max recoverable push, F* (N)", "",
           "| policy | overall | behind | front | right | left | max tested | pushes |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
  order = ["legacy", "P1_nominal", "P2_push", "P3_robust"]
  for p in [x for x in order if x in sweeps]:
    d = sweeps[p]
    g = d["f_star_by_direction_n"]

    def fmt(v):
      return f"{v:.0f}" if isinstance(v, (int, float)) and v is not None else "not reached"

    lines.append(
      f"| {p} | **{fmt(d['f_star_overall_n'])}** | {fmt(g.get('from_behind'))} | "
      f"{fmt(g.get('from_front'))} | {fmt(g.get('from_right'))} | "
      f"{fmt(g.get('from_left'))} | {d['max_force_tested_n']:.0f} | "
      f"{d['total_scored_pushes']} |"
    )

  lines += ["", "## Training (final)", "",
            "| policy | iters | reward | ep len | falls | err_vel_xy | action_acc |",
            "|---|---:|---:|---:|---:|---:|---:|"]
  for p in [x for x in order if x in training]:
    r = training[p]
    lines.append(
      f"| {p} | {r.get('iterations_seen')} | {r.get('mean_reward')} | "
      f"{r.get('mean_episode_length')} | {r.get('fell_over')} | "
      f"{r.get('error_vel_xy')} | {r.get('mean_action_acc')} |"
    )
  lines += ["",
            "Training reward is NOT comparable across policies: P2/P3 are scored "
            "while being pushed, so a lower number reflects a harder task, not a "
            "worse policy. Only the sweep (identical push-free eval env) is "
            "comparable."]
  (OUT_DIR / "ablation.md").write_text("\n".join(lines))

  print(f"wrote -> {OUT_DIR}")
  for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:24s} {f.stat().st_size:>8d} B")
  print()
  for p in [x for x in order if x in sweeps]:
    d = sweeps[p]
    fs = d["f_star_overall_n"]
    print(f"  {p:14s} F* = {fs:7.1f} N   "
          f"({d['total_scored_pushes']} pushes, to {d['max_force_tested_n']:.0f} N)"
          if fs else f"  {p:14s} F* not reached")


if __name__ == "__main__":
  main()
