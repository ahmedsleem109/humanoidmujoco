"""Analyse a disturbance sweep: per-direction F*, Wilson intervals, plots.

Reads the `results.json` written by `eval_push.py` and produces:

  * `curve.png`      recovery probability vs force, overall + per direction
  * `summary.md`     the table, with sample counts and confidence intervals
  * `analysis.json`  machine-readable, for the ablation table later

Every rate is reported with its `n` and a Wilson score interval. Wilson rather
than the normal approximation because the interesting region includes rates at
0.0 and 1.0, where the normal interval has zero width and is simply wrong.

Usage:  python scripts/analyze_sweep.py results/push/<run_dir> [more_dirs...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Direction labels use the convention in disturbance.py: the azimuth of the
# APPLIED FORCE in the robot's yaw frame.
DIR_LABEL = {
  0.0: "from behind (shoved forward)",
  45.0: "behind-right",
  90.0: "from its right (shoved left)",
  135.0: "front-right",
  180.0: "from the front (shoved back)",
  225.0: "front-left",
  270.0: "from its left (shoved right)",
  315.0: "behind-left",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  """Wilson score interval. Correct at p=0 and p=1, unlike the normal approx."""
  if n == 0:
    return (float("nan"), float("nan"))
  p = k / n
  d = 1 + z * z / n
  centre = (p + z * z / (2 * n)) / d
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (max(0.0, centre - half), min(1.0, centre + half))


def f_star(points: list[tuple[float, float]]) -> float | None:
  """Force at which recovery crosses 0.5, linearly interpolated."""
  pts = sorted(points)
  for (f0, r0), (f1, r1) in zip(pts, pts[1:]):
    if r0 >= 0.5 > r1:
      span = r0 - r1
      t = (r0 - 0.5) / span if span > 0 else 0.0
      return f0 + t * (f1 - f0)
  return None


def analyse(run_dir: Path) -> dict:
  data = json.loads((run_dir / "results.json").read_text())
  cells = data["cells"]
  label = data.get("label", run_dir.name)

  forces = sorted({c["push_n"] for c in cells})
  dirs = sorted({c["direction_deg"] for c in cells})

  per_dir: dict[float, list[tuple[float, float]]] = {d: [] for d in dirs}
  overall: list[dict] = []

  for f in forces:
    sel = [c for c in cells if c["push_n"] == f]
    n = sum(c["episodes"] for c in sel)
    k = sum(round(c["recovery_rate"] * c["episodes"]) for c in sel)
    lo, hi = wilson(k, n)
    overall.append(
      {"push_n": f, "n": n, "recovered": k, "rate": k / n if n else float("nan"),
       "ci_lo": lo, "ci_hi": hi,
       "mean_peak_dv": sum(c.get("mean_peak_dv", 0) * c["episodes"] for c in sel) / n
       if n else float("nan")}
    )
    for d in dirs:
      cd = [c for c in sel if c["direction_deg"] == d]
      nd = sum(c["episodes"] for c in cd)
      if nd:
        kd = sum(round(c["recovery_rate"] * c["episodes"]) for c in cd)
        per_dir[d].append((f, kd / nd))

  result = {
    "label": label,
    "checkpoint": data.get("checkpoint"),
    "f_star_overall_n": f_star([(o["push_n"], o["rate"]) for o in overall]),
    "f_star_by_direction_n": {str(d): f_star(per_dir[d]) for d in dirs},
    "curve": overall,
  }

  # ---- plot --------------------------------------------------------------
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  xs = [o["push_n"] for o in overall]
  ys = [o["rate"] for o in overall]
  lo = [o["rate"] - o["ci_lo"] for o in overall]
  hi = [o["ci_hi"] - o["rate"] for o in overall]
  ax1.errorbar(xs, ys, yerr=[lo, hi], marker="o", lw=2, capsize=3, color="#1f77b4")
  ax1.axhline(0.5, ls="--", lw=1, color="#888")
  fs = result["f_star_overall_n"]
  if fs:
    ax1.axvline(fs, ls=":", lw=1.5, color="#d62728")
    ax1.annotate(f"F* = {fs:.0f} N", (fs, 0.52), color="#d62728",
                 fontsize=11, ha="left")
  ax1.set_xlabel("push magnitude (N, 0.1 s torso impulse)")
  ax1.set_ylabel("recovery probability")
  ax1.set_title(f"Recovery vs disturbance — {label}")
  ax1.set_ylim(-0.03, 1.03)
  ax1.grid(alpha=0.3)

  for d in dirs:
    pts = sorted(per_dir[d])
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=1.8,
             label=DIR_LABEL.get(d, f"{d:.0f} deg"))
  ax2.axhline(0.5, ls="--", lw=1, color="#888")
  ax2.set_xlabel("push magnitude (N)")
  ax2.set_ylabel("recovery probability")
  ax2.set_title("By push direction")
  ax2.set_ylim(-0.03, 1.03)
  ax2.grid(alpha=0.3)
  ax2.legend(fontsize=8)

  fig.tight_layout()
  fig.savefig(run_dir / "curve.png", dpi=150)
  plt.close(fig)

  # ---- markdown ----------------------------------------------------------
  lines = [
    f"# Disturbance sweep — {label}", "",
    f"Checkpoint: `{data.get('checkpoint')}`", "",
    f"**F\\* (50% recovery) = {fs:.0f} N**" if fs else "**F\\* not reached**", "",
    "| force (N) | recovery | 95% CI | n | mean peak dv (m/s) |",
    "|---:|---:|:---:|---:|---:|",
  ]
  for o in overall:
    lines.append(
      f"| {o['push_n']:.0f} | {o['rate']:.3f} | "
      f"{o['ci_lo']:.2f}–{o['ci_hi']:.2f} | {o['n']} | {o['mean_peak_dv']:.2f} |"
    )
  lines += ["", "## F* by direction", "",
            "| direction | F* (N) |", "|---|---:|"]
  for d in dirs:
    v = result["f_star_by_direction_n"][str(d)]
    lines.append(f"| {DIR_LABEL.get(d, f'{d:.0f} deg')} | "
                 f"{v:.0f} |" if v else f"| {DIR_LABEL.get(d, d)} | not reached |")
  lines += [
    "", "### Note on sample counts",
    "",
    "`n` grows with force because a fallen environment resets and is pushed "
    "again inside the same 7 s episode, so high-force cells accumulate more "
    "episodes. Rates are unaffected -- each push is scored independently -- but "
    "the confidence intervals are correspondingly tighter at high force.",
  ]
  (run_dir / "summary.md").write_text("\n".join(lines))
  (run_dir / "analysis.json").write_text(json.dumps(result, indent=2))
  return result


if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise SystemExit(__doc__)
  for arg in sys.argv[1:]:
    r = analyse(Path(arg))
    print(f"\n=== {r['label']} ===")
    print(f"F* overall: {r['f_star_overall_n']}")
    for d, v in r["f_star_by_direction_n"].items():
      print(f"  dir {float(d):5.1f} deg -> F* = {v if v else 'not reached'}")
