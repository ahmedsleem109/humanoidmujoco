"""Publication figures for the README, in light and dark variants.

Two variants because GitHub renders READMEs in both themes and a white-background
PNG looks broken in dark mode; the README pairs them with <picture> +
prefers-color-scheme.

Palette is the validated categorical order (slots 1-3), assigned by entity and
never by rank, so a policy keeps its colour across every figure:

    P1 nominal   blue      P2 +pushes   orange      P3 +pushes+DR   aqua

Both variants passed scripts/validate_palette.js on the all-pairs list
(worst CVD dE 9.2 light / 9.4 dark; normal-vision 24.0 / 20.9). Every series is
also directly labelled, so identity never rests on colour alone -- which is
required anyway, since aqua sits below 3:1 on the light surface.

Usage:  python scripts/make_figures.py [--out results/figures]
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

THEMES = {
  "light": {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": {"P1_nominal": "#2a78d6", "P2_push": "#eb6834",
               "P3_robust": "#1baf7a", "legacy": "#898781"},
  },
  "dark": {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "series": {"P1_nominal": "#3987e5", "P2_push": "#d95926",
               "P3_robust": "#199e70", "legacy": "#898781"},
  },
}

LABEL = {
  "P1_nominal": "P1  nominal",
  "P2_push": "P2  + pushes",
  "P3_robust": "P3  + pushes & DR",
  "legacy": "legacy baseline",
}
ORDER = ["P1_nominal", "P2_push", "P3_robust"]

DIRS = [
  ("from_behind", "from behind"),
  ("from_front", "from the front"),
  ("from_right", "from its right"),
  ("from_left", "from its left"),
]


def style(ax, t) -> None:
  ax.set_facecolor(t["surface"])
  ax.grid(True, color=t["grid"], lw=0.8, zorder=0)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(t["axis"])
    ax.spines[side].set_linewidth(1.0)
  ax.tick_params(colors=t["muted"], labelsize=10, length=0)
  for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_color(t["ink2"])


def fig_curves(m: dict, t: dict, dest: Path) -> None:
  """Recovery probability vs push magnitude -- the headline figure."""
  fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=t["surface"])
  style(ax, t)

  for pol in ORDER:
    d = m["sweeps"][pol]
    xs = [c["push_n"] for c in d["curve"]]
    ys = [c["recovery_rate"] for c in d["curve"]]
    lo = [c["ci_lo"] for c in d["curve"]]
    hi = [c["ci_hi"] for c in d["curve"]]
    col = t["series"][pol]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
    ax.plot(xs, ys, color=col, lw=2.0, zorder=3, solid_capstyle="round")
    ax.plot(xs, ys, "o", color=col, ms=4.5, zorder=4,
            markeredgecolor=t["surface"], markeredgewidth=1.5)

    fs = d["f_star_overall_n"]
    ax.plot([fs, fs], [0, 0.5], color=col, lw=1.2, ls=(0, (2, 3)), zorder=2)
    ax.plot([fs], [0.5], "o", color=col, ms=8, zorder=5,
            markeredgecolor=t["surface"], markeredgewidth=2)

  ax.axhline(0.5, color=t["muted"], lw=1.0, ls=(0, (4, 4)), zorder=1)
  ax.text(1008, 0.53, "50% recovery", color=t["muted"], fontsize=9,
          va="bottom", ha="right")

  # Legend sits in the lower-left, the one region all three curves leave empty
  # (every policy is near 1.0 below 250 N). Swatch + label + F*, so identity
  # never rests on colour alone -- required here anyway, since aqua is below
  # 3:1 on the light surface.
  # One text object per row: a second column of F* values would run into the
  # P1 curve, which descends through this quadrant between 250 and 500 N.
  for i, pol in enumerate(ORDER):
    y = 0.285 - i * 0.082
    col = t["series"][pol]
    ax.plot([28, 76], [y, y], color=col, lw=3.0, zorder=6,
            solid_capstyle="round")
    short = LABEL[pol].split("  ", 1)
    ax.text(90, y, f"{short[0]}  {short[1]}   ·   "
                   f"F* {m['sweeps'][pol]['f_star_overall_n']:.0f} N",
            color=t["ink2"], fontsize=11, va="center", fontweight="bold",
            zorder=6)

  ax.set_xlabel("push magnitude  (N, 0.1 s torso impulse)",
                color=t["ink2"], fontsize=11, labelpad=8)
  ax.set_ylabel("recovery probability", color=t["ink2"], fontsize=11, labelpad=8)
  ax.set_title("How hard a push can it take?",
               color=t["ink"], fontsize=15.5, fontweight="bold", loc="left",
               pad=34)
  ax.text(0, 1.035,
          "shaded band = 95% Wilson interval  ·  22,000+ scored pushes  ·  "
          "dot marks F*, the 50% crossing",
          transform=ax.transAxes, color=t["muted"], fontsize=9.5)
  ax.set_xlim(-15, 1015)
  ax.set_ylim(-0.03, 1.06)
  fig.tight_layout()
  fig.savefig(dest, dpi=170, facecolor=t["surface"])
  plt.close(fig)


def fig_directions(m: dict, t: dict, dest: Path) -> None:
  """F* per push direction -- shows the lateral inversion."""
  fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=t["surface"])
  style(ax, t)

  n = len(ORDER)
  width = 0.26
  for i, pol in enumerate(ORDER):
    g = m["sweeps"][pol]["f_star_by_direction_n"]
    vals = [g.get(k) or 0 for k, _ in DIRS]
    xs = [j + (i - (n - 1) / 2) * (width + 0.02) for j in range(len(DIRS))]
    ax.bar(xs, vals, width=width, color=t["series"][pol], zorder=3,
           label=LABEL[pol], edgecolor=t["surface"], linewidth=1.5)
    for x, v in zip(xs, vals):
      ax.text(x, v + 14, f"{v:.0f}", ha="center", color=t["ink2"],
              fontsize=9.5, fontweight="bold")

  ax.set_xticks(range(len(DIRS)))
  ax.set_xticklabels([lbl for _, lbl in DIRS], fontsize=11)
  ax.set_ylabel("F*  (N)", color=t["ink2"], fontsize=11, labelpad=8)
  ax.set_title("The weakest direction becomes the strongest",
               color=t["ink"], fontsize=15.5, fontweight="bold", loc="left",
               pad=34)
  ax.text(0, 1.035,
          "lateral was P1's worst axis at 267 N — after disturbance training it leads",
          transform=ax.transAxes, color=t["muted"], fontsize=9.5)
  ax.set_ylim(0, 860)

  leg = ax.legend(frameon=False, fontsize=11, loc="upper left",
                  ncol=3, bbox_to_anchor=(0.0, -0.12))
  for txt in leg.get_texts():
    txt.set_color(t["ink2"])
  fig.tight_layout()
  fig.savefig(dest, dpi=170, facecolor=t["surface"], bbox_inches="tight")
  plt.close(fig)


def fig_gain(m: dict, t: dict, dest: Path) -> None:
  """Where the robustness actually comes from -- the ablation, as one figure."""
  fig, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=t["surface"])
  style(ax, t)
  ax.grid(True, axis="x", color=t["grid"], lw=0.8)
  ax.grid(False, axis="y")

  vals = [m["sweeps"][p]["f_star_overall_n"] for p in ORDER]
  ys = list(range(len(ORDER)))[::-1]
  for y, pol, v in zip(ys, ORDER, vals):
    ax.barh(y, v, height=0.52, color=t["series"][pol], zorder=3)
    ax.text(v + 12, y, f"{v:.0f} N", va="center", color=t["ink"],
            fontsize=13, fontweight="bold")

  ax.set_yticks(ys)
  ax.set_yticklabels([LABEL[p] for p in ORDER], fontsize=12)
  ax.set_xlim(0, 790)
  ax.set_xlabel("F*  —  maximum recoverable push (N)",
                color=t["ink2"], fontsize=11, labelpad=8)
  ax.set_title("Which ingredient creates robustness?",
               color=t["ink"], fontsize=15.5, fontweight="bold", loc="left",
               pad=18)

  # The two deltas are the whole story; annotate them directly.
  ax.annotate("", xy=(vals[1], ys[1] + 0.42), xytext=(vals[0], ys[1] + 0.42),
              arrowprops=dict(arrowstyle="-|>", color=t["ink2"], lw=1.4))
  ax.text((vals[0] + vals[1]) / 2, ys[1] + 0.62, "+90%",
          ha="center", color=t["ink"], fontsize=12, fontweight="bold")
  ax.annotate("", xy=(vals[2], ys[2] + 0.42), xytext=(vals[1], ys[2] + 0.42),
              arrowprops=dict(arrowstyle="-|>", color=t["ink2"], lw=1.4))
  ax.text(vals[2] + 60, ys[2] + 0.62, "+2.4%",
          ha="left", color=t["muted"], fontsize=12, fontweight="bold")
  fig.tight_layout()
  fig.savefig(dest, dpi=170, facecolor=t["surface"], bbox_inches="tight")
  plt.close(fig)


def main(out: str = "results/figures") -> None:
  m = json.loads((REPO / "results/metrics/all_metrics.json").read_text())
  dest = REPO / out
  dest.mkdir(parents=True, exist_ok=True)

  for mode, t in THEMES.items():
    fig_curves(m, t, dest / f"recovery_curves_{mode}.png")
    fig_directions(m, t, dest / f"fstar_by_direction_{mode}.png")
    fig_gain(m, t, dest / f"ablation_{mode}.png")

  print(f"wrote -> {dest}")
  for f in sorted(dest.iterdir()):
    print(f"  {f.name:34s} {f.stat().st_size // 1024:>5d} KB")


if __name__ == "__main__":
  tyro.cli(main)
