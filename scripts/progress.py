"""Summarise training progress and project the final outcome.

Reads an rsl_rl training log and reports the reward/episode-length trajectory,
whether learning has plateaued, and the projected time remaining.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

log = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/sleem/runs/baseline.log")
text = log.read_text(errors="ignore")

rewards = [float(m) for m in re.findall(r"Mean reward:\s*(-?[\d.]+)", text)]
lengths = [float(m) for m in re.findall(r"Mean episode length:\s*([\d.]+)", text)]
iters = len(re.findall(r"Learning iteration", text))
eta = re.findall(r"ETA:\s*([\d:]+)", text)
itime = re.findall(r"Iteration time:\s*([\d.]+)s", text)

print(f"iterations: {iters}")
print(f"iter time:  {itime[-1] if itime else '?'}s   ETA: {eta[-1] if eta else '?'}")
print()

n = len(rewards)
if n:
  print("reward trajectory")
  for frac in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
    i = min(int(n * frac), n - 1)
    it = int(iters * frac)
    ln = lengths[i] if i < len(lengths) else float("nan")
    print(f"  iter ~{it:>5}   reward {rewards[i]:>7.2f}   ep_len {ln:>7.2f}")
  print()

  # Plateau check on the last 20% of reports.
  tail = rewards[-max(20, n // 5):]
  half = len(tail) // 2
  a = sum(tail[:half]) / max(1, half)
  b = sum(tail[half:]) / max(1, len(tail) - half)
  delta = b - a
  pct = (delta / abs(a) * 100) if a else 0.0
  print(f"recent trend (last {len(tail)} reports): {a:.2f} -> {b:.2f}  "
        f"({delta:+.2f}, {pct:+.1f}%)")
  if abs(pct) < 2.0:
    print("  => PLATEAUED: reward change under 2%; further training buys little")
  elif delta > 0:
    print("  => STILL IMPROVING: reward rising; continue")
  else:
    print("  => REGRESSING: reward falling; investigate")

if lengths:
  recent = lengths[-20:]
  avg = sum(recent) / len(recent)
  print()
  print(f"episode length (last 20): {avg:.1f} / 1000 max  "
        f"({avg / 10:.1f}% of the 20 s limit)")
  print(f"  Gate G1 needs >= 900 (18 s): "
        f"{'PASS' if avg >= 900 else 'not yet'}")
  print(f"  implied fall rate: ~{max(0.0, (1000 - avg) / 1000) * 100:.1f}% "
        f"(Gate G1 needs <= 5%)")
