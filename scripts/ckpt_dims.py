"""Print the actor/critic input width of every checkpoint under checkpoints/.

HANDOFF.md says model_6600.pt is the 98-dim flat baseline and
model_6600_obstacle.pt is its 103-dim widened twin. Verify rather than trust:
loading the wrong one into the flat env fails loudly, but loading a subtly wrong
one would not.
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "checkpoints"

# Candidate originals: the flat-baseline training runs still hold their own
# checkpoints, which is how a clobbered staged copy can be recovered.
EXTRA = [
  REPO / "logs/rsl_rl/g1_velocity/2026-08-10_13-23-26/model_6600.pt",
  REPO / "logs/rsl_rl/g1_velocity/2026-08-10_13-07-20/model_6600.pt",
  REPO / "logs/rsl_rl/g1_velocity/2026-08-10_12-57-32/model_6600.pt",
]

for p in [*sorted(ROOT.rglob("*.pt")), *[e for e in EXTRA if e.exists()]]:
  try:
    d = torch.load(p, map_location="cpu", weights_only=False)
  except Exception as exc:  # noqa: BLE001
    print(f"{p.relative_to(ROOT)}: UNREADABLE {exc}")
    continue

  actor = d.get("actor_state_dict", {})
  critic = d.get("critic_state_dict", {})

  def width(sd: dict) -> str:
    for k, v in sd.items():
      if k.endswith("mlp.0.weight") or k == "mlp.0.weight":
        return str(tuple(v.shape))
    return "?"

  it = d.get("iter", d.get("iteration", "?"))
  try:
    name = str(p.relative_to(REPO))
  except ValueError:
    name = str(p)
  print(f"{name:62s} actor{width(actor):>14s} "
        f"critic{width(critic):>14s}  iter={it}")
