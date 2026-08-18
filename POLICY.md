# P3 — Disturbance + Domain Randomization

One arm of the three-policy ablation in
[`main`](../../tree/main). `main` is the complete tree; this branch isolates this
policy's checkpoint, sweep and video.

## Trained with

Everything P2 had, plus: foot friction 0.5–1.3, link mass ±15%, joint damping ±20%, joint friction ±20%, actuator PD gains ±10%, encoder bias ±0.015 rad, base CoM ±5 cm, randomized initial joint state.

Everything else — reward function (16 terms), observations, command distribution,
network, seed, 4096 environments, 6600 PPO iterations — is **identical** across all
three arms. Verified by `scripts/r1_verify_variants.py`, which exits non-zero if
anything outside the event dict differs.

## Result

| | |
|---|---|
| **F\*** (50% recovery) | **654 N  (+2.4% over P2)** |
| by direction | behind 628 · front 594 · right 668 · **left 750** |

**The headline null result.** Domain randomization adds +2.4% over disturbance randomization alone — within noise — and the 'from the front' direction actually got *worse* (598 → 594).

This reproduces a contested finding ([Xie et al. 2020](https://arxiv.org/abs/2011.02404)) on a humanoid.

⚠️ **P3 has not had its fair test yet.** It should win on transfer to *different dynamics*, not on raw force — and P2 has never seen a different friction, mass or damping. That evaluation is specified in `configs/protocol.yaml` and not yet run. Do not read this as 'domain randomization is useless'.

## Files on this branch

```
checkpoints/p3_robust/model_6599.pt
results/push/p3_robust/           results.json, curve.png, summary.md, episodes.csv
results/videos/v3_hero.mp4
results/metrics/                 full ablation across all three arms
```

## Reproduce

```bash
./scripts/run_sweep.sh checkpoints/p3_robust/model_6599.pt p3_robust
python scripts/analyze_sweep.py results/push/p3_robust
```

---

Full context: [README](README.md) · [engineering log](docs/ENGINEERING_LOG.md) ·
[frozen protocol](configs/protocol.yaml)
