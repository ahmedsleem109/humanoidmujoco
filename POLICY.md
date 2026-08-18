# P2 — Disturbance Randomization

One arm of the three-policy ablation in
[`main`](../../tree/main). `main` is the complete tree; this branch isolates this
policy's checkpoint, sweep and video.

## Trained with

Random torso impulses via `apply_body_impulse`: ±200 N per component (up to 346 N realised), 0.05–0.20 s, every 2–5 s, independent timer per environment. Dynamics still pinned.

Everything else — reward function (16 terms), observations, command distribution,
network, seed, 4096 environments, 6600 PPO iterations — is **identical** across all
three arms. Verified by `scripts/r1_verify_variants.py`, which exits non-zero if
anything outside the event dict differs.

## Result

| | |
|---|---|
| **F\*** (50% recovery) | **639 N  (+90% over P1)** |
| by direction | behind 591 · front 598 · right 654 · **left 711** |

The arm that does the work. +90% over nominal.

It also **generalizes past its training range**: trained to 346 N, it recovers 86% at 450 N and 82% at 500 N — 44% beyond anything it saw.

And it inverts the lateral result: P1's weakest axis (267 N) becomes P2's strongest (711 N, +166%).

## Files on this branch

```
checkpoints/p2_push/model_6599.pt
results/push/p2_push/           results.json, curve.png, summary.md, episodes.csv
results/videos/v2_p1_vs_p2.mp4
results/metrics/                 full ablation across all three arms
```

## Reproduce

```bash
./scripts/run_sweep.sh checkpoints/p2_push/model_6599.pt p2_push
python scripts/analyze_sweep.py results/push/p2_push
```

---

Full context: [README](README.md) · [engineering log](docs/ENGINEERING_LOG.md) ·
[frozen protocol](configs/protocol.yaml)
