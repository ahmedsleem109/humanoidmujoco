# P1 — Nominal

One arm of the three-policy ablation in
[`main`](../../tree/main). `main` is the complete tree; this branch isolates this
policy's checkpoint, sweep and video.

## Trained with

No disturbances. No dynamics randomization. Friction pinned at 0.9.

Everything else — reward function (16 terms), observations, command distribution,
network, seed, 4096 environments, 6600 PPO iterations — is **identical** across all
three arms. Verified by `scripts/r1_verify_variants.py`, which exits non-zero if
anything outside the event dict differs.

## Result

| | |
|---|---|
| **F\*** (50% recovery) | **337 N** |
| by direction | behind 397 · front 387 · right 312 · **left 267** |

The control arm, and the strongest *nominal* walker of the three: reward 37.0, a full 1000/1000-step episode, **0.0% falls undisturbed**. That matters — a baseline that walks beautifully and still collapses under a routine push is a far stronger premise than a mediocre one.

Its weakest axis is lateral at 267 N, which sits **inside** the published 50–300 N benchmark band. It fails at forces the literature treats as routine.

## Files on this branch

```
checkpoints/p1_nominal/model_6599.pt
results/push/p1_nominal/           results.json, curve.png, summary.md, episodes.csv
results/videos/v1_problem.mp4
results/metrics/                 full ablation across all three arms
```

## Reproduce

```bash
./scripts/run_sweep.sh checkpoints/p1_nominal/model_6599.pt p1_nominal
python scripts/analyze_sweep.py results/push/p1_nominal
```

---

Full context: [README](README.md) · [engineering log](docs/ENGINEERING_LOG.md) ·
[frozen protocol](configs/protocol.yaml)
