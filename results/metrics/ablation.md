# Ablation summary

Generated 2026-08-17T20:53:57+00:00

## Max recoverable push, F* (N)

| policy | overall | behind | front | right | left | max tested | pushes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | **431** | 514 | 462 | 404 | 388 | 700 | 4981 |
| P1_nominal | **337** | 397 | 387 | 312 | 267 | 700 | 5411 |
| P2_push | **639** | 591 | 598 | 654 | 711 | 1000 | 6447 |
| P3_robust | **654** | 628 | 594 | 668 | 750 | 1000 | 6333 |

## Training (final)

| policy | iters | reward | ep len | falls | err_vel_xy | action_acc |
|---|---:|---:|---:|---:|---:|---:|
| P1_nominal | 6599 | 38.24 | 1000.0 | 0.0 | 0.6966 | 0.6832 |
| P2_push | 6599 | 13.85 | 1000.0 | 0.0 | 1.3061 | 0.9527 |
| P3_robust | 6599 | 8.78 | 981.89 | 0.125 | 1.5578 | 0.9862 |

Training reward is NOT comparable across policies: P2/P3 are scored while being pushed, so a lower number reflects a harder task, not a worse policy. Only the sweep (identical push-free eval env) is comparable.