# Disturbance sweep — p1_nominal

Checkpoint: `checkpoints/p1_nominal/model_6599.pt`

**F\* (50% recovery) = 337 N**

| force (N) | recovery | 95% CI | n | mean peak dv (m/s) |
|---:|---:|:---:|---:|---:|
| 0 | 1.000 | 0.99–1.00 | 256 | 0.22 |
| 50 | 1.000 | 0.99–1.00 | 256 | 0.32 |
| 100 | 1.000 | 0.99–1.00 | 256 | 0.53 |
| 150 | 0.992 | 0.97–1.00 | 256 | 0.77 |
| 200 | 0.961 | 0.93–0.98 | 256 | 1.01 |
| 250 | 0.854 | 0.81–0.89 | 261 | 1.24 |
| 300 | 0.663 | 0.61–0.72 | 276 | 1.45 |
| 350 | 0.441 | 0.39–0.50 | 306 | 1.68 |
| 400 | 0.230 | 0.19–0.28 | 378 | 1.87 |
| 450 | 0.116 | 0.09–0.15 | 430 | 2.16 |
| 500 | 0.049 | 0.03–0.07 | 465 | 2.39 |
| 550 | 0.025 | 0.01–0.04 | 487 | 2.62 |
| 600 | 0.006 | 0.00–0.02 | 504 | 2.81 |
| 650 | 0.000 | 0.00–0.01 | 512 | 2.98 |
| 700 | 0.000 | 0.00–0.01 | 512 | 3.14 |

## F* by direction

| direction | F* (N) |
|---|---:|
| from behind (shoved forward) | 397 |
| from its right (shoved left) | 312 |
| from the front (shoved back) | 387 |
| from its left (shoved right) | 267 |

### Note on sample counts

`n` grows with force because a fallen environment resets and is pushed again inside the same 7 s episode, so high-force cells accumulate more episodes. Rates are unaffected -- each push is scored independently -- but the confidence intervals are correspondingly tighter at high force.