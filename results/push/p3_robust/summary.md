# Disturbance sweep — p3_robust

Checkpoint: `checkpoints/p3_robust/model_6599.pt`

**F\* (50% recovery) = 655 N**

| force (N) | recovery | 95% CI | n | mean peak dv (m/s) |
|---:|---:|:---:|---:|---:|
| 0 | 1.000 | 0.99–1.00 | 256 | 0.29 |
| 50 | 1.000 | 0.99–1.00 | 256 | 0.38 |
| 100 | 1.000 | 0.99–1.00 | 256 | 0.58 |
| 150 | 1.000 | 0.99–1.00 | 256 | 0.81 |
| 200 | 1.000 | 0.99–1.00 | 256 | 1.06 |
| 250 | 0.996 | 0.98–1.00 | 256 | 1.31 |
| 300 | 0.992 | 0.97–1.00 | 256 | 1.56 |
| 350 | 0.969 | 0.94–0.98 | 257 | 1.80 |
| 400 | 0.961 | 0.93–0.98 | 256 | 2.06 |
| 450 | 0.961 | 0.93–0.98 | 256 | 2.30 |
| 500 | 0.927 | 0.89–0.95 | 261 | 2.51 |
| 550 | 0.798 | 0.75–0.84 | 272 | 2.71 |
| 600 | 0.669 | 0.61–0.72 | 290 | 2.92 |
| 650 | 0.514 | 0.46–0.57 | 319 | 3.13 |
| 700 | 0.369 | 0.32–0.42 | 350 | 3.35 |

## F* by direction

| direction | F* (N) |
|---|---:|
| from behind (shoved forward) | 628 |
| from its right (shoved left) | 671 |
| from the front (shoved back) | 594 |
| from its left (shoved right) | not reached |

### Note on sample counts

`n` grows with force because a fallen environment resets and is pushed again inside the same 7 s episode, so high-force cells accumulate more episodes. Rates are unaffected -- each push is scored independently -- but the confidence intervals are correspondingly tighter at high force.