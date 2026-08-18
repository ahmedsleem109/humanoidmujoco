# Disturbance sweep — p2_push

Checkpoint: `checkpoints/p2_push/model_6599.pt`

**F\* (50% recovery) = 639 N**

| force (N) | recovery | 95% CI | n | mean peak dv (m/s) |
|---:|---:|:---:|---:|---:|
| 0 | 1.000 | 0.99–1.00 | 256 | 0.25 |
| 50 | 1.000 | 0.99–1.00 | 256 | 0.36 |
| 100 | 1.000 | 0.99–1.00 | 256 | 0.58 |
| 150 | 0.996 | 0.98–1.00 | 256 | 0.83 |
| 200 | 0.973 | 0.94–0.99 | 257 | 1.09 |
| 250 | 0.954 | 0.92–0.97 | 260 | 1.33 |
| 300 | 0.902 | 0.86–0.93 | 264 | 1.58 |
| 350 | 0.901 | 0.86–0.93 | 263 | 1.84 |
| 400 | 0.869 | 0.82–0.90 | 268 | 2.06 |
| 450 | 0.860 | 0.81–0.90 | 265 | 2.33 |
| 500 | 0.820 | 0.77–0.86 | 266 | 2.57 |
| 550 | 0.732 | 0.68–0.78 | 280 | 2.77 |
| 600 | 0.628 | 0.57–0.68 | 293 | 2.99 |
| 650 | 0.464 | 0.41–0.52 | 321 | 3.19 |
| 700 | 0.323 | 0.28–0.37 | 359 | 3.37 |

## F* by direction

| direction | F* (N) |
|---|---:|
| from behind (shoved forward) | 591 |
| from its right (shoved left) | 654 |
| from the front (shoved back) | 598 |
| from its left (shoved right) | not reached |

### Note on sample counts

`n` grows with force because a fallen environment resets and is pushed again inside the same 7 s episode, so high-force cells accumulate more episodes. Rates are unaffected -- each push is scored independently -- but the confidence intervals are correspondingly tighter at high force.