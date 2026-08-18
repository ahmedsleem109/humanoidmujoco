# Disturbance sweep — r2_legacy_baseline

Checkpoint: `checkpoints/baseline_flat/model_6600_flat98.pt`

**F\* (50% recovery) = 431 N**

| force (N) | recovery | 95% CI | n | mean peak dv (m/s) |
|---:|---:|:---:|---:|---:|
| 0 | 1.000 | 0.99–1.00 | 256 | 0.17 |
| 50 | 1.000 | 0.99–1.00 | 256 | 0.29 |
| 100 | 1.000 | 0.99–1.00 | 256 | 0.51 |
| 150 | 1.000 | 0.99–1.00 | 256 | 0.77 |
| 200 | 1.000 | 0.99–1.00 | 256 | 1.03 |
| 250 | 1.000 | 0.99–1.00 | 256 | 1.29 |
| 300 | 1.000 | 0.99–1.00 | 256 | 1.55 |
| 350 | 0.894 | 0.85–0.93 | 263 | 1.78 |
| 400 | 0.683 | 0.63–0.74 | 278 | 2.04 |
| 450 | 0.391 | 0.34–0.45 | 317 | 2.23 |
| 500 | 0.191 | 0.16–0.23 | 392 | 2.40 |
| 550 | 0.070 | 0.05–0.10 | 443 | 2.63 |
| 600 | 0.010 | 0.00–0.02 | 481 | 2.85 |
| 650 | 0.000 | 0.00–0.01 | 504 | 3.04 |
| 700 | 0.000 | 0.00–0.01 | 511 | 3.22 |

## F* by direction

| direction | F* (N) |
|---|---:|
| from behind (shoved forward) | 514 |
| from its right (shoved left) | 404 |
| from the front (shoved back) | 462 |
| from its left (shoved right) | 388 |

### Note on sample counts

`n` grows with force because a fallen environment resets and is pushed again inside the same 7 s episode, so high-force cells accumulate more episodes. Rates are unaffected -- each push is scored independently -- but the confidence intervals are correspondingly tighter at high force.