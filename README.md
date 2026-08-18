# Robust Humanoid Locomotion Under Unexpected Disturbances

**A controlled ablation of robustness training for a Unitree G1 in MuJoCo.**
Three policies, identical in every respect except the disturbances they saw during
training, evaluated under 22,000+ scored pushes.

https://github.com/user-attachments/assets/v3_hero
<!-- If the embed above does not render, the clip is at results/videos/v3_hero.mp4 -->

> Same robot. Same seed. Same command. Same 550 N push.
> Only the training distribution differs.

---

## The result in one table

**F\*** = the push magnitude at which the policy recovers 50% of the time.
Torso impulse, 0.1 s, applied while walking forward at 1.0 m/s.

| policy | trained with | **F\*** | vs nominal |
|---|---|---:|---:|
| **P1** nominal | nothing | **337 N** | — |
| **P2** + disturbances | random impulses | **639 N** | **+90%** |
| **P3** + disturbances + domain rand. | impulses, friction, mass, damping, gains | **654 N** | **+94%** |

### The finding

**Disturbance randomization does essentially all of the work. Domain randomization
adds +2.4% — within noise, and one direction got *worse*.**

That reproduces a contested result ([Xie et al., 2020](https://arxiv.org/abs/2011.02404)
found dynamics randomization is not always necessary) on a humanoid, with a
three-arm ablation rather than a two-way comparison.

**Important caveat, stated up front:** this measures robustness to *forces*. P2 has
never seen a different friction, mass or damping, so the axis where domain
randomization *should* pay — transfer to different dynamics — is not yet tested
here. Do not read this as "domain randomization is useless."

---

## The unexpected result: lateral balance is a training artifact

`F*` broken down by push direction:

| direction | P1 | P2 | P3 | P1→P2 |
|---|---:|---:|---:|---:|
| from behind | 397 | 591 | 628 | +49% |
| from the front | 387 | 598 | 594 | +55% |
| from its right | 312 | 654 | 668 | +110% |
| **from its left** | **267** | **711** | **750** | **+166%** |

Lateral was the nominal policy's **weakest** axis — the textbook bipedal result,
since the frontal plane has far less base of support.

After disturbance training it becomes the **strongest**.

The conclusion is not that lateral balance is easy. It is that **the frontal plane
is the axis with the most headroom left when you do not train for it.** The classic
finding measures untrained policies and attributes to morphology what is partly a
property of the training distribution.

---

## Generalization beyond the training range

P2's training distribution tops out at **346 N** realisable. Measured recovery
above that:

| push | vs training max | P2 recovery |
|---:|---|---:|
| 450 N | +30% | 0.860 |
| 500 N | +44% | 0.820 |
| 550 N | +59% | 0.732 |

It holds above 80% at forces 44% beyond anything it experienced.

---

## Videos

| clip | what it shows |
|---|---|
| [`v1_problem.mp4`](results/videos/v1_problem.mp4) | The problem. One policy, with and without a push. |
| [`v2_p1_vs_p2.mp4`](results/videos/v2_p1_vs_p2.mp4) | P1 vs P2 under an identical push. |
| [`v3_hero.mp4`](results/videos/v3_hero.mp4) | All three policies, identical push. |

Every clip uses the same seed, command, camera and push. Only the checkpoint changes.
The renderer **refuses to emit a clip whose push did not fire** — see
[the engineering log](docs/ENGINEERING_LOG.md) for why that check exists.

---

## Method

```
MuJoCo (mjlab / MuJoCo-Warp)
        ↓
G1 velocity-tracking locomotion, 29 DoF
        ↓
PPO, 4096 envs, 6600 iterations          ← identical for all three arms
        ↓
P1: nothing   P2: + impulses   P3: + impulses + domain randomization
        ↓
Deterministic evaluation push (exact N, azimuth, instant)
        ↓
Latched per-episode recovery metric
```

### What makes this an ablation rather than a comparison

All three policies share an **identical** reward function (16 terms, none added by
this project), observation space, command distribution, network, seed, environment
count and iteration budget. The **only** difference is the event dict.

That is verified mechanically, not asserted:

```bash
python scripts/r1_verify_variants.py
```

It compares reward weights, observation terms, corruption flags, command ranges,
episode length, decimation and curriculum across all three arms, and exits
non-zero if anything outside `events` differs.

### Observations

The actor is **proprioceptive**: base angular velocity, projected gravity, velocity
command, gait phase, joint positions, joint velocities, previous actions — all with
noise. **Force magnitude, direction and timing are never observable.** The policy
reacts to consequences.

(The critic is privileged, as is standard for asymmetric actor-critic. Stated
explicitly because "proprioceptive policy" would be misleading.)

### Evaluation protocol

Adopted from the published G1 push-recovery benchmark so numbers are comparable
rather than self-referential: torso impulses, 0.1 s, 45° tilt failure bound.

| | |
|---|---|
| push | `torso_link`, exact magnitude and azimuth, 0.1 s, at t=3 s |
| command | fixed 1.0 m/s forward, no heading randomization, no standing envs |
| **fall** | terminated, or torso tilt > 45° |
| **recovered** | within 3 s: never fell, tilt back under 15°, **and** velocity tracking back within 1.5× its pre-push mean — both held 0.5 s |
| grid | 0–1000 N × 4 directions × 64 envs |

The velocity condition is load-bearing: **a policy that survives by freezing in a
crouch must not score as a recovery.**

The full protocol was **frozen before any policy was trained**
([`configs/protocol.yaml`](configs/protocol.yaml)) so the generalization test could
not be defined after seeing results.

---

## Reproducing

Trained on a **laptop RTX 3060, 6 GB** — ~9.5 h per policy, ~28 h total.

```bash
# 1. Verify the environment (pins, DR surface, reward audit)
python scripts/r0_audit.py

# 2. Confirm the three arms differ only in `events`
python scripts/r1_verify_variants.py

# 3. Train one arm (4096 envs, 6600 iters)
./scripts/run_train.sh p1nominal Unitree-G1-Robust-P1-Nominal 4096 6600

# 4. Sweep a checkpoint (~40 min)
./scripts/run_sweep.sh checkpoints/p1_nominal/model_6599.pt p1_nominal

# 5. Per-direction F*, Wilson intervals, plots
python scripts/analyze_sweep.py results/push/p1_nominal

# 6. Consolidate everything into results/metrics/
python scripts/collect_metrics.py

# 7. Render a comparison video
python scripts/make_split_video.py \
  --panels 'NOMINAL=checkpoints/p1_nominal/model_6599.pt,ROBUST=checkpoints/p3_robust/model_6599.pt' \
  --force-n 550 --direction-deg 270 --label compare
```

Pinned versions are load-bearing — see [`docs/HANDOFF.md`](docs/HANDOFF.md) §2 for why
each one matters:

```
torch==2.9.1+cu129   mjlab==1.2.0   mujoco==3.5.0   warp-lang==1.12.0
```

---

## What broke, and how it was found

Most of the value in this project is in [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md).
Every entry states **what looked fine**, **what was actually wrong**, and **the
measurement that distinguished them**. Four highlights:

**1. The recovery metric silently discarded every failure.**
Recovery read 1.000 up to 300 N. Plausible — until you looked at the sample counts:
`n=2` out of 32 at 600 N, `n=0` at 1000 N. The push event cleared its per-env state
*inside* `env.step`, so every episode that fell and reset erased its own evidence
before the tracker looked. Recovery rate is computed over scored episodes, and the
dropped ones were **exactly the failures** — the metric would have reported the
policy as invincible precisely where it was collapsing.
→ *Every rate in this repo is reported with its `n`.*

**2. The video showed the opposite of what happened.**
A clip showed the nominal policy walking calmly after a 550 N push that it fails
**128 times out of 128**. It had fallen at t≈3.2 s, the episode terminated, **reset**,
and the robot stood up and walked again inside the same clip. A frame at t=8 s reads
as "it shrugged off the push."
→ *Fall termination is disabled for video only, and the renderer now refuses to emit
a clip whose push did not fire.*

**3. `dr.pseudo_inertia` cost a fixed ~4 GB and made training 9× slower.**
43 s/iteration instead of 4.7 — 79 hours instead of 10. The overhead is nearly
independent of environment count (5,649 MB at 1024 envs, 5,914 MB at 2048), so it is
the `set_const` recomputation, not per-env storage.
→ *And the diagnostic probe initially blamed the wrong term, because it leaked warp
allocations between variants — a memory-leak bug inside the memory-leak diagnostic.*

**4. Single clips at probabilistic forces are not reproducible.**
Two renders at an identical 450 N and seed gave opposite outcomes. MuJoCo-Warp on GPU
is not bit-deterministic, and the nominal policy recovers 1.6% of the time there.
Re-rendering until the desired outcome appears is cherry-picking.
→ *Videos use conditions where the outcome is not a coin flip (550 N lateral: 0/128
recoveries), and captions state the measured probability.*

---

## Data

Everything is machine-readable and regenerable — no hand-typed numbers.

| file | contents |
|---|---|
| [`results/metrics/all_metrics.json`](results/metrics/all_metrics.json) | everything, nested |
| [`results/metrics/curves.csv`](results/metrics/curves.csv) | policy × direction × force → recovery |
| [`results/metrics/curves_overall.csv`](results/metrics/curves_overall.csv) | aggregated, with Wilson intervals |
| [`results/metrics/fstar.csv`](results/metrics/fstar.csv) | policy × direction → F\* |
| [`results/metrics/ablation.md`](results/metrics/ablation.md) | the summary table |
| `results/push/*/` | raw sweeps: `results.json`, `curve.png`, per-episode CSV |

`python scripts/collect_metrics.py` rebuilds all of it from the raw artifacts.

---

## Limitations

Stated plainly rather than buried:

- **Simulation only.** No real-robot validation, and therefore **no sim-to-real
  claim**. `F*` in MuJoCo is a property of the contact solver as much as of the
  policy — treat it as a *relative* measure across these three policies, not an
  absolute spec.
- **Domain randomization is not fully tested.** The unseen-dynamics evaluation
  (friction ±50%, mass ±25%, damping ±30%, consecutive pushes) is specified in
  `configs/protocol.yaml` but **not yet run**. P3's real case is unproven.
- **Mass randomization models a point mass at the CoM** (a payload), not a density
  change. `dr.pseudo_inertia` would have been more physical but was unaffordable —
  see the engineering log.
- **Flat terrain, one gait speed, one robot.**
- **P1's advantage is understated:** it converges to a *better* nominal walker than
  the others (reward 37.0, 0.0% falls undisturbed), which makes its collapse under
  push a stronger baseline, not a weaker one.

---

## Attribution

This repository is derived from
[**unitreerobotics/unitree_rl_mjlab**](https://github.com/unitreerobotics/unitree_rl_mjlab)
and builds on [**mjlab**](https://github.com/mujocolab/mjlab) and
[**MuJoCo**](https://github.com/google-deepmind/mujoco) / MuJoCo-Warp.

The G1 robot model, the base velocity task, the PPO runner and the mjlab manager
framework are upstream work. The contributions here are the disturbance/robustness
study: `src/tasks/velocity/mdp/disturbance.py`,
`src/tasks/velocity/mdp/recovery_metrics.py`,
`src/tasks/velocity/config/g1/robust_env_cfg.py`, the evaluation and analysis
scripts, the protocol, and the documentation.

> ⚠️ **License:** upstream terms govern the derived code. Confirm
> `unitree_rl_mjlab`'s license and add the corresponding `LICENSE` file before
> treating this repository as freely reusable.

Non-G1 robot mesh directories were removed to keep the repository small; their
constants modules are retained because the package imports them.
