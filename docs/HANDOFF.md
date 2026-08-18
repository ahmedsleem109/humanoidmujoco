# Session Handoff — Robust Humanoid Locomotion Under Disturbances

**Read this first in a new session.** Everything needed to continue without
re-deriving anything.

- **Repo:** `D:\humanoid\unitree_rl_mjlab`
- **Brief:** `D:\humanoid\neworientation.md`
- **Plan:** `ROBUSTNESS_PLAN.md` (4 phases, gates R0–R9)
- **Engineering log:** `ROBUSTNESS_LOG.md` — bugs + the measurement that caught each
- **Frozen protocol:** `configs/protocol.yaml` — do not edit silently
- **All metrics:** `results/metrics/` — regenerate with `scripts/collect_metrics.py`
- **Predecessor:** the G1 obstacle project is **closed**, retained in-directory.
  Its `HANDOFF.md` gotchas 1–18 all still apply.
- **Last session:** 2026-08-16 → 2026-08-17

---

## 1. Where we are

| Phase | Status |
|---|---|
| R0 environment revalidation | ✅ PASS |
| R1 new code (5 modules) | ✅ complete |
| R2 kill-switch #1 (legacy sweep) | ✅ PASS — F\* = 431 N |
| R3 protocol frozen | ✅ `configs/protocol.yaml` |
| R4 **P1 nominal** | ✅ trained, swept, V1 rendered — **F\* = 337 N** |
| R5 **P2 push** | ✅ trained, swept, V2 rendered — **F\* = 639 N** |
| R6 **P3 robust** | 🔄 **TRAINING NOW** (~9.5 h) |
| P2 extended sweep to 1000 N | 🔄 running |
| R8 Phase 3 (generalization, failure analysis, ablation) | ⬜ |
| R9 Phase 4 (plots, ONNX, README, portfolio repo) | ⬜ |

### Headline results so far

| policy | F\* overall | behind | front | right | left |
|---|---:|---:|---:|---:|---:|
| legacy (pre-existing) | 431 N | 514 | 462 | 404 | 388 |
| **P1** nominal | **337 N** | 397 | 387 | 312 | **267** |
| **P2** + pushes | **639 N** | 591 | 598 | 654 | **>700** |

**P1's weakest axis (lateral, 267 N) became P2's strongest (>700 N).** The
classic "lateral balance is harder for bipeds" result is partly a property of the
training distribution, not only of morphology.

**P2 generalizes past its training range:** trained to 346 N realisable, it
recovers 86% at 450 N and 82% at 500 N.

---

## 2. What is running right now

```bash
# P3 training (relaunched after the pseudo_inertia fix)
~/runs/p3robust.log          4096 envs, 6600 iters, ~4.75 s/iter
~/runs/p3.cooldown2.log      15 min cooling per 3 h, SIGSTOP/SIGCONT

# P2 extended-grid sweep (700-1000 N; P2's lateral F* is only a bound at 700)
~/runs/p2_sweep_hi.log
```

Check progress:
```bash
wsl -d Ubuntu -- bash -c "cd /mnt/d/humanoid/unitree_rl_mjlab; /home/sleem/venvs/mjlab/bin/python -u scripts/progress.py /home/sleem/runs/p3robust.log"
```

---

## 3. Exact commands (all verified this session)

**Never use a shell `&`, `$()`, or backslash escapes through PowerShell** —
see §6. Launch through `detach.sh`.

```bash
# Install launchers (Python, because the .sh installer cannot bootstrap itself)
wsl -d Ubuntu -- bash -c "cd /mnt/d/humanoid/unitree_rl_mjlab; /home/sleem/venvs/mjlab/bin/python scripts/install_launchers.py --all"

# Train
wsl -d Ubuntu -- /home/sleem/detach.sh /home/sleem/runs/pX.launch.log /home/sleem/run_train.sh <TAG> <TASK> 4096 6600 --agent.run-name <NAME>

# Cooldown scheduler (user preference: 15 min per 3 h)
wsl -d Ubuntu -- /home/sleem/detach.sh /home/sleem/runs/pX.cooldown.log /home/sleem/gpu_cooldown.sh <TAG> 10800 900 300

# Sweep (default grid 0-700 N x 4 directions x 64 envs, ~40 min)
wsl -d Ubuntu -- /home/sleem/detach.sh /home/sleem/runs/pX_sweep.log /home/sleem/run_sweep.sh <CKPT> <LABEL>

# Sweep with a different grid
wsl -d Ubuntu -- /home/sleem/detach.sh /home/sleem/runs/x.log /usr/bin/env FORCES=700,750,800,850,900,1000 /home/sleem/run_sweep.sh <CKPT> <LABEL>

# Analyse one sweep (per-direction F*, Wilson CIs, curve.png, summary.md)
python scripts/analyze_sweep.py results/push/<run_dir>

# Consolidate EVERYTHING into results/metrics/ (idempotent, re-run any time)
python scripts/collect_metrics.py

# Split-screen video; LABEL=ckpt[@force] gives per-panel force override
python scripts/make_split_video.py \
  --panels 'NOMINAL=checkpoints/p1_nominal/model_6599.pt,ROBUST=checkpoints/p3_robust/model_6599.pt' \
  --force-n 450 --direction-deg 270 --label v3_hero
```

Python is always `/home/sleem/venvs/mjlab/bin/python`, run from
`/mnt/d/humanoid/unitree_rl_mjlab`, with `MUJOCO_GL=egl WANDB_MODE=disabled`.

---

## 4. Checkpoints

| path | what | dims |
|---|---|---|
| `checkpoints/baseline_flat/model_6600_flat98.pt` | **legacy baseline, recovered** | 98/113 |
| `checkpoints/baseline_flat/model_6600.pt` | ⚠️ **MISLABELLED** — is the obstacle-expanded copy | 103/118 |
| `checkpoints/p1_nominal/model_6599.pt` | P1 | 98/113 |
| `checkpoints/p2_push/model_6599.pt` | P2 | 98/113 |
| `checkpoints/p3_robust/` | **create when P3 finishes** | 98/113 |

Freeze P3 when done:
```bash
cp logs/rsl_rl/g1_velocity/*p3_robust*/model_6599.pt checkpoints/p3_robust/model_6599.pt
python scripts/ckpt_dims.py        # verify 98/113 before trusting it
```

---

## 5. New code this project

| file | role |
|---|---|
| `src/tasks/velocity/mdp/disturbance.py` | `deterministic_push` — exact N / azimuth / instant, once per episode |
| `src/tasks/velocity/mdp/recovery_metrics.py` | `RecoveryTracker` — latched per-env outcome |
| `src/tasks/velocity/config/g1/robust_env_cfg.py` | the three ablation arms |
| `scripts/eval_push.py` | scenario + sweep harness (imports `eval_gate.py`, does not fork it) |
| `scripts/analyze_sweep.py` | per-direction F\*, Wilson CIs, plots |
| `scripts/collect_metrics.py` | consolidates everything into `results/metrics/` |
| `scripts/make_split_video.py` | split-screen video, PIL captions |
| `scripts/r1_verify_variants.py` | proves the arms differ ONLY in `events` |
| `scripts/dr_cost_probe.py` | per-DR-term VRAM and step cost (use `--only N`) |
| `scripts/ckpt_dims.py` | checkpoint obs-width audit |
| `scripts/install_launchers.py` | CRLF-safe launcher installer |
| `scripts/run_sweep.sh` | sweep launcher (`FORCES`/`DIRECTIONS`/`NUM_ENVS` overridable) |

---

## 6. Gotchas added this project (19–22) — read before touching anything

19. **PowerShell deletes backslash escapes passed to native commands.**
    `tr -d "\r"` arrived as `tr -d "r"`, silently stripping every letter `r`.
    Silent corruption, not a parse error.
20. **Git Bash mangles absolute WSL paths** — `/mnt/d/...` became
    `C:/Program Files/Git/mnt/d/...`. Invoke `wsl` from PowerShell.
21. **Gotcha 14 re-confirmed:** `... & echo LAUNCHED` through PowerShell ran only
    the `echo` and returned exit 0. The job never started; the only evidence was
    a missing log file. Use `detach.sh`.
22. **`install_launchers.sh` cannot bootstrap itself** (CRLF in its own shebang).
    Use `scripts/install_launchers.py`.

Also: **PowerShell expands `$(...)` and `$VAR`** inside strings passed to `bash -c`.
Avoid both; put anything non-trivial in a file.

---

## 7. The four real bugs found (all in `ROBUSTNESS_LOG.md` with full detail)

1. **`checkpoints/baseline_flat/model_6600.pt` was the wrong file** — 103/118, not
   the documented 98/113. Original recovered from its training run directory.
2. **The recovery metric silently dropped every failure.** The push event clears
   per-env state inside `env.step`, before the tracker looks — so fallen episodes
   erased their own evidence. Recovery rates of 1.000 computed over n=2. Fixed by
   committing at the moment of the fall. **Always report `n` next to a rate.**
3. **Push delivery had to be proven, not assumed.** `peak_dv` (root speed change
   within 0.3 s) is logged on every episode and must scale with commanded force.
   It does: 0.16 / 0.41 / 1.57 / 2.96 / 4.00 m/s at 0 / 100 / 300 / 600 / 1000 N.
4. **`dr.pseudo_inertia` costs a fixed ~4 GB** (its `set_const` recomputation),
   nearly independent of env count — 5649 MB at 1024, 5914 MB at 2048, saturating
   6 GB at 4096 → **43 s/iter, a 9× slowdown**. Replaced with `dr.body_mass`
   (2502 MB, free). *The diagnostic probe initially blamed the wrong term because
   it leaked warp allocations between variants — use `--only N`.*

---

## 8. Next steps, in order

1. **When P3 finishes:** freeze checkpoint → verify dims → run sweep → run
   extended sweep (700–1000 N) → `collect_metrics.py` → render **V3**
   (`P1 │ P2 │ P3`, the hero clip).
2. **R8.2 generalization gap** — the headline experiment. Evaluate all three
   against the *unseen* distribution frozen in `configs/protocol.yaml`
   (friction 0.35–1.6, mass ±25%, damping ±30%, consecutive pushes, 0.35 s
   duration). **No retraining.** Expect P2 to degrade on dynamics and P3 to hold;
   that dissociation is the most interesting available result.
3. **R8.3 failure analysis** — gait phase at push (the actor observes a `phase`
   term; `feet_ground_contact` has `track_air_time=True`). Save failure clips.
4. **R8.4 ablation table** — from `results/metrics/ablation.md`.
5. **R9** — plots, ONNX export + latency against `deploy/` (aarch64 ONNX Runtime
   is already vendored), README, portfolio repo.

### Phase 5 — agreed future extensions, in this order (after P3/V3)

Full detail in `ROBUSTNESS_PLAN.md` §6c. Both reuse the existing harness unchanged.
Connecting idea: **a carried load is a sustained external disturbance**, so these
extend the thesis rather than starting a new project.

1. **E1 — carry a payload while walking** (~1 day, ~6 h GPU). Mass welded to
   `torso_link` (E1a) then a box resting on the forearms (E1b). Sweep payload mass
   for `max_carryable_mass_kg` using the same 50%-crossing method as `F*`. Then
   cross-tabulate **force × payload** — a 2-D robustness surface. Directly tests
   whether P3's `body_mass` DR (a point mass at the COM, i.e. a payload) transfers.
2. **E2 — non-prehensile manipulation** (~4–6 days, ~20 h GPU). Push a box/cart
   with the forearms, lean on a door. **No grasping** — the G1 model has no hands
   or fingers, so prehensile work needs a new asset and is out of scope. Needs
   object state in observations and new reward terms; add **one term at a time**
   and reuse `preflight_g4.py`'s guard that refuses to start if a task term
   exceeds 3× the largest locomotion term.

E1 first: one day, reuses everything, and banks a complete result before E2's
reward-shaping risk.

### Open items

- **`deterministic_push` has no debug visualiser**, so evaluation clips have no
  in-sim force arrow (training's `apply_body_impulse` has `VizCfg`). Captions name
  the force instead. Worth implementing before V3.
- **Add an undisturbed-tracking-error column to the ablation table** — measured in
  the identical eval env at 0 N, it quantifies what robustness *costs*. P2's
  training-time `error_vel_xy` is 1.31 vs P1's 0.70, but that comparison is
  contaminated by P2 being pushed while measured.
- **`body_mass` vs `pseudo_inertia`** — record in the README that mass
  randomization models a point mass at the COM (payload uncertainty), not a
  density change, and say why.
