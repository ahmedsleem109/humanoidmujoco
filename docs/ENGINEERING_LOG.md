# Engineering Log — Robust Humanoid Locomotion

Findings, bugs, and the measurements that caught them. Kept from day one rather
than reconstructed at the end, because the reconstruction is always a story and
this is meant to be a record.

Convention: every entry states **what looked fine**, **what was actually wrong**,
and **the measurement that distinguished them**. If an entry cannot answer the
third, it is a guess and is labelled as one.

---

## Phase 1

### R0 — environment revalidation · PASS

`scripts/r0_audit.py`. 0 failures, 0 warnings.

- All four load-bearing pins hold: `torch 2.9.1+cu129`, `mjlab 1.2.0`,
  `mujoco 3.5.0`, `warp-lang 1.12.0`.
- `mjlab.envs.mdp.apply_body_impulse` exists, **including its `VizCfg`** — the
  in-sim force-arrow renderer. The disturbance generator and the video's force
  indicator are both library features.
- All 12 domain-randomization functions the brief's §14 asks for are present.

**Finding: the reward function needs no work.** §12 asks for energy and
joint-motion penalties; the config already carries `joint_acc_l2`,
`joint_pos_limits` and `action_rate_l2`, plus `foot_gait`, `foot_clearance`,
`foot_slip`, `soft_landing`, `self_collisions` — and a **`stand_still` penalty at
-1.0**, which is precisely the "don't let standing still be optimal" guard §12
worries about. 16 reward terms, none added by this project.

That matters beyond convenience: because no reward term changes between arms,
the ablation compares three policies optimising an *identical* objective.

**Finding: the actor observes a `phase` term** (2-D gait clock). §18 asks whether
failures concentrate in a gait phase; that is directly readable rather than
inferred.

---

### 🔴 R0.1 — the flat baseline in `checkpoints/` was not the flat baseline

**What looked fine.** `HANDOFF.md` §1 and §3 both name
`checkpoints/baseline_flat/model_6600.pt` as the frozen flat walking baseline,
and the file is present, correctly sized, and loads without error.

**What was actually wrong.** It is the *obstacle-expanded* checkpoint — actor
103, critic 118 — not the flat one. `HANDOFF.md` §10.2 states plainly that the
flat baseline is 98/113 and "cannot be loaded into [the obstacle env] at all",
so at some point during the obstacle work the staged copy was overwritten by its
widened twin. `model_6600.pt` and `model_6600_obstacle.pt` now have identical
shapes.

**The measurement.** `scripts/ckpt_dims.py` prints the first-layer width of every
checkpoint. Loading it into the flat env also fails loudly — but only because the
shapes happen to be incompatible. Had the clobbering gone the other way, or had
the widths matched, it would have loaded a *different policy than documented* and
every downstream number would have been attributed to the wrong model.

**Recovery.** The original survives in its training run directory,
`logs/rsl_rl/g1_velocity/2026-08-10_13-23-26/model_6600.pt` (98/113, iter 6600),
and is now staged as `checkpoints/baseline_flat/model_6600_flat98.pt`. Nothing
was overwritten — the mislabelled file is left in place, since deleting evidence
of the fault is not a fix.

**Generalisable lesson.** A checkpoint filename is a claim, not a fact. Shapes are
cheap to check and are checked from now on.

---

### R1 — new code

| file | role |
|---|---|
| `src/tasks/velocity/mdp/disturbance.py` | `deterministic_push` — exact magnitude / azimuth / instant, once per episode, identical across the batch |
| `src/tasks/velocity/mdp/recovery_metrics.py` | `RecoveryTracker` — latched per-env recovery outcome, 45°/15° tilt bounds |
| `src/tasks/velocity/config/g1/robust_env_cfg.py` | the three ablation arms |
| `scripts/eval_push.py` | scenario + sweep harness, importing `eval_gate.py` rather than forking it |
| `scripts/r1_verify_variants.py` | proves the arms differ only in `events` |

**Why a separate push term at all.** `apply_body_impulse` samples `force_range`
**per component**, so a nominal "50 N push" is really `U(-50,50)³`: corner-biased
magnitude, non-uniform direction. Excellent as a training distribution, unusable
as an evaluation axis — a force-vs-recovery curve built on it is a curve of
nothing. `deterministic_push` fixes magnitude, azimuth (in the robot's yaw frame)
and trigger instant.

**Ablation validity — verified, not assumed.** `r1_verify_variants.py` compares
reward weights, observation terms and corruption flags, command ranges,
`episode_length_s`, `decimation` and curriculum across all three arms:

```
OK  rewards          OK  observations     OK  commands
OK  episode_length_s: 20.0    OK  decimation: 4
OK  num reward terms: 16      OK  curriculum: []
```

Only `events` differs. All 12 design assertions pass.

**Three deliberate departures from the mjlab default**, each recorded so it is
never mistaken for an accident:

1. `push_robot` removed from all three arms — it teleports root velocity and
   cannot be expressed in Newtons, so a policy trained under it cannot be
   evaluated on a force axis without changing units mid-experiment.
2. Observation noise stays **on** everywhere — it models the sensor, not the
   dynamics; removing it from P1 would confound the ablation with an
   observability change.
3. The `command_vel` curriculum is popped — it doubles the command range at
   iteration 5000, making the command distribution non-stationary in the last
   quarter of the run. It cost the previous project ~40% of its reward when it
   fired.

---

### 🔴 R1.1 — the recovery metric silently discarded every failure

**What looked fine.** The first sweep of the legacy baseline returned
`recovery=1.000, fall=0.000` at every force from 0 to 300 N. Plausible: this
policy *was* trained with pushes, so high robustness is expected.

**The tell.** Episode counts. At 600 N the cell reported `n=2` out of 32
environments; at 1000 N it reported `n=0`. A cell that scores no episodes is not
a measurement.

**What was actually wrong.** `RecoveryTracker` scored an episode when its
recovery window closed or when `dones` fired. But `deterministic_push` clears its
own per-env state when `episode_length_buf` wraps — and that happens *inside*
`env.step`, i.e. **before** the tracker next runs. For any environment that fell
and reset, `st.fired` was already `False` by the time the tracker looked, so the
episode was never committed to `records`.

The bias is the worst possible direction: recovery rate is computed over *scored*
episodes, and the dropped episodes were **exactly the failures**. The metric
would have reported a policy as invincible precisely where it was collapsing.

**The fix.** Commit at the moment of the fall. Once fallen the outcome cannot
change, so there is nothing to wait for, and the ordering hazard disappears
entirely.

**Verification.** Same sweep, after the fix: `n=64` per cell at 600 N and 1000 N
(32 environments × 2 pushes, since a fallen env resets and is pushed again inside
the 7 s episode), `fall=1.000`.

**Generalisable lesson.** This is the same failure family as the previous
project's `Episode_Metrics` time-average bug: a metric that is *silently
incomplete* rather than visibly wrong. Sample counts belong in every report, next
to every rate. A rate without an `n` is not a result.

---

### 🔴 R1.2 — is the push real, or configured-but-inert?

**What looked fine.** `recovery=1.000` at 60 N. Could be a robust policy; could
equally be a force that is computed, logged, and never reaches the solver. Both
produce identical output.

**The measurement that distinguishes them.** `peak_dv` — the largest root-speed
deviation in the 0.3 s after the impulse. If the push is inert, this stays at the
gait-noise floor regardless of commanded force. If it is real, it scales with it.

| commanded | measured `peak_dv` |
|---:|---:|
| 0 N | 0.163 m/s *(gait noise floor)* |
| 100 N | 0.411 m/s |
| 300 N | 1.570 m/s |
| 600 N | 2.960 m/s |
| 1000 N | 3.999 m/s |

Monotone and near-linear in force. The disturbance is real, and `peak_dv` is now
recorded on every episode as a standing sanity channel.

**Generalisable lesson.** For any injected quantity, log a *response* channel
that must move when the input moves. "It ran without error" is not evidence that
it did anything.

---

### 🔴 R2 — the kill-switch · **PASS**

The question: *does a nominally-trained G1 policy actually fall under a realistic
push?* If not, there is no project, and the honest time to discover that is
before spending 28 h of GPU.

Swept the recovered legacy baseline (`model_6600_flat98.pt`): 15 magnitudes ×
4 directions × 64 environments, ~40 min. **5,300 scored pushes.**

| force (N) | recovery | n |
|---:|---:|---:|
| 0–300 | 1.000 | 256 each |
| 350 | 0.894 | 263 |
| 400 | 0.683 | 278 |
| 450 | 0.391 | 317 |
| 500 | 0.191 | 392 |
| 550 | 0.070 | 443 |
| 600 | 0.010 | 481 |
| 650–700 | 0.000 | 504, 511 |

**F\* = 431 N.** A clean, well-resolved sigmoid: flat at 1.0 through 300 N, a
sharp transition across 350–600 N, saturated at 0 by 650 N. Exactly the
"almost always recover → occasional failure → almost always fail" shape the
brief's §19 asks for.

**F\* is strongly direction-dependent:**

| push direction | F\* (N) |
|---|---:|
| from behind (shoved forward) | **514** |
| from the front (shoved backward) | 462 |
| from its right (shoved left) | 404 |
| from its left (shoved right) | **388** |

Lateral pushes are **~25% harder to recover from than sagittal** ones. That is
the classic bipedal result — the stance is narrow in the frontal plane, so
lateral balance has far less base of support and recovery requires a genuine
cross-step rather than a longer stride. It emerged from the data rather than
being looked for, and it is the first real answer to §18's "when does it fail?".

The left/right asymmetry (388 vs 404 N) is small; whether it is a gait-phase
artefact or noise is not yet established and is **not** claimed as a finding.

**Interpretation — F\* is high, and that is expected.** 431 N sits above the
published 50–300 N benchmark band. The reason is §1.5: this policy was trained
*with* `push_robot` and partial domain randomization, so it is already a
partially robust policy, not a naive one. P1 — trained with pushes and DR
genuinely removed — should sit well below it. That gap is the project.

It also confirms the training force range is well placed: the maximum realisable
training magnitude (346 N) sits just below the legacy F\*, straddling the
interesting region rather than being uniformly trivial or uniformly fatal.

**Note on `n`.** Sample counts rise with force because a fallen environment
resets and is pushed again inside the same 7 s episode. Rates are unaffected —
each push is scored independently — but confidence intervals tighten at high
force. Reported via Wilson score intervals, which unlike the normal
approximation remain correct at rates of exactly 0 and 1.

---

### Tooling gotchas added this session

19. **PowerShell deletes backslash escapes passed to native commands.**
    `bash -c 'tr -d "\r" < x > y'` arrived as `tr -d "r"`, which stripped every
    letter `r` from the script — producing `fo` for `for` and `set -` for
    `set -u`. Related to gotcha 18, but the failure is *silent corruption* rather
    than a parse error. Never pass escapes through PowerShell; use a file.
20. **Git Bash mangles absolute WSL paths** (`/mnt/d/...` became
    `C:/Program Files/Git/mnt/d/...`) via MSYS path conversion. Invoke `wsl`
    from PowerShell, or set `MSYS_NO_PATHCONV=1`.
21. **Gotcha 14 re-confirmed the hard way.** `... > log 2>&1 & echo LAUNCHED`
    through PowerShell ran only the `echo` and reported exit 0. The job never
    started, and the *only* evidence was a missing log file — the exit status was
    clean. Background work through the harness, never with a shell `&`.
22. **`install_launchers.sh` cannot bootstrap itself.** It lives on the Windows
    filesystem with CRLF endings, so `/usr/bin/env bash\r` fails before it can
    strip anything, and the inline workaround is destroyed by gotcha 19.
    Replaced with `scripts/install_launchers.py` — Python has neither shell
    quoting nor escape mangling.

---

## Phase 2

### R4 — P1 nominal, launched

`Unitree-G1-Robust-P1-Nominal`, 4096 envs, 6600 iterations, run name
`p1_nominal`, seed 0. ~5.06 s/iteration → **9.3 h**. Cooldown scheduler (15 min
per 3 h, SIGSTOP/SIGCONT) and the 3-hour metrics watcher are attached under the
same tag.

**Smoke-tested all three arms at 3 iterations before committing 9.3 h**, which
caught two real config errors that would otherwise have surfaced hours in:

- `dr.pd_gains` takes `kp_range` / `kd_range`, not a single `ranges` tuple —
  a hard `TypeError` at env construction.
- `dr.body_mass` emits a warning that it randomizes mass while leaving the
  inertia tensor unchanged. A body 15% heavier with unchanged inertia is not a
  physically realisable robot, and a policy can learn to exploit the
  inconsistency. Switched to `dr.pseudo_inertia`, which perturbs mass, COM and
  inertia jointly via a Cholesky factorisation of the pseudo-inertia matrix and
  stays positive-definite for any magnitude. Mass scales as `exp(2*alpha)`, so
  `alpha_range=(-0.07, 0.07)` gives ~±15% mass *with consistent inertia*.

The second one matters beyond correctness: unphysical randomization teaches
robustness to something the real robot cannot do, which is the opposite of the
project's goal.

**P1 result — converged, and better than the previous project's baseline.**

| | P1 final | old flat baseline |
|---|---:|---:|
| reward | **36.98** | 26.11 |
| episode length | **1000.0 / 1000** | 997.75 |
| fall rate (undisturbed) | **0.0%** | ~4.2% |
| `error_vel_xy` | 0.69 | 0.89 |
| `mean_action_acc` | 0.694 | 0.82 |

**The plateau heuristic was wrong, and following it would have biased the
ablation.** `progress.py` reported "PLATEAUED — further training buys little" at
iteration 2004 (reward 28.8). Reward then climbed to **36.98** by 6600, a **+28%**
gain past the point the heuristic said to stop. Freezing P1 there would have
handicapped the nominal arm and inflated whatever P2/P3 appeared to gain.

Generalisable lesson: a plateau heuristic tuned on one project's reward scale is
a hypothesis, not a stopping rule. The identical-budget constraint of an ablation
is a better rule, and it is the one that was followed.

---

### 🔴 R4.1 — kill-switch #2 · **PASS**

P1 swept on the grid used for the legacy baseline. **5,155 scored pushes.**

**F\* = 337 N**, against the legacy policy's 431 N — a **22% drop**.

| push direction | P1 | legacy | Δ |
|---|---:|---:|---:|
| from behind | 397 | 514 | −23% |
| from the front | 387 | 462 | −16% |
| from its right | 312 | 404 | −23% |
| **from its left** | **267** | 388 | **−31%** |
| overall | **337** | 431 | −22% |

At 300 N the legacy policy recovered **100%** of the time in every direction;
P1 recovers **31%** laterally. P1's weakest axis, 267 N, is **inside the published
50–300 N benchmark band** — it fails at forces the literature treats as routine.

The lateral penalty seen on the legacy policy is *amplified* in P1: removing push
training costs 31% laterally but only 16% in the sagittal plane. Frontal-plane
balance is both the harder problem and the one that benefits most from training
on disturbances. That is a real, quantified finding and it came out of the
control arm, before the intervention was even trained.

---

### V1 — the problem statement · rendered

`results/videos/v1_problem_20260816-194755/`. Same policy, same seed, same
command, same camera; only the push differs.

**A cherry-picking trap, avoided.** The first cut used 350 N — and the episode
*recovered*. At 350 N lateral, P1's measured recovery rate is 0.100, so the video
seed simply drew one of the 10%. The tempting fix is to try seeds until one
falls. That is cherry-picking, and §27 forbids exactly this.

The honest fix is to choose the force from the measured curve and **state the
measured probability in the caption**: at 450 N lateral, P1 recovers 1.6% of the
time, so a single clip is representative rather than selected. The caption reads
*"Walks 1000/1000 steps. Recovers from this push 1.6% of the time."*

**Outstanding polish.** `deterministic_push` has no debug visualiser, so the
in-sim force arrow that `apply_body_impulse.VizCfg` provides for training is
absent during evaluation. The burned-in caption names the force instead. Worth
implementing before V3, which is the hero clip.

**Camera.** mjlab's default (distance 5.0, elevation −45°) is a distant survey
shot that wastes the top third of frame on sky. Changed to distance 3.2,
elevation −12°, azimuth 120° — a 3/4 view that shows forward travel and lateral
toppling simultaneously.

---

### R5 — P2, and the result that reframes the project

**F\* = 639 N**, against P1's 337 N — **+90%**.

| direction | P1 | P2 | Δ |
|---|---:|---:|---:|
| from behind | 397 | 591 | +49% |
| from the front | 387 | 598 | +55% |
| from its right | 312 | 654 | +110% |
| **from its left** | **267** | **>700** *(not reached)* | **>+162%** |

**P1's weakest axis became P2's strongest.** At R2 the lateral penalty looked
like the textbook bipedal result — narrow frontal-plane support, less base to
recover over. P2 inverts it: at 700 N it still recovers 55% of lateral pushes.

The conclusion is not that lateral balance is easy. It is that **the frontal
plane is the axis with the most headroom left when you do not train for it.**
The classic finding measures untrained policies and attributes to morphology
what is partly a property of the training distribution.

**Generalization is visible before the dedicated test.** P2's training maximum is
346 N realisable, yet it recovers 86% at 450 N and 82% at 500 N — 44% beyond
anything it experienced.

**Methodological problem: the grid saturates.** It was sized against P1 and the
legacy policy (both F\* < 450 N). P2's lateral axis never crosses 0.5 inside it,
so ">700 N" is a bound, not a measurement. Extended to 1000 N and re-run so all
arms sit on one non-saturating axis. Reporting a bound in an ablation table when
a real number is obtainable would be sloppy.

---

### 🔴 R6.1 — P3 ran 9× slower, and the probe that found it had the same bug

**What looked fine.** P3 launched cleanly. All seven DR terms registered, the
impulse registered, training started.

**What was wrong.** 43 s/iteration against P1/P2's 4.7 s — from iteration 0, so
not contention. At that rate 6600 iterations is **79 hours** instead of 10.

**First measurement, and its defect.** `scripts/dr_cost_probe.py` adds DR terms
cumulatively and times fixed steps. It showed VRAM climbing 2470 → 3708 → 4982 →
6144 MB and appeared to indict `joint_damping`. That conclusion was **wrong**:
warp/mujoco allocations are not released by `env.close()` +
`torch.cuda.empty_cache()`, so each row reported a partly *cumulative* footprint.

The probe built to diagnose a memory problem was itself leaking memory. Adding
`--only N` to run a single variant in a fresh process gave the real numbers:

| config (4096 envs, fresh process) | VRAM | ms/step |
|---|---:|---:|
| P2 | 2470 MB | 179.0 |
| **P3 without `pseudo_inertia`** | **2502 MB** | **178.2** |
| P3 with `pseudo_inertia` | saturates 6 GB | thrashes |

**The actual culprit is `dr.pseudo_inertia`, and it does not scale with env count:**

| envs | P3 config VRAM |
|---:|---:|
| 1024 | 5649 MB |
| 2048 | 5914 MB |
| 4096 | saturated |

A **fixed ~4 GB overhead** from its `set_const` recomputation. Every other DR
term is free — the full set without it costs 32 MB more than P2.

**The fix, and what it costs.** Dropping to a smaller env count for P3 alone
would have broken the ablation's identical-budget requirement and handicapped
precisely the arm under test. So `pseudo_inertia` was replaced by `dr.body_mass`
(2502 MB, 181 ms/step — free), and P3 runs at the full 4096 envs, 4.7 s/iter, no
retraining of P1 or P2 required.

**Honest accounting of what was lost.** `body_mass` scales mass without scaling
inertia, so it does *not* model "this link is denser than modelled". What it
models *exactly* is a **point mass at the COM** — an unmodelled payload, an added
battery, mounted equipment. That is a real and relevant source of mass
uncertainty for a humanoid, so the term is physically meaningful, just narrower
than intended. This partially reverses the reasoning recorded at R4: the earlier
entry called `body_mass` unphysical, which is too strong. It is exact for one
scenario and wrong for another.

**Generalisable lesson, twice over.** A diagnostic tool is code and can carry the
same class of bug as the thing it measures — the cumulative sweep would have led
to blaming `joint_damping` and deleting the wrong term. And "measure it" only
helps if the measurement is isolated: one variant, one fresh process.

---

### R6.2 — P3 result: **domain randomization added almost nothing**

**F\* = 654 N against P2's 639 N — +2.4%.** For comparison, P1 → P2 was **+90%**.

| policy | overall | behind | front | right | left |
|---|---:|---:|---:|---:|---:|
| P1 nominal | 337 | 397 | 387 | 312 | 267 |
| P2 + pushes | 639 | 591 | 598 | 654 | 711 |
| **P3 + pushes + DR** | **654** | 628 | 594 | 668 | 750 |
| legacy | 431 | 514 | 462 | 404 | 388 |

Per direction P3 gains +6.3% (behind), **−0.7% (front)**, +2.2% (right), +5.5%
(left). One direction is *worse*. The honest reading is that on the force axis,
**disturbance randomization does essentially all of the work and domain
randomization adds noise-level improvement.**

This reproduces the contested Xie et al. finding — that dynamics randomization is
not always necessary — on a humanoid, with a controlled three-arm ablation.

**What this does NOT yet show.** P2 has never seen a different friction, mass or
damping. The unseen-*dynamics* axis is untested and is where P3 should win if it
wins anywhere. That is R8.2 and it is now the decisive experiment: the project's
conclusion hinges on whether DR buys *transfer* rather than *force robustness*.

**Mid-training check paid for itself.** A reduced sweep of P3 at iteration 3600
(55% trained) gave a 2-direction F\* of ~580 N against P2's ~655 N, and showed a
large sagittal deficit (0.153 vs 0.463 at 600 N). Both closed by 6600. Stopping
early — which was considered — would have reported a spurious "DR hurts the
sagittal plane" finding that was pure undertraining.

---

### 🔴 R9.1 — the video that showed the opposite of what happened

**What looked fine.** V3 rendered cleanly, three panels, correct labels, correct
force caption. Every panel showed a robot walking.

**What was wrong.** At 550 N lateral, P1 falls **128 out of 128 times** in the
sweep. A clip showing it walking contradicts the measurement.

**Two compounding causes, found by instrumenting rather than guessing:**

1. **Episode termination inside the clip.** The robot fell at ~t=3.2 s, the
   `fell_over` termination fired, the episode **reset**, and the robot stood up
   and walked again. A frame taken at t=8 s shows a healthy walking robot — which
   reads as "it shrugged off a 550 N push" when the truth is "it fell and the
   simulation restarted". **This is the most misleading artifact the pipeline
   could produce, and it nearly shipped.** Fixed by popping `fell_over` from the
   *video* env only, so a downed robot stays down. Metrics are unaffected: they
   come from `eval_push.py` and never from video.

2. **The verification I added was itself wrong.** It read `push_state.fired` at
   the end of the rollout — but the push event clears its own state on episode
   reset, so it reported `fired=False` *while simultaneously reporting
   `max_tilt=61.8°`*. Two outputs of the same check contradicting each other is
   what exposed cause 1. Fixed by latching `fired` across the rollout.

**Also fixed: single clips at probabilistic forces are not reproducible.** An
earlier V3 at 450 N showed P1 recovering, while V2 at the identical 450 N and
seed showed it falling. MuJoCo-Warp on GPU is not bit-deterministic across runs,
and at 450 N lateral P1 recovers 1.6% of the time — the two renders drew
different outcomes. Re-rendering until the desired outcome appears is
cherry-picking. The fix is to choose a condition where the outcome is not a
coin flip: at **550 N lateral P1 fell 128/128**, so any single clip is
representative.

**Generalisable lesson.** A video is a measurement and must be verified like one.
`make_split_video.py` now refuses to emit a clip whose push did not fire, and
prints the scenario, fired flag, magnitude and max tilt per panel.
