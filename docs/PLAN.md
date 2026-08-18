# Robust Humanoid Locomotion Under Unexpected Disturbances — Execution Plan

**Source brief:** `D:\humanoid\neworientation.md`
**Written:** 2026-08-16
**Predecessor:** `plan.md` + `HANDOFF.md` (G1 obstacle traversal). That project is
**closed and retained in-directory**. Nothing in it is deleted; several parts are reused
verbatim (§0.2).

---

## 0. What this plan assumes, and what it reuses

### 0.1 The headline reuse claim

This is **not** a from-scratch project. Roughly 60% of the engineering the brief
describes already exists in this repo and is battle-tested:

| Brief section | Status | Where |
|---|---|---|
| §8 MuJoCo env + PPO pipeline | ✅ **done** | mjlab 1.2.0, WSL2/CUDA env, pinned deps |
| §9 G1 robot | ✅ **done** | `src/assets/robots/unitree_g1/` |
| §10 velocity-command locomotion task | ✅ **done** | `src/tasks/velocity/velocity_env_cfg.py` |
| §11 proprioceptive obs (no force info) | ✅ **done, verified compliant** | §1.3 below |
| §12 reward design | ✅ **done** — already matches the brief term-for-term | §1.4 below |
| §13 disturbance generator | ✅ **library feature** — `mdp.apply_body_impulse` | §1.1 below |
| §14 domain randomization | ✅ **library feature** — `mjlab.envs.mdp.dr.*` | §1.2 below |
| §15 M1 environment validation | ✅ **done** | `preflight_gpu.py`, `preflight_g4.py` |
| §15 M2 learn to walk | ✅ **done once** — `checkpoints/baseline_flat/model_6600.pt` | but see §1.5 ⚠️ |
| §17 quantitative eval harness | ✅ **80% done** — `scripts/eval_gate.py` | needs a push-aware pass |
| §21 repo / §22 README | ⬜ new | Phase 4 |
| Video rendering + ffmpeg + clip assembly | ✅ **done** | `make_clips.py`, `build_clips.sh` |
| Unattended training supervision, GPU cooldown, detach | ✅ **done** | `cycle_supervisor.sh`, `gpu_cooldown.sh`, `detach.sh` |

**What is genuinely new:** a deterministic evaluation push, a recovery metric, the
disturbance sweep, the generalization protocol, the failure-phase analysis, the
three-panel video, and the portfolio repo. That is Phases 1, 3 and 4. Phase 2 is
compute, not code.

### 0.2 Carried over from the obstacle project (do not rebuild)

`eval_gate.py` (and its three fixed bugs — infinite play episodes, unbounded
`max_steps`, diluted `extras["log"]` sampling), `run_gate.sh`, `gate_bg.sh`,
`detach.sh`, `install_launchers.sh`, `cycle_supervisor.sh`, `gpu_cooldown.sh`,
`watch_run.sh`, `progress.py`, `preflight_gpu.py`, `pyrun.sh`, `make_clips.py`,
`build_clips.sh`, the gate-YAML + `GATES.md` ledger convention, and **all 18 gotchas
in `HANDOFF.md` §5 / §10.6 / §11.8**. Those gotchas are the most valuable artifact of
the previous project; every one of them still applies.

### 0.3 Hardware and budget (unchanged)

RTX 3060 Laptop 6 GB, WSL2 Ubuntu 24.04, `~/venvs/mjlab/bin/python`.
Flat task measured at **4096 envs, 3166 MB (52% VRAM), 4.7 s/iter**.
→ **one 6,600-iteration policy ≈ 8.6 h**, ≈ 9.3 h with the 15-min-per-3-h cooldown.

---

## 1. Analysis — six findings that shape the plan

### 1.1 The disturbance generator already exists, in the right units

`mjlab/envs/mdp/events.py::apply_body_impulse` is a stateful `mode="step"` event that:

- writes a real force to `xfrc_applied` in **Newtons** (not a velocity teleport),
- samples `force_range`, `torque_range`, `duration_s`, `cooldown_s`,
- runs an **independent timer per environment**, so impulses are decorrelated across
  the 4096-env batch,
- supports `body_point_offset` — applying the force above the CoM produces
  `cross(offset, force)` torque, i.e. a push that *tips* rather than only translates,
- ships a **built-in force-arrow visualiser** (`VizCfg`: rgba, `scale` m/N, `width`,
  `min_force`).

That last point is worth stating plainly: the brief's §7 "← EXTERNAL DISTURBANCE"
force indicator is a library feature, not a video-editing task. The arrow is rendered
in-sim, in the correct direction, scaled to the actual force.

**One real limitation.** `force_range` is sampled **per component**, so a nominal
"50 N push" is actually `U(-50,50)³` — magnitude is χ-like with a corner bias and
direction is non-uniform on the sphere. That is *fine for training* (it is just a
disturbance distribution) but **unusable for evaluation**, where §19 demands an exact
magnitude sweep and §5 demands an identical push across three policies. Hence R1.2.

### 1.2 The domain-randomization surface is complete

`mjlab.envs.mdp.dr` provides, against the brief's §14 checklist:

| Brief asks for | mjlab function |
|---|---|
| Ground friction, contact params | `dr.geom_friction` ✅ *(already active)* |
| Link masses | `dr.body_mass` ✅ |
| CoM offset | `dr.body_com_offset` ✅ *(already active)* |
| Joint damping / friction | `dr.joint_damping`, `dr.joint_friction` ✅ |
| Joint armature / stiffness / limits | `dr.joint_armature`, `dr.joint_stiffness`, `dr.joint_limits` ✅ |
| Actuator parameters | `dr.pd_gains`, `dr.effort_limits`, `dr.sync_actuator_delays` ✅ |
| Sensor noise | `dr.encoder_bias` ✅ + per-observation `UniformNoiseCfg` ✅ |
| Initial conditions | `reset_root_state_uniform`, `reset_joints_by_offset` ✅ |
| Full inertia perturbation | `dr.pseudo_inertia` ✅ *(bonus — physically consistent)* |

Nothing needs to be written. Phase 1 only chooses ranges and freezes them.

### 1.3 The observation spec already complies with §11

`velocity_env_cfg.py` actor group: base ang vel, projected gravity, twist command,
joint pos, joint vel, previous actions — each with `UniformNoiseCfg`, and
`enable_corruption=True` for the actor, `False` for the critic.

**No force, force direction, or disturbance timing is observable.** The brief's §11
prohibition is satisfied by construction. The policy is asymmetric-actor-critic, which
is the standard and correct setup here — but note the critic *is* privileged, so the
README must say "proprioceptive **actor**", not "proprioceptive policy", to stay honest.

### 1.4 The reward already matches §12 — do not redesign it

| §12 asks for | existing term |
|---|---|
| Forward velocity + command tracking | `track_linear_velocity`, `track_angular_velocity` |
| Upright stability | `body_orientation_l2`, `body_ang_vel`, `angular_momentum` |
| Healthy foot contacts | `foot_clearance`, `foot_slip`, `feet_ground_contact` sensor |
| Excessive energy / joint motion | action-rate & joint-limit penalties (verify at R0) |
| Falling | `is_terminated`, weight **-200** |
| "Reward task completion, not survival" | `pose` uses `variable_posture` with distinct standing / walking / running std sets — standing still is *penalised* relative to tracking |

The previous project's single hardest lesson (`HANDOFF.md` §10.10) was that adding a
reward term destroyed a working gait. **This project adds none.** The intervention is
in the *event* manager (pushes + DR), not the reward function. That is a deliberate
design choice and it is what makes the ablation clean: all three policies optimise the
*identical* objective, so any difference is attributable to the training distribution
alone.

### 1.5 ⚠️ The existing baseline is NOT the brief's "Policy B"

This is the most important finding, and it costs compute.

`unitree_g1_rough_env_cfg` pops `push_robot` **only inside `if play:`**. During
*training* it is active. So `model_6600.pt` was trained with:

- `push_robot` — velocity impulses ±0.5 m/s (x,y), ±0.4 (z), ±0.52 rad/s (roll/pitch)
  every 5–6 s,
- `foot_friction` ∈ (0.3, 1.6) — a **5.3×** friction range,
- `encoder_bias` ±0.015 rad, `base_com` ±5 cm on all three axes,
- actor observation noise on every channel.

That is "PPO + weak velocity pushes + partial DR". It is **already a partially robust
policy**. Presenting it as the naive baseline would inflate the measured gain and is
exactly the fabrication §17 and §27 forbid.

**Consequence:** Policy B must be retrained with those events stripped. And since the
§20 ablation needs three policies trained under identical hyperparameters, seeds and
iteration counts, all three are fresh runs. That is the Phase-2 budget.

**The existing checkpoint is still valuable** — it is a fourth data point
("mjlab default recipe"), a convergence reference, and proof the pipeline reaches a
stable gait in ~6,600 iterations. Keep it, report it, do not use it as Policy B.

### 1.6 The 3-screen storyboard works — but the third panel must be P2, not "untrained"

§6/§7 wants all three robots **already walking** at t=0, then a shared push at t≈4 s.
An untrained policy emits near-random joint targets and collapses within ~0.5 s. It
cannot be walking at t=0, and it is on the ground *before* the push lands — so the push
does not explain its failure and the panel proves nothing. Delaying its clock to hide
this would break §5's "same disturbance timing" and is exactly the presentational trick
§27 warns against.

**Resolution (adopted).** Keep the brief's three-panel layout, but populate it with the
three *trained* policies:

```
┌──────────────┬──────────────┬──────────────┐
│   NOMINAL    │  + PUSHES    │ + PUSHES &DR │
│      P1      │      P2      │      P3      │
│     FALL     │   WOBBLE     │   RECOVER    │
└──────────────┴──────────────┴──────────────┘
```

This is strictly better than the original on every axis: all three *are* walking at
t=0 so the storyboard runs exactly as written; the comparison is the §20 ablation made
visual rather than a strawman; and there is no honesty compromise. Policy A (untrained)
remains a README table row, where it belongs as a sanity floor.

### 1.7 Precedent — this has been done, and there is a real open question

The user's requirement was to confirm each training regime has prior art. It does, and
the protocol numbers below are lifted from it rather than invented:

- **RecoverFormer** (G1-29dof, PPO, MuJoCo) evaluates torso impulses of **50–300 N over
  0.1 s across 8 directions**, scoring **Recovery Success Rate** with a **45° tilt**
  bound and a 10 s return-to-stable window. → This plan adopts that protocol shape
  directly (R1.3), which makes our numbers comparable to a published baseline.
- **Variable Stiffness for Robust Locomotion through RL** uses the same 50–300 N / 0.1 s
  trunk-push benchmark. → Confirms the range is standard, not arbitrary.
- **Hierarchical MPC push-recovery** (Zhang et al.) reports *maximum recoverable
  impulse* in 8 directions — the exact §19 curve, with a %-of-body-weight
  normalisation we should copy so the number is transferable across robots.
- **H2-COMPACT** randomises ground contact, base mass and applies external pushes,
  then deploys to a **physical** G1-23dof. → Confirms this DR recipe survives sim-to-real,
  which lets the README claim relevance without claiming sim-to-real results (§27).
- **Dynamics Randomization Revisited** (Xie et al.) found dynamics randomization is
  **not always necessary** and can cost performance; other work finds disabling DR
  raises failure probability, with balance more DR-dependent than walking.

That last pair is the reason the §20 ablation is worth running: **the field disagrees.**
The project's positioning should say so — "we test a contested claim on the G1" is a far
stronger portfolio line than "we applied domain randomization".

---

## 2. PHASE 1 — Foundation, instrumentation, and the early kill-switch

**No policy training. Target: 2–3 working sessions. Cheap, and it can cancel Phase 2.**

### R0 — Environment revalidation *(~30 min)*

Re-run `preflight_gpu.py --num-envs 64` (disk, torch CUDA kernel, warp GPU kernel, real
G1 env on `cuda:0` — `torch.cuda.is_available()` alone is not sufficient, warp falls back
independently). Confirm the four load-bearing pins still hold:
`torch==2.9.1+cu129`, `mjlab==1.2.0`, `mujoco==3.5.0`, `warp-lang==1.12.0`.
Confirm `apply_body_impulse` is importable and exported from `mjlab.envs.mdp`.
Audit the reward dict for the §12 energy / joint-motion penalties (§1.4 open item).

**Exit:** preflight green, one training step at 4096 envs on GPU.

### R1 — New code (the only substantial coding phase)

**R1.1 — Three env configs** → `src/tasks/velocity/config/g1/robust_env_cfg.py`

Derived from `unitree_g1_flat_env_cfg` (flat, not rough — no `terrain_scan`, matching
the obstacle project's rationale and keeping the observation space small).

| task id | events | brief name |
|---|---|---|
| `Unitree-G1-Robust-P1-Nominal` | pushes **off**, DR **off** (friction fixed at 1.0, no encoder bias, no CoM offset), obs noise **kept** | Policy B / ablation P1 |
| `Unitree-G1-Robust-P2-Push` | `apply_body_impulse` **on**, DR off | ablation P2 |
| `Unitree-G1-Robust-P3-Robust` | `apply_body_impulse` **on**, full DR **on** | Policy C / ablation P3 |

Obs noise stays on in all three: it is part of the *sensing* model, not the
*dynamics* intervention, and removing it would confound the ablation with an
observability change. State this explicitly in the README.

`push_by_setting_velocity` is **popped in all three** and replaced by
`apply_body_impulse`, so training and evaluation speak the same unit (N). This is a
deliberate departure from the mjlab default — record it in `GATES.md`.

**R1.2 — Deterministic evaluation push** → `src/tasks/velocity/mdp/disturbance.py`

The single most important new file. `deterministic_push`, a `mode="step"` event taking
**exact** magnitude / direction / trigger-time / duration:

```
force_n        : float          exact magnitude in Newtons
direction_deg  : float          azimuth in the robot's heading frame (0=front, 90=left)
trigger_time_s : float          seconds after episode reset
duration_s     : float          impulse width (protocol default 0.1)
body           : "torso_link"
```

Every environment in the batch receives the *identical* push at the *identical* phase
of its own episode. This is what makes §5's "only the policy changes" true, and what
makes the §19 sweep a controlled experiment instead of a sampling exercise.

Also emits a per-env `push_applied` flag and the pre-push state snapshot the recovery
metric needs.

**R1.3 — Recovery metric** → `src/tasks/velocity/mdp/recovery_metrics.py`

Adopting the RecoverFormer protocol shape (§1.7) so the numbers are comparable:

- **Fall** — terminated, or torso tilt from vertical > **45°** (via projected gravity).
- **Recovered** — over a 3.0 s window starting at the push: never fell, **and** tilt
  returns below 15° and stays there for 0.5 s, **and** `error_vel_xy` returns to within
  1.5× its pre-push 1 s mean for 0.5 s continuously.
- **Recovery time** — push instant → first sample of that sustained-stable window.
- **Velocity loss** — integral of `error_vel_xy` above the pre-push mean over the window.

Latched **per environment**, using the `MetricsManager._step_values` latching pattern
already proven in `eval_gate.py`. **Gotcha 4 applies with full force:**
`Episode_Metrics/<name>` is a *time-average over the episode*, not a terminal outcome —
reading recovery rate from the training log would be off by roughly an order of
magnitude, exactly as `traversal_success` was (0.0117 logged vs 0.109 true).

**R1.4 — Push-aware evaluation harness** → `scripts/eval_push.py`

Wraps `eval_gate.py`'s proven `build_env_and_policy` / `load_policy` / provenance /
report / ledger machinery — **do not fork it, import it.** Adds: a scenario spec
(force, direction, time, seed, command), the recovery metrics pass, per-episode CSV
export for Phase 3, and deterministic seeding across policies.

**R1.5 — Split-screen video pipeline, built NOW not in Phase 4**
→ `scripts/make_split_video.py`

Moved forward from Phase 4 because a video is produced **after every training run**
(§3), not once at the end. Building it in Phase 1 means it is debugged against the
existing `model_6600.pt` before any new policy exists, and every later run gets its
video for free.

Responsibilities:

1. Render N policies separately, 1 env each, `MUJOCO_GL=egl`, under a byte-identical
   scenario (same seed, terrain, command, camera pose on `torso_link`, same
   `deterministic_push`). Only the checkpoint differs.
2. Frame-align the clips (all start at their own reset; the push lands at the same
   frame index by construction).
3. Composite with `hstack`, with a 4 px divider via `pad`.
4. Burn in captions.

⚠️ **Verified constraint: `drawtext` is NOT available.** The `imageio-ffmpeg` static
binary (`ffmpeg-linux-x86_64-v7.0.2`) is built without libfreetype — `-filters` lists
`hstack` and `overlay` but **no `drawtext`**. There is no system ffmpeg and no
passwordless sudo to install one.

**Caption approach (decided, not to be rediscovered later):** render caption text to
RGBA PNGs with PIL using `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`
(present), then composite with the `overlay` filter, timed via `enable='between(t,a,b)'`.
This is more work than `drawtext` but gives better typography control and is the only
route that works here.

Caption layers needed:
- **Persistent panel labels** — `NOMINAL` / `+ PUSHES` / `+ PUSHES & DR`, top of each panel.
- **Timed event captions** — the hook at 0–2 s, `← EXTERNAL DISTURBANCE  {F} N` at the
  push instant, the payoff line at the end.
- **Live outcome badge** — `FALL` / `RECOVERED` stamped per panel once the recovery
  metric resolves, driven by the metrics JSON so the label can never contradict the data.

The in-sim force arrow (`apply_body_impulse.VizCfg`, §1.1) supplies the physical
disturbance indicator — the caption only names it.

**Validate at R2** by producing a throwaway split-screen of `model_6600.pt` against
itself at two different push magnitudes. If the pipeline works there, it works for
every later run.

### R2 — 🔴 THE KILL-SWITCH: measure the failure baseline first *(~1 h)*

**Run this before committing 30 h of GPU time.**

The entire project rests on one empirical assumption: *a nominally-trained G1 policy
actually falls under a reasonable push.* If it does not, there is no story, no
divergence, and no video — and we would only discover that after Phase 2.

So: take the **existing** `model_6600.pt` and sweep it with `eval_push.py` over
0/10/20/…/150 N × 8 directions × 32 episodes. ~1 h.

This is brief §15-M3 executed on a checkpoint we already own, at effectively zero cost.

**Exit criteria:**
- There exists a force `F*` at which the existing policy's recovery rate falls below
  50%, and `F*` ≤ 300 N (within the published benchmark range).
- The curve is monotone-ish, not noise.
- The recovery metric agrees with human judgement on 20 hand-labelled clips
  (§7 risk: a mis-specified metric is the failure mode that killed Gate G4).
- `make_split_video.py` produces a captioned two-panel clip end to end (R1.5).

**This one hour answers "is the orientation good?" before any training spend.** It
yields, at zero training cost: the failure baseline (§15-M3), a validated recovery
metric, a validated video pipeline, and the force range for every later experiment. If
it comes back wrong, we re-scope having spent an hour rather than a week.

**If `F*` does not exist below 300 N:** stop and re-scope before spending Phase 2. The
likely response would be to raise the commanded velocity (harder to recover at 2 m/s),
push during single-support only, or push at the head rather than the torso. Better to
learn that in hour 1 than in hour 30.

**Bonus:** since `model_6600.pt` was trained *with* weak pushes and partial DR (§1.5),
its `F*` is an informative upper-ish bound. Policy P1's `F*` should be **lower**, and
if it is not, that itself is a finding worth reporting.

### R3 — Freeze the protocol *before* training *(~1 h, pure paperwork, high value)*

Write `configs/protocol.yaml` and commit it. It must fix, in advance:

- **Training distribution** — force range, duration, cooldown, DR ranges per parameter.
- **Evaluation distribution (in-distribution)** — the sweep grid.
- **Unseen distribution** — strictly outside the training support:
  friction ±40–50% vs training's ±30%, force magnitudes above the training max,
  8 directions offset 22.5° from any trained axis, push timing locked to specific gait
  phases, and **consecutive double pushes** (never trained).
- **Seeds** — one fixed seed set for training, a *disjoint* set for evaluation.

Committing this before any policy exists is what converts §17.5's generalization gap
from a post-hoc story into an actual pre-registered experiment. It is also the single
cheapest credibility gain in the whole project.

**Phase 1 exit gate:** R0–R3 green, `F*` established, protocol committed and hashed.

---

## 3. PHASE 2 — Train the three policies

**Pure compute. ~28–30 h wall clock. 3–4 unattended overnight sessions.**

All three runs use **identical** hyperparameters, seed, iteration count (6,600) and
env count (4,096). The *only* difference is the event dict. Anything else invalidates
the ablation.

| run | task | est. | product | **video shipped at end of run** |
|---|---|---|---|---|
| R4 | `Unitree-G1-Robust-P1-Nominal` | 9.3 h | Policy B, the failure baseline | **V1** — `P1 undisturbed │ P1 pushed` |
| R5 | `Unitree-G1-Robust-P2-Push` | 9.3 h | ablation middle term | **V2** — `P1 │ P2` under an identical push |
| R6 | `Unitree-G1-Robust-P3-Robust` | 9.3 h | Policy C, the hero | **V3** — `P1 │ P2 │ P3`, the hero clip |

### Videos are deliverables of Phase 2, not Phase 4

Each run ends with `make_split_video.py` (built at R1.5) producing a captioned
split-screen clip **before the next run starts**. Three reasons this is the right
structure and not just impatience:

- **Each video is a falsification test.** V1 must show P1 falling. If it doesn't, the
  premise is dead and we stop before R5 — a second early kill-switch, 9 h in rather
  than 28 h in.
- **Video review catches what metrics cannot.** The standing rule from the previous
  project: numeric gates cannot see visual pathologies. A policy that "recovers" by
  freezing in a crouch scores perfectly and looks terrible.
- **You accumulate showcase material continuously.** V1 alone ("this is the problem")
  is postable. If the project stalls at any point, there is already something to show.

| video | panels | caption arc |
|---|---|---|
| **V1** | 2 | `NOMINAL: undisturbed` │ `NOMINAL: pushed` → *"Learning to walk ≠ learning to recover."* |
| **V2** | 2 | `NOMINAL` │ `+ DISTURBANCE TRAINING` → *"Same push. Different training."* |
| **V3** | 3 | `NOMINAL` │ `+ PUSHES` │ `+ PUSHES & DR` → *"It didn't avoid the disturbance. It learned to recover from it."* |

V3 is the §6/§7 hero clip and the README header. V1 and V2 are the story beats — and
useful on their own, since a two-panel "here is the problem" clip is often more
compelling than the solution.

### Know within the first hour of each run, not the last

The per-run early-signal rule, distinct from the project-level R2 kill-switch:

- **Iteration ~100 (≈8 min):** mean episode length trending up, reward not NaN, no
  actuator saturation storm. Anything else = misconfiguration, kill immediately.
- **Iteration ~400 (≈30 min):** episode length ≥ 400/1000. The previous project's flat
  baseline is the reference curve — a new run materially below it at the same iteration
  is diverging, not "slow".
- **Cycle 1 (3 h):** a real latched evaluation, never a log scrape (gotcha 4 — the
  logged `Episode_Metrics/*` is a time-average and was off by ~9× at Gate G3).
- **Abort on:** two consecutive missed floors **or** a single >30% regression from the
  previous cycle. The regression rule exists because with a 9 h run and a 3 h cycle the
  consecutive-miss rule can barely fire — the exact defect that let Gate G4 attempt 1
  burn a full night.

**Train from scratch, not by fine-tuning.** Warm-starting P2/P3 from P1 would be ~40%
cheaper and is tempting, but P3 would inherit a gait already specialised to fixed
friction, and the ablation would measure "adaptation cost" rather than "what the
training distribution buys you". The previous project's §11 lesson was that a wrong
measurement is worse than an expensive one.

**Curriculum (brief §15-M4 "gradually increase difficulty").** P2/P3 ramp the force
range over the first ~2,000 iterations via the existing `mdp.reward_weight`-style ramp
pattern. ⚠️ `RewardManager.compute` skips zero-weight terms entirely, and the ramp keys
on `common_step_counter` which resets to 0 on every fresh env — both documented in
`HANDOFF.md` §10.11 and §11.4. The event manager does not share the reward manager's
zero-skip behaviour, but verify this at R0 rather than assume it.

**Infrastructure — reused unchanged:** `cycle_supervisor.sh` (SIGSTOP/SIGCONT pause →
15 min cool → evaluate → resume), `gpu_cooldown.sh`, `detach.sh`, `watch_run.sh` at the
3-hour reporting cadence, disk guard at 2 GB.

**Milestones must use the real latched metric, not the log.** The supervisor already
knows how to pause and run a real evaluation instead of scraping the log — that
capability was built for exactly this reason and transfers directly.

**Early-abort rule** (improved per §10.10's own critique): abort on two consecutive
missed floors **or** a >30% regression from the previous cycle. For P1 the milestone is
plain locomotion (episode length ≥ 900/1000 by cycle 2). For P2/P3 it is in-distribution
recovery rate.

**Phase 2 exit gate (R7):** all three policies walk — mean episode length ≥ 18 s,
fall rate ≤ 5% under **no** disturbance. A robust policy that cannot walk cleanly is a
failed run, not a trade-off. Video review of each, per the standing rule that numeric
gates cannot detect visual pathologies.

---

## 4. PHASE 3 — The experiments

**This is where the project stops being a tutorial. ~6–8 h of GPU, mostly evaluation.**

### R8.1 — Disturbance sweep (brief §19)

Every policy × 0→300 N in 20 N steps × 8 directions × 64 episodes, fixed seeds.
Deliverable: **recovery probability vs. force magnitude**, three curves, with
Wilson confidence intervals — the brief's "most valuable experiment".

Report the 50%-recovery force `F*` per policy, and per-direction (lateral pushes are
typically the hardest for bipeds — if that shows up, it is a real finding).
Normalise as **% of body weight × s of impulse** so the number is comparable to the
MPC literature.

### R8.2 — Generalization gap (brief §16, §17.5) — *the headline experiment*

Evaluate all three against the **unseen** distribution frozen at R3. **No retraining.**
Report in-distribution vs unseen recovery rate per policy, and the gap.

The interesting outcome is not "P3 wins" — it is *how much of P3's advantage survives
the distribution shift*. If P3's gap is smaller than P2's, randomization bought
generalization rather than memorization, and that is the paper-grade claim. If the gaps
are equal, that is the Xie et al. result reproduced on a humanoid, which is equally
publishable and must be reported as such.

### R8.3 — Failure analysis (brief §18)

Log at push instant: gait phase (single vs double support — from the existing
`feet_ground_contact` sensor with `track_air_time=True`, already in the config),
swing foot, commanded velocity, actual velocity, friction sample, tilt.

Cross-tabulate failures against each. Concretely answerable questions:
does failure concentrate in single-support? does it concentrate on pushes toward the
swing leg? does it concentrate at low friction? is actuator saturation present at
failure (`effort_limits` vs applied torque)?

**Save the failures.** Render 3–5 failure clips. §18 and §27 both insist on this, and
it is what separates an investigation from a demo reel.

### R8.4 — Ablation table (brief §20)

P1 / P2 / P3 (+ the legacy `model_6600.pt` as a fourth "mjlab default" row) across:
nominal walking success, in-distribution recovery, unseen recovery, `F*`, recovery time,
velocity loss.

Answers "which part of the training actually creates robustness?" — and against a
literature that disagrees with itself (§1.7).

**Phase 3 exit gate:** every number in the final README traceable to a committed
JSON/CSV under `results/`, produced by a reproducible command. No hand-typed values.

---

## 5. PHASE 4 — The artifacts

### R9.1 — Final cut of the videos (brief §6, §7, §23–26)

**V1–V3 already exist** — they were produced at the end of each Phase 2 run (§3). Phase 4
only re-cuts them with the final numbers, since by now the true recovery rates and `F*`
are known and the outcome badges can be exact.

Scenario spec (fixed once, in `configs/protocol.yaml`, used by all three):
same seed, same flat terrain, 1.0 m/s forward command, camera tracking `torso_link`,
push magnitude set near P1's measured `F*` so the divergence is maximal *and* honest —
a force chosen because it is the nominal policy's real failure point, not because it
looks dramatic. Only the checkpoint differs.

12–15 s, no intro card, no music, minimal labels. The in-sim force arrow (§1.1) is the
disturbance indicator — no post-production fakery.

⚠️ Never render video during training (gotcha 2 — CPU rasterisation pegged at 264% with
the GPU idle). Video is always a separate 1-env pass.

⚠️ `drawtext` is unavailable in this ffmpeg build (verified, §R1.5) — captions are PIL
PNGs composited with `overlay`.

### R9.2 — Plots

Recovery-vs-force curves (the hero plot), the generalization-gap bar chart, the
ablation bars, the failure-mode breakdown, and the training curves. One consistent
visual system across all five.

### R9.3 — Portfolio repo (brief §21, §22)

New public repo `robust-humanoid-locomotion`, structured exactly as §21 specifies.
**This is a clean repo, not the fork.** Rationale (and consistent with the existing
private-repo/public-fork split): development continues in
`D:\humanoid\unitree_rl_mjlab`; the portfolio repo is a curated export with the
protocol, configs, eval scripts, analysis, results and README — no dead ends, no
obstacle-project residue.

README order per §22: hero video → one-line method → **results first** → method →
failure analysis → reproduction last.

**Claims discipline (§27):** no sim-to-real claim, no fabricated numbers, failures shown
alongside successes, positioned as "an experimental study of robust humanoid locomotion
under external disturbances and simulation uncertainty" (§28).

---

## 6. Decisions — all resolved, none pending

The user directed that every recommendation be adopted. Recorded here so they are not
relitigated later.

| # | Decision | **Resolved** |
|---|---|---|
| **D1** | Video panel composition (§1.6) | **3 panels: P1 │ P2 │ P3.** Keeps the brief's layout, all three walk at t=0, and the panels *are* the ablation. The untrained policy stays a README table row. Supersedes the earlier 2-panel recommendation — this is strictly better. |
| **D2** | From scratch (~28 h) or warm-start P2/P3 (~17 h)? | **From scratch.** A contaminated ablation is the one thing that would sink the project's credibility, and the ablation is the project's whole value (§1.7). |
| **D3** | Fixed eval command speed or sweep it? | **Fix at 1.0 m/s** for the headline. Speed sweep is a Phase-3 extra if time allows. |
| **D4** | Portfolio repo public from day one? | **Published at the end.** Consistent with the established private-first policy. |
| **D5** | Push at torso CoM or offset above it? | **Torso CoM** for the headline sweep — matches the published protocol and removes a variable. `body_point_offset` becomes a Phase-3 extra. |
| **D6** | When are videos produced? | **After every training run**, not at the end (§3). Pipeline therefore built at R1.5, in Phase 1. |
| **D7** | How are captions burned in? | **PIL-rendered PNGs composited via `overlay`.** `drawtext` is absent from the available ffmpeg build — verified, not assumed. |

---

## 6b. Positioning for hiring — what actually moves a robotics team

**Stated goal: get the attention of robotics teams and be hired.** That changes what
"best result" means, so it is recorded here explicitly.

### The uncomfortable truth

A humanoid push-recovery video in MuJoCo is **table stakes**. A robotics lead sees it and
thinks "they can run mjlab" — which is true of every applicant. The video gets you three
seconds of attention. It does not get you an interview.

What gets a senior engineer to forward your profile is evidence of **judgement under
uncertainty**: that you can tell a broken metric from a broken policy, that you measure
before you optimise, and that you do not fool yourself. That evidence is *already in this
repo* and it is currently invisible.

### Your strongest asset is not the policy — it's `HANDOFF.md`

The previous project produced a series of findings that are exactly what hiring managers
probe for in interviews:

- **"Gate G4's 0.70 criterion was unreachable by construction"** — 62% of episodes never
  approached the obstacle, so the metric was capped at 0.38. *You diagnosed a broken
  experiment rather than blaming the policy.*
- **"`traversal_bonus` paid every timestep past the box, so lunging and falling
  outscored walking across"** — a structural reward-hacking diagnosis with the
  measurement to back it (19× the locomotion term).
- **"40% of failures were falling off the edge of the world, not failing at the
  obstacle"** — a `border_width=0.0` bug that would have taught the policy the wrong
  thing entirely, found by classifying *where* terminations happened.
- **"`Episode_Metrics` is a time-average, not a terminal outcome"** — off by ~9×;
  a silent error that would have made every downstream number plausible and wrong.
- **"`color_scheme='none'` repainted the obstacle grey on a grey floor"** — the human
  sign-off the gate demanded was impossible to perform, across 44 frames.

**Almost nobody writes this up.** Most portfolios show a working demo and hide the
process. A short, honest engineering log of five real bugs — each with the measurement
that found it and the fix — is more differentiating than a second push-recovery clip.

**Action:** the portfolio README gets a **"What broke and how I found it"** section, and
`ROBUSTNESS_PLAN.md` keeps the same discipline going forward. Deliberately falsifiable
claims, thresholds never quietly relaxed.

### Two additions that materially raise hiring signal

**A. Deployment readiness — ONNX export against the real G1 runtime.** *(~4 h, Phase 4)*

The repo already ships Unitree's **real-robot deployment stack**: `deploy/` contains a
C++ FSM, `unitree_articulation.h`, a joystick DSL, and **ONNX Runtime built for
`linux-aarch64`** — which is the G1's onboard compute architecture.

So without owning a G1 you can still:

- export the trained policy to ONNX,
- load it in the deploy harness and step it at the real control rate,
- measure **inference latency** on CPU and confirm it fits the control period,
- verify observation ordering and action scaling match the C++ side.

That converts *"I trained a policy in simulation"* into *"I produced a deployable
artifact for the robot's onboard runtime, and measured that it meets the control
deadline."* Those are different sentences to a hiring manager, and the second one is
rare in portfolio work. It stays fully within §27's rules — it is a **deployment
readiness** claim, never a sim-to-real *performance* claim.

**B. Cross-simulator validation — the honest proxy for sim-to-real.** *(~3 h, Phase 3)*

You cannot claim sim-to-real without a robot. But the *thesis* of this project —
robustness to unmodelled variation — has a testable proxy that costs almost nothing:

> Train under MuJoCo-Warp (mjlab, GPU batch). Evaluate under **plain MuJoCo** on CPU
> with a different solver configuration, timestep, and contact parameters.

Any policy overfitted to one physics backend degrades. `mujoco==3.5.0` is already a
dependency, and `deploy/` already needs a plain-MuJoCo path. Adding this as a third
column of the generalization table — *in-distribution / unseen disturbances /
different physics backend* — makes the robustness claim substantially harder to dismiss,
and it is the closest honest approximation of the sim-to-real gap available on a laptop.

### What this means for effort allocation

| | hiring signal | effort |
|---|---|---|
| The V3 hero video | attention only — gets 3 seconds | already budgeted |
| Ablation + generalization gap | **high** — shows you test contested claims | Phase 3 |
| Failure analysis + failure clips | **high** — shows you don't hide results | Phase 3 |
| "What broke and how I found it" | **highest, lowest cost** — already written, needs curating | ~3 h |
| ONNX + deploy-readiness + latency | **high** — few portfolios reach the robot's runtime | ~4 h |
| Cross-simulator validation | **high** — the honest sim-to-real proxy | ~3 h |
| Polishing video aesthetics | low | cap it |

**If time runs short, cut video polish and Phase 3 extras — never the ablation, the
failure analysis, or the engineering log.**

---

## 6c. PHASE 5 — Future extensions, in order (after P3 lands)

Agreed roadmap. Both items extend the disturbance thesis rather than starting a
new project, and both reuse the existing harness, launchers, metric pattern and
video pipeline unchanged.

**The connecting idea:** a carried load *is* a sustained external disturbance. A
5 kg box held at arm's length applies a continuous force and torque to the torso —
exactly what P2/P3 were trained to reject, but *persistent* rather than impulsive.
So this is a genuine generalization test of the trained capability, not a pivot:
**impulsive disturbance → sustained disturbance → actively applied disturbance.**

---

### E1 — Carry a payload while walking *(~1 day, ~6 h GPU)*

Attach a mass to the torso, or rest a box on the hands/forearms. Walk. Sweep the
payload mass and find the limit.

**New assets:** none beyond a body plus a weld or contact. The G1 model already
has full 7-DOF arms per side (`shoulder_pitch/roll/yaw`, `elbow`,
`wrist_roll/pitch/yaw`) ending at `wrist_yaw_link`.

**Two variants, in increasing difficulty:**

| variant | mechanism | what it tests |
|---|---|---|
| **E1a** rigid payload | `weld` a box to `torso_link` | pure mass/inertia disturbance — no balancing skill needed |
| **E1b** carried payload | box resting on forearms, contact only | mass *plus* the object can be dropped |

E1a is the cleaner experiment and should come first: it isolates the dynamics
change with no manipulation component. E1b adds a drop-failure mode, which is a
second metric and a much better video.

**Metrics — reuse `RecoveryTracker`, add:**
- `max_carryable_mass_kg` — the payload analogue of `F*`, same 50%-crossing method
- payload retention rate (E1b only: did the box stay on?)
- tracking-error degradation vs unloaded, per mass
- **cross-tabulate with push:** recovery rate at a given force *while loaded*.
  That two-dimensional surface (force × payload) is a far stronger artifact than
  either curve alone.

**The headline question:** does P3's domain randomization — which includes ±15%
mass as a **point mass at the COM**, i.e. exactly a payload — transfer to a payload
it never saw? This is the cleanest possible test of the `body_mass` term, and it
retroactively justifies the choice forced on us at R6.1.

**Expected additions:** ~1 day. `nconmax` needs re-measuring for E1b with
`g2_contact_probe.py`; VRAM headroom exists (P3 uses 2.6 GB of 6 GB).

---

### E2 — Non-prehensile manipulation *(~4–6 days, ~20 h GPU)*

Push a box or cart with the forearms; lean on a door. **No grasping** — the G1
model has **no hands and no fingers**, so prehensile manipulation would require a
new hand asset (Dex3 or a gripper) and is explicitly out of scope here.

**New work required:**
- object in the scene, with contact tuning (`nconmax` will need real headroom)
- object state in the observation group (relative position, orientation, velocity)
- reward: approach → contact → move-object-toward-goal
- termination on object loss / robot fall

**Where this will be hard, from the obstacle project's own lesson:** adding a
reward term destroyed a working gait once already (`traversal_bonus` ended up 19×
the locomotion term). Add **one** term at a time, measure every term's magnitude
before launching — `preflight_g4.py` already refuses to start if any task term
exceeds 3× the largest locomotion term, and that guard should be reused verbatim.

**Why it's worth it:** loco-manipulation is the most active area in humanoid
robotics right now, and this is the version of it that is reachable without hands,
without a bigger GPU, and without new hardware.

---

### Sequencing rationale

E1 before E2, deliberately:

1. E1 is one day and reuses everything; E2 is a week and needs new observations
   and rewards.
2. E1 produces a second measured curve (`mass` vs `recovery`) that pairs with the
   force curve using **identical methodology** — two robustness axes, one method.
3. E1 answers a question the current project already raised, so it closes a loop
   rather than opening a new one.
4. If E2 stalls on reward shaping — the likely failure mode — E1 has already
   banked a complete, publishable result.

**Do not start either until P3 is trained, swept and V3 is rendered.** The
three-policy ablation is the spine of the project; E1/E2 are extensions that gain
their meaning from it.

---

## 7. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Nominal policy never falls below 300 N — no story | Fatal | **R2 kill-switch in Phase 1**, ~1 h, before any training spend |
| P3 learns "walk cautiously/slowly" instead of recovering | High — undermines the claim | `pose`'s standing/walking/running std split already penalises this; gate on velocity tracking, not just fall rate |
| 28 h of training on a 6 GB laptop across days | Medium | Proven supervisor + cooldown + detach; 3 independent overnight runs, each individually resumable |
| Recovery metric mis-specified (repeat of the G4 lesson) | High | Separate "did it get pushed?" from "did it recover?" from the start; validate the metric against hand-labelled clips at R2 |
| Ablation shows DR doesn't help | **Not a risk — it's a result** | Literature disagrees (§1.7); report it either way |
| Disk exhaustion killing WSL | Medium | 2 GB disk guard already in the supervisor; ~6.5 GB free on D: |

---

## 8. Sequencing summary

```
PHASE 1   R0 revalidate env
(2-3 sess)  → R1 code: 3 env cfgs · deterministic push · recovery metric
                       · eval_push.py · make_split_video.py (R1.5)
  ~1 h GPU  → R2 🔴 KILL-SWITCH #1  sweep the EXISTING checkpoint
                  ├─ F* exists below 300 N?        ── no ──▶ RE-SCOPE
                  ├─ recovery metric matches 20 hand-labelled clips?
                  └─ captioned split-screen renders end to end?
            → R3 freeze train / eval / unseen distributions + seeds
                                    ↓
PHASE 2   R4  P1 nominal      9.3 h ──▶ 🎬 V1  "P1 undisturbed │ P1 pushed"
(3-4       │                              🔴 KILL-SWITCH #2: does P1 actually fall?
 nights)   │                                  no ──▶ STOP at 9 h, not 28 h
           R5  P2 +pushes     9.3 h ──▶ 🎬 V2  "P1 │ P2"
           R6  P3 +pushes+DR  9.3 h ──▶ 🎬 V3  "P1 │ P2 │ P3"   ← hero clip
             identical hyperparams · seed · iters · reward
             per-run early signal at iter 100 / 400 / cycle 1
            → R7 all three walk cleanly undisturbed (≥18 s, ≤5% falls)
                                    ↓
PHASE 3   R8.1 force sweep (0-300 N × 8 dir)  → R8.2 generalization gap
(1-2 sess)  → R8.3 failure analysis + failure clips → R8.4 ablation table
                                    ↓
PHASE 4   R9.1 final cut of V1-V3 → R9.2 plots → R9.3 portfolio repo + README
(1-2 sess)
```

**Three points where the project can be stopped cheaply:**

| | when | cost so far | question it answers |
|---|---|---|---|
| 🔴 #1 | R2, **hour 1** | ~1 h | Does a nominal G1 policy fall under a realistic push at all? |
| 🔴 #2 | end of R4, **hour 9** | 9 h | Does *our* P1 fall — visibly, on camera? |
| ⚠️ #3 | each cycle, every 3 h | ≤3 h | Is this run diverging or regressing? |

**Critical path:** R2 gates everything. One hour, and it can save thirty.
