"""Video-gated evaluation harness for the G1 obstacle-traversal project.

This is the single entry point that "finishes" a training task. It runs a
deterministic evaluation, computes metrics, renders review videos, compares the
result against a declared gate config, and writes a report card.

It deliberately does NOT reuse ``scripts/play.py``: play.py hands control to an
interactive viewer (native or viser), which never returns and cannot be used
headlessly.

Exit code is 0 only if every declared criterion passes. A green exit code is
still not a pass -- a human must watch the videos and sign off in
``results/gates/GATES.md``.

Usage:
  python scripts/eval_gate.py Unitree-G1-Flat \
    --checkpoint logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt \
    --gate gates/T1_2_baseline_flat.yaml \
    --episodes 200
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class GateConfig:
  checkpoint: str
  """Path to the .pt checkpoint to evaluate."""
  gate: str | None = None
  """Path to the gate YAML declaring exit criteria. Omit to report metrics only."""
  episodes: int = 200
  """Number of completed episodes to collect for metrics."""
  num_envs: int = 64
  """Parallel envs for the (headless) metrics pass. Keep modest to fit VRAM."""
  video_episodes: int = 6
  """Number of single-env episodes to render for review."""
  max_video_clips: int = 5
  """Cap on clips concatenated into failures.mp4."""
  video_width: int = 960
  video_height: int = 540
  episode_length_s: float = 20.0
  """Episode time limit for evaluation, in seconds.

  MUST be set explicitly. mjlab's *play* configs set episode_length_s to 1e9 so an
  interactive viewer never resets itself. In a batch rollout that means `time_out`
  never fires, and against a policy that does not fall, no episode ever ends -- the
  evaluation runs forever. Default matches the training config (20 s).
  """
  video_length: int = 500
  """Max frames per clip. At the 0.02 s control step, 500 frames = 10 s."""
  seed: int = 0
  no_video: bool = False
  """Skip the video pass. For debugging only -- a gate without video is not a gate."""
  out: str = "results/gates"
  device: str | None = None
  label: str = ""
  """Optional short label folded into the output directory name."""


# ---------------------------------------------------------------------------
# ffmpeg resolution
# ---------------------------------------------------------------------------


def resolve_ffmpeg() -> str | None:
  """Find an ffmpeg binary.

  mediapy (used by mjlab's VideoRecorder) shells out to ffmpeg. On a WSL box
  without sudo we cannot apt-install it, so fall back to the static binary
  bundled with the imageio-ffmpeg wheel and put it on PATH.
  """
  found = shutil.which("ffmpeg")
  if found:
    return found
  try:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
  except Exception:
    return None
  if not exe or not Path(exe).exists():
    return None
  # mediapy resolves "ffmpeg" from PATH, so expose the bundled binary there.
  bin_dir = Path.home() / ".local" / "bin"
  bin_dir.mkdir(parents=True, exist_ok=True)
  link = bin_dir / "ffmpeg"
  if not link.exists():
    try:
      link.symlink_to(exe)
    except OSError:
      shutil.copy2(exe, link)
      link.chmod(0o755)
  os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
  return str(link)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
  try:
    return subprocess.check_output(
      ["git", *args], stderr=subprocess.DEVNULL, text=True
    ).strip()
  except Exception:
    return "unknown"


def _version(mod: str) -> str:
  try:
    import importlib.metadata as md

    return md.version(mod)
  except Exception:
    return "unknown"


def collect_provenance(task_id: str, cfg: GateConfig, num_envs: int) -> dict:
  gpu = "cpu"
  if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
  return {
    "task": task_id,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "git_commit": _git("rev-parse", "HEAD"),
    "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty": bool(_git("status", "--porcelain")),
    "checkpoint": str(Path(cfg.checkpoint).resolve()),
    "seed": cfg.seed,
    "eval_num_envs": num_envs,
    "eval_episodes": cfg.episodes,
    "gpu": gpu,
    "cuda": torch.version.cuda or "n/a",
    "torch": torch.__version__,
    "mujoco": _version("mujoco"),
    "mujoco_warp": _version("mujoco-warp"),
    "mjlab": _version("mjlab"),
    "python": platform.python_version(),
    "platform": platform.platform(),
  }


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def build_env_and_policy(
  task_id: str, cfg: GateConfig, num_envs: int, device: str, render: bool
):
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = cfg.seed
  # See GateConfig.episode_length_s: play configs use 1e9 and would never time out.
  env_cfg.episode_length_s = cfg.episode_length_s
  if render:
    env_cfg.viewer.width = cfg.video_width
    env_cfg.viewer.height = cfg.video_height

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if render else None
  )
  return env, agent_cfg


def load_policy(env, agent_cfg, task_id: str, checkpoint: str, device: str):
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  return wrapped, runner.get_inference_policy(device=device)


# ---------------------------------------------------------------------------
# Metrics pass
# ---------------------------------------------------------------------------


def run_metrics_pass(task_id: str, cfg: GateConfig, device: str) -> dict:
  """Roll out headlessly and collect per-episode statistics.

  Distinguishes falls (terminated) from timeouts (truncated). The wrapper folds
  both into ``dones``, so we recover ``truncated`` from ``extras["time_outs"]``.
  """
  env, agent_cfg = build_env_and_policy(
    task_id, cfg, cfg.num_envs, device, render=False
  )
  wrapped, policy = load_policy(env, agent_cfg, task_id, cfg.checkpoint, device)

  torch.manual_seed(cfg.seed)
  np.random.seed(cfg.seed)
  wrapped.seed(cfg.seed)

  obs, _ = wrapped.reset()
  n = cfg.num_envs
  dt = float(env.cfg.sim.mujoco.timestep) * int(env.cfg.decimation)

  ep_len = torch.zeros(n, dtype=torch.long, device=device)
  ep_rew = torch.zeros(n, dtype=torch.float, device=device)
  # Latch: did this episode EVER satisfy the success condition?
  ep_success = torch.zeros(n, dtype=torch.bool, device=device)

  lengths: list[int] = []
  returns: list[float] = []
  outcomes: list[str] = []  # "timeout" | "fall"
  log_acc: dict[str, list[float]] = defaultdict(list)
  success_flags: list[float] = []

  # Bounded: enough for the required episode cycles plus slack, but never runaway.
  steps_per_episode = env.cfg.episode_length_s / dt
  cycles = math.ceil(cfg.episodes / n) + 1
  max_steps = int(min(steps_per_episode * cycles * 1.5, 200_000))
  steps = 0
  print(
    f"  [metrics] {n} envs, {steps_per_episode:.0f} steps/episode, "
    f"budget {max_steps} steps for {cfg.episodes} episodes",
    flush=True,
  )

  t0 = time.time()
  with torch.inference_mode():
    while len(lengths) < cfg.episodes and steps < max_steps:
      actions = policy(obs)
      obs, rew, dones, extras = wrapped.step(actions)
      steps += 1

      if steps % 200 == 0:
        el = time.time() - t0
        print(
          f"  [metrics] step {steps}/{max_steps}  episodes {len(lengths)}/{cfg.episodes}"
          f"  {steps / max(el, 1e-6):.0f} steps/s",
          flush=True,
        )

      ep_len += 1
      ep_rew += rew

      succ = _read_success(env)
      if succ is not None:
        ep_success |= succ.to(torch.bool)

      done_idx = (dones > 0).nonzero(as_tuple=False).flatten()
      if done_idx.numel() > 0:
        time_outs = extras.get("time_outs")
        if time_outs is None:
          time_outs = torch.zeros_like(dones, dtype=torch.bool)
        time_outs = time_outs.to(torch.bool)

        has_success = _read_success(env) is not None
        for i in done_idx.tolist():
          if len(lengths) >= cfg.episodes:
            break
          lengths.append(int(ep_len[i].item()))
          returns.append(float(ep_rew[i].item()))
          outcomes.append("timeout" if bool(time_outs[i].item()) else "fall")
          if has_success:
            success_flags.append(float(ep_success[i].item()))

        ep_len[done_idx] = 0
        ep_rew[done_idx] = 0.0
        ep_success[done_idx] = False

        # mjlab publishes episode-average scalars in extras["log"] only when
        # episodes reset. Sampling it on every step averages in ~999 stale
        # entries and dilutes the value by roughly the episode length -- it made
        # velocity tracking error read 0.0003 instead of the true ~0.9.
        log = extras.get("log")
        if isinstance(log, dict):
          for k, v in log.items():
            try:
              log_acc[str(k)].append(float(v))
            except (TypeError, ValueError):
              continue

  wrapped.close()

  if not lengths:
    raise RuntimeError(
      "No episodes completed during the metrics pass. Check the checkpoint and env."
    )

  n_ep = len(lengths)
  falls = sum(1 for o in outcomes if o == "fall")
  metrics = {
    "episodes": n_ep,
    "mean_episode_length_steps": float(np.mean(lengths)),
    "mean_episode_length_s": float(np.mean(lengths) * dt),
    "max_episode_length_s": float(env.cfg.episode_length_s),
    "fall_rate": falls / n_ep,
    "timeout_rate": 1.0 - falls / n_ep,
    "mean_episode_return": float(np.mean(returns)),
    "control_dt": dt,
  }
  if success_flags:
    metrics["success_rate"] = float(np.mean(success_flags))
  for k, vals in log_acc.items():
    if vals:
      metrics[f"log/{k}"] = float(np.mean(vals))
  return metrics


def _read_success(env) -> torch.Tensor | None:
  """Per-env, current-step traversal-success flag. None if the task has none.

  Deliberately does NOT use ``Episode_Metrics/traversal_success`` from
  ``MetricsManager.reset()``. That value is ``mean(episode_sum / step_count)``
  -- a *time-average over the episode*, so an episode that clears the obstacle
  halfway through scores 0.5. Reporting it as "success rate" would silently
  mean "fraction of timesteps spent past the obstacle", which is a different
  quantity and would understate a genuine success.

  Instead we read the raw per-env step value and let the caller latch it: an
  episode counts as a success if the condition held at ANY point during it.
  """
  mgr = getattr(env, "metrics_manager", None)
  if mgr is None:
    return None
  try:
    names = list(mgr.active_terms)
  except Exception:
    return None
  for name in ("traversal_success", "success"):
    if name in names:
      return mgr._step_values[:, names.index(name)]
  return None


# ---------------------------------------------------------------------------
# Video pass
# ---------------------------------------------------------------------------


def run_video_pass(task_id: str, cfg: GateConfig, device: str, out_dir: Path) -> dict:
  """Render one clip per episode on a single env, then classify each clip.

  VideoRecorder only captures env[0], so the video pass runs with num_envs=1 and
  an episode trigger, producing one mp4 per episode which we then group into
  overview.mp4 and failures.mp4.
  """
  clip_dir = out_dir / "clips"
  env, agent_cfg = build_env_and_policy(task_id, cfg, 1, device, render=True)
  env = VideoRecorder(
    env,
    video_folder=clip_dir,
    episode_trigger=lambda ep: True,
    video_length=cfg.video_length,  # bounded: 1e9-length episodes would never end
    name_prefix="ep",
    disable_logger=True,
  )
  wrapped, policy = load_policy(env, agent_cfg, task_id, cfg.checkpoint, device)

  torch.manual_seed(cfg.seed + 1)
  wrapped.seed(cfg.seed + 1)
  obs, _ = wrapped.reset()

  results: list[tuple[int, str]] = []
  ep_index = 0
  steps = 0
  step_cap = cfg.video_episodes * 4000 + 5000

  with torch.inference_mode():
    while ep_index < cfg.video_episodes and steps < step_cap:
      actions = policy(obs)
      obs, _, dones, extras = wrapped.step(actions)
      steps += 1
      if int(dones[0].item()) > 0:
        time_outs = extras.get("time_outs")
        fell = True
        if time_outs is not None:
          fell = not bool(time_outs.to(torch.bool)[0].item())
        results.append((ep_index, "fall" if fell else "timeout"))
        ep_index += 1

  wrapped.close()

  clips = sorted(clip_dir.glob("ep-episode-*.mp4"))
  good = [c for c, (_, o) in zip(clips, results) if o == "timeout"]
  bad = [c for c, (_, o) in zip(clips, results) if o == "fall"]

  made = {}
  overview = _concat(clips[: cfg.video_episodes], out_dir / "overview.mp4")
  if overview:
    made["overview"] = str(overview)
  if bad:
    failures = _concat(bad[: cfg.max_video_clips], out_dir / "failures.mp4")
    if failures:
      made["failures"] = str(failures)
  made["clips"] = [str(c) for c in clips]
  made["episode_outcomes"] = [o for _, o in results]
  made["clean_episodes"] = len(good)
  return made


def _concat(clips: list[Path], dest: Path) -> Path | None:
  clips = [c for c in clips if c.exists()]
  if not clips:
    return None
  if len(clips) == 1:
    shutil.copy2(clips[0], dest)
    return dest
  ffmpeg = resolve_ffmpeg()
  if not ffmpeg:
    print("[WARN] ffmpeg unavailable; skipping concatenation.", file=sys.stderr)
    return None
  listing = dest.with_suffix(".txt")
  listing.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips))
  try:
    subprocess.run(
      [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
       "-c", "copy", str(dest)],
      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
  except subprocess.CalledProcessError:
    print(f"[WARN] ffmpeg concat failed for {dest.name}", file=sys.stderr)
    return None
  finally:
    listing.unlink(missing_ok=True)
  return dest


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

_OPS = {
  "min": lambda v, t: v >= t,
  "max": lambda v, t: v <= t,
  "equals": lambda v, t: v == t,
}


def evaluate_gate(metrics: dict, gate: dict) -> tuple[list[dict], bool]:
  """Compare metrics against declared criteria.

  Gate YAML shape:
    name: T1.2 flat baseline
    criteria:
      - metric: mean_episode_length_s
        min: 18.0
        why: episodes must run near the 20 s limit
      - metric: fall_rate
        max: 0.05
  """
  rows: list[dict] = []
  for crit in gate.get("criteria", []):
    key = crit["metric"]
    value = metrics.get(key)
    row = {"metric": key, "value": value, "why": crit.get("why", "")}
    if value is None:
      row.update(threshold="-", op="-", passed=False, note="metric not produced")
      rows.append(row)
      continue
    passed = True
    parts = []
    for op, fn in _OPS.items():
      if op in crit:
        parts.append(f"{op} {crit[op]}")
        passed = passed and fn(value, crit[op])
    row.update(
      threshold=", ".join(parts) or "-",
      op="",
      passed=passed,
      note="",
    )
    rows.append(row)
  return rows, all(r["passed"] for r in rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
  out_dir: Path, task_id: str, gate: dict | None, rows, ok: bool,
  metrics: dict, videos: dict, prov: dict,
) -> Path:
  lines: list[str] = []
  name = gate.get("name", task_id) if gate else task_id
  verdict = "PASS" if ok else "FAIL"
  lines.append(f"# Gate report — {name}")
  lines.append("")
  lines.append(f"**Automated verdict: {verdict}**")
  lines.append("")
  lines.append(
    "> A green verdict is necessary but NOT sufficient. Watch `overview.mp4` and "
    "`failures.mp4` and record a human verdict in `results/gates/GATES.md`. "
    "Reject any success that depends on a simulator artifact (obstacle "
    "penetration, contact tunnelling, ramp-launching, box-surfing)."
  )
  lines.append("")

  if rows:
    lines.append("## Criteria")
    lines.append("")
    lines.append("| Metric | Measured | Threshold | Verdict | Why |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
      v = r["value"]
      vs = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
      mark = "PASS" if r["passed"] else "**FAIL**"
      note = r.get("note") or r.get("why", "")
      lines.append(f"| `{r['metric']}` | {vs} | {r['threshold']} | {mark} | {note} |")
    lines.append("")
  else:
    lines.append("## Criteria\n\n_No gate config supplied — metrics only._\n")

  lines.append("## Videos")
  lines.append("")
  if videos.get("overview"):
    lines.append(f"- Overview: `{Path(videos['overview']).name}`")
  if videos.get("failures"):
    lines.append(f"- Failures: `{Path(videos['failures']).name}`")
  if videos.get("episode_outcomes"):
    lines.append(f"- Episode outcomes: {videos['episode_outcomes']}")
  if not videos:
    lines.append("- _Video pass skipped._")
  lines.append("")

  lines.append("## Metrics")
  lines.append("")
  lines.append("| Key | Value |")
  lines.append("|---|---|")
  for k in sorted(metrics):
    v = metrics[k]
    vs = f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
    lines.append(f"| `{k}` | {vs} |")
  lines.append("")

  lines.append("## Provenance")
  lines.append("")
  lines.append("```json")
  lines.append(json.dumps(prov, indent=2))
  lines.append("```")

  report = out_dir / "report.md"
  report.write_text("\n".join(lines), encoding="utf-8")
  (out_dir / "metrics.json").write_text(
    json.dumps({"metrics": metrics, "provenance": prov}, indent=2), encoding="utf-8"
  )
  return report


def append_ledger(root: Path, task_id: str, gate_name: str, ok: bool, out_dir: Path):
  ledger = root / "GATES.md"
  if not ledger.exists():
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
      "# Gate ledger\n\n"
      "Every training task ends here. The automated verdict is not a pass on its "
      "own -- a human must watch the videos and fill in the last two columns.\n\n"
      "| Gate | Date | Task | Report | Auto | Video reviewed | Human verdict | Notes |\n"
      "|---|---|---|---|---|---|---|---|\n",
      encoding="utf-8",
    )
  date = datetime.now().strftime("%Y-%m-%d %H:%M")
  rel = out_dir.relative_to(root) if out_dir.is_relative_to(root) else out_dir
  with ledger.open("a", encoding="utf-8") as fh:
    fh.write(
      f"| {gate_name} | {date} | {task_id} | [{rel}]({rel}/report.md) | "
      f"{'PASS' if ok else 'FAIL'} | ☐ | ☐ | |\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_gate(task_id: str, cfg: GateConfig) -> int:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  ckpt = Path(cfg.checkpoint)
  if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

  gate = None
  if cfg.gate:
    gate = yaml.safe_load(Path(cfg.gate).read_text(encoding="utf-8"))

  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  tag = f"{task_id}_{cfg.label}_{stamp}" if cfg.label else f"{task_id}_{stamp}"
  root = Path(cfg.out)
  out_dir = root / tag
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"[GATE] task={task_id} device={device} out={out_dir}")

  print("[GATE] metrics pass ...")
  metrics = run_metrics_pass(task_id, cfg, device)

  videos: dict = {}
  if not cfg.no_video:
    if resolve_ffmpeg() is None:
      print(
        "[ERROR] No ffmpeg available. Install imageio-ffmpeg into the venv. "
        "Refusing to report a gate without video.",
        file=sys.stderr,
      )
      return 2
    print("[GATE] video pass ...")
    videos = run_video_pass(task_id, cfg, device, out_dir)

  rows, ok = ([], True)
  if gate:
    rows, ok = evaluate_gate(metrics, gate)

  prov = collect_provenance(task_id, cfg, cfg.num_envs)
  report = write_report(out_dir, task_id, gate, rows, ok, metrics, videos, prov)
  append_ledger(root, task_id, gate.get("name", "-") if gate else "-", ok, out_dir)

  print("\n" + "=" * 68)
  for r in rows:
    v = r["value"]
    vs = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
    print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['metric']}: "
          f"{vs} (needs {r['threshold']})")
  print("=" * 68)
  print(f"  AUTOMATED VERDICT: {'PASS' if ok else 'FAIL'}")
  print(f"  Report: {report}")
  if videos.get("overview"):
    print(f"  Watch:  {videos['overview']}")
  if videos.get("failures"):
    print(f"  Watch:  {videos['failures']}")
  print("  Human video sign-off still required in results/gates/GATES.md")
  print("=" * 68 + "\n")
  return 0 if ok else 1


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(GateConfig, args=remaining)
  sys.exit(run_gate(chosen_task, cfg))


if __name__ == "__main__":
  main()
