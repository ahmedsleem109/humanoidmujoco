"""Split-screen policy comparison video with burned-in captions.

Produces V1/V2/V3: N policies rendered under a byte-identical scenario, stacked
horizontally, with captions.

Why the scenario is pinned so hard
----------------------------------
The comparison only means anything if the policy is the only thing that differs.
So every panel shares the seed, the terrain, the command, the camera, and the
push (magnitude, azimuth, instant, duration). `deterministic_push` guarantees the
last of those; `_fix_scenario` in `eval_push.py` guarantees the rest.

Captions: why PIL and not drawtext
----------------------------------
The only ffmpeg available here is the static binary shipped inside
`imageio-ffmpeg`, and it is built WITHOUT libfreetype -- `-filters` lists
`hstack` and `overlay` but no `drawtext`. There is no system ffmpeg and no
passwordless sudo to install one. So captions are rendered to RGBA PNGs with PIL
(DejaVuSans-Bold is present under /usr/share/fonts) and composited with
`overlay`, timed via `enable='between(t,a,b)'`.

This was verified before the pipeline was written rather than discovered at the
end of Phase 4.

Usage
-----
  ~/pyrun.sh scripts/make_split_video.py \
      --panels "NOMINAL=checkpoints/p1/model_6600.pt,ROBUST=checkpoints/p3/model_6600.pt" \
      --force-n 300 --direction-deg 90 --label v2
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

import eval_gate  # noqa: E402
import eval_push  # noqa: E402
from src.tasks.velocity.mdp.disturbance import PUSH_STATE_ATTR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass
class SplitVideoConfig:
  panels: str
  """Comma-separated LABEL=checkpoint pairs, left to right."""
  label: str = "split"
  task: str = "Unitree-G1-Flat"

  force_n: float = 300.0
  direction_deg: float = 90.0
  push_time_s: float = 3.0
  push_duration_s: float = 0.1
  episode_length_s: float = 9.0
  command_x: float = 1.0

  width: int = 640
  height: int = 480
  fps: int = 50
  seed: int = 12345
  device: str = "cuda:0"

  # Camera. The mjlab default (distance 5.0, elevation -45) is a high, distant
  # survey shot: the robot occupies a small part of the frame and the top third
  # is empty sky. Closer and nearer eye-level reads far better at the size a
  # video is actually watched.
  cam_distance: float = 3.2
  cam_elevation: float = -12.0
  cam_azimuth: float = 120.0
  """3/4 view: shows forward travel and lateral toppling at the same time."""
  cam_lookat_z: float = 0.55
  """Raise the target toward the pelvis so the robot sits centred, not low."""

  hook: str = "What happens when you push a humanoid mid-walk?"
  payoff: str = "It didn't avoid the disturbance. It learned to recover from it."
  out_root: str = "results/videos"


# ---------------------------------------------------------------------------
# Caption rendering (PIL -> RGBA PNG -> ffmpeg overlay)
# ---------------------------------------------------------------------------


def _text_png(path: Path, text: str, width: int, height: int, size: int,
              fill=(255, 255, 255, 255), bg=(0, 0, 0, 150), pad: int = 18) -> Path:
  from PIL import Image, ImageDraw, ImageFont

  try:
    font = ImageFont.truetype(FONT, size)
  except OSError:
    font = ImageFont.truetype(FONT_REG, size)

  img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
  d = ImageDraw.Draw(img)
  box = d.textbbox((0, 0), text, font=font)
  tw, th = box[2] - box[0], box[3] - box[1]
  x, y = (width - tw) // 2, (height - th) // 2
  if bg is not None:
    d.rounded_rectangle(
      [x - pad, y - pad, x + tw + pad, y + th + pad], radius=10, fill=bg
    )
  d.text((x - box[0], y - box[1]), text, font=font, fill=fill)
  img.save(path)
  return path


# ---------------------------------------------------------------------------
# Rendering one panel
# ---------------------------------------------------------------------------


def render_panel(cfg: SplitVideoConfig, checkpoint: str, out_dir: Path,
                 name: str, force_n: float | None = None) -> Path:
  """Render one policy under the pinned scenario. Returns the clip path.

  `force_n` overrides the global magnitude for this panel only. That is needed
  for V1, whose whole point is the SAME policy with and without a push -- the
  panels must differ in the disturbance, not in the checkpoint.
  """
  clip_dir = out_dir / f"raw_{name}"
  clip_dir.mkdir(parents=True, exist_ok=True)

  push_cfg = eval_push.PushEvalConfig(
    checkpoint=checkpoint,
    task=cfg.task,
    push_time_s=cfg.push_time_s,
    push_duration_s=cfg.push_duration_s,
    episode_length_s=cfg.episode_length_s,
    command_x=cfg.command_x,
    seed=cfg.seed,
  )

  env_cfg = eval_gate.load_env_cfg(cfg.task, play=True)
  agent_cfg = eval_gate.load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.episode_length_s = cfg.episode_length_s
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = cfg.cam_distance
  env_cfg.viewer.elevation = cfg.cam_elevation
  env_cfg.viewer.azimuth = cfg.cam_azimuth
  env_cfg.viewer.lookat = (0.0, 0.0, cfg.cam_lookat_z)
  eval_push._fix_scenario(env_cfg, push_cfg)  # noqa: SLF001 - shared scenario

  # Disable the fall termination FOR VIDEO ONLY.
  #
  # With it active, a robot that falls terminates, resets, and walks again inside
  # the same clip -- so a frame taken a few seconds after the push shows a robot
  # walking happily, which reads as "it shrugged off the push" when the truth is
  # "it fell and the episode restarted". That is the most misleading artifact
  # this pipeline could produce, and it nearly shipped.
  #
  # Removing the termination makes a downed robot stay down for the rest of the
  # clip, which is what actually happened. Metrics are unaffected: they come from
  # eval_push.py, never from the video.
  env_cfg.terminations.pop("fell_over", None)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  base_env = env  # keep a handle: wrappers hide the event manager and push state
  params = eval_push._push_params(env)  # noqa: SLF001
  params["force_n"] = float(cfg.force_n if force_n is None else force_n)
  params["direction_deg"] = float(cfg.direction_deg)
  print(f"    scenario: {params['force_n']:.0f} N @ {params['direction_deg']:.0f} deg, "
        f"t={params['trigger_time_s']}s, {params['duration_s']}s", flush=True)

  steps = int(round(cfg.episode_length_s / env.step_dt))
  env = VideoRecorder(
    env,
    video_folder=clip_dir,
    episode_trigger=lambda ep: ep == 0,
    video_length=steps,
    name_prefix="panel",
    disable_logger=True,
  )
  wrapped, policy = eval_gate.load_policy(
    env, agent_cfg, cfg.task, checkpoint, cfg.device
  )

  torch.manual_seed(cfg.seed)
  wrapped.seed(cfg.seed)
  obs, _ = wrapped.reset()

  # Verify the push actually lands, rather than trusting that it was configured.
  # A clip whose disturbance silently failed to fire looks exactly like a policy
  # that shrugged it off -- and would be published as the latter.
  # `fired` must be LATCHED across the rollout, not read at the end: the push
  # event clears its own state whenever an episode resets, so the end-of-rollout
  # value reports the *current* episode and reads False even when the push landed.
  max_tilt = 0.0
  fired = False
  mag = 0.0
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, _, _ = wrapped.step(policy(obs))
      g = base_env.scene["robot"].data.projected_gravity_b[0, 2]
      tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g)))))
      max_tilt = max(max_tilt, tilt)
      st = getattr(base_env, PUSH_STATE_ATTR, None)
      if st is not None and bool(st.fired.any()):
        fired = True
        mag = max(mag, float(st.magnitude.max()))
  wrapped.close()
  print(f"    push fired={fired} magnitude={mag:.0f} N  max_tilt={max_tilt:.1f} deg",
        flush=True)
  if not fired:
    raise RuntimeError(
      f"panel {name}: the push never fired -- refusing to emit a clip that "
      f"misrepresents the experiment"
    )

  clips = sorted(clip_dir.glob("*.mp4"))
  if not clips:
    raise RuntimeError(f"no clip rendered for {name}")
  return clips[0]


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def build(cfg: SplitVideoConfig, clips: list[Path], labels: list[str],
          out_dir: Path, ffmpeg: str) -> Path:
  n = len(clips)
  w, h = cfg.width, cfg.height
  total_w = w * n

  cap_dir = out_dir / "captions"
  cap_dir.mkdir(parents=True, exist_ok=True)

  # Per-panel persistent labels.
  label_pngs = [
    _text_png(cap_dir / f"label{i}.png", lab, w, 70, 30) for i, lab in enumerate(labels)
  ]
  hook_png = _text_png(cap_dir / "hook.png", cfg.hook, total_w, 90, 34)
  push_png = _text_png(
    cap_dir / "push.png",
    f"EXTERNAL DISTURBANCE   {cfg.force_n:.0f} N",
    total_w, 90, 34, fill=(255, 210, 90, 255),
  )
  payoff_png = _text_png(cap_dir / "payoff.png", cfg.payoff, total_w, 90, 30)

  inputs: list[str] = []
  for c in clips:
    inputs += ["-i", str(c)]
  for p in [*label_pngs, hook_png, push_png, payoff_png]:
    inputs += ["-i", str(p)]

  fc: list[str] = []
  for i in range(n):
    fc.append(f"[{i}:v]scale={w}:{h},setsar=1[v{i}]")
  fc.append("".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[row]")

  cur = "row"
  # Panel labels sit at the top of each panel, always visible.
  for i in range(n):
    src = n + i
    fc.append(f"[{cur}][{src}:v]overlay=x={i * w}:y=12[lb{i}]")
    cur = f"lb{i}"

  t_push = cfg.push_time_s
  hook_i, push_i, payoff_i = 2 * n, 2 * n + 1, 2 * n + 2
  fc.append(
    f"[{cur}][{hook_i}:v]overlay=x=0:y=H-110:"
    f"enable='between(t,0.2,{t_push - 0.4:.2f})'[c1]"
  )
  fc.append(
    f"[c1][{push_i}:v]overlay=x=0:y=H-110:"
    f"enable='between(t,{t_push:.2f},{t_push + 1.6:.2f})'[c2]"
  )
  fc.append(
    f"[c2][{payoff_i}:v]overlay=x=0:y=H-110:"
    f"enable='gte(t,{t_push + 2.6:.2f})'[out]"
  )

  dest = out_dir / f"{cfg.label}.mp4"
  cmd = [
    ffmpeg, "-y", *inputs,
    "-filter_complex", ";".join(fc),
    "-map", "[out]",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
    "-r", str(cfg.fps), str(dest),
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0 or not dest.exists():
    raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")
  return dest


def main() -> None:
  cfg = tyro.cli(SplitVideoConfig)

  # LABEL=checkpoint[@force_n]. The optional @force overrides the global
  # magnitude for that panel only.
  pairs: list[tuple[str, str, float | None]] = []
  for chunk in cfg.panels.split(","):
    if "=" not in chunk:
      raise SystemExit(
        f"bad --panels entry {chunk!r}; expected LABEL=checkpoint[@force_n]"
      )
    lab, ck = chunk.split("=", 1)
    force: float | None = None
    if "@" in ck:
      ck, f = ck.rsplit("@", 1)
      force = float(f)
    pairs.append((lab.strip(), ck.strip(), force))

  ffmpeg = eval_gate.resolve_ffmpeg()
  if not ffmpeg:
    raise SystemExit("ffmpeg unavailable; refusing to report a video")

  stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  out_dir = REPO / cfg.out_root / f"{cfg.label}_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)

  clips = []
  for i, (lab, ck, force) in enumerate(pairs):
    shown = cfg.force_n if force is None else force
    print(f"[render {i + 1}/{len(pairs)}] {lab}  <- {ck}  @{shown:.0f} N", flush=True)
    clips.append(render_panel(cfg, ck, out_dir, f"p{i}", force_n=force))

  print("[compose] stacking and captioning", flush=True)
  dest = build(cfg, clips, [p[0] for p in pairs], out_dir, ffmpeg)
  print(f"\nvideo -> {dest}")


if __name__ == "__main__":
  main()
