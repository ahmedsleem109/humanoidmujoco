"""Convert the comparison clips to GIFs so they DISPLAY in the README.

GitHub will not inline-play an .mp4 referenced by a repo-relative path -- it
renders as a link. An animated GIF referenced as an image plays inline, so the
reader sees the result without clicking anything. The .mp4 files stay in the repo
for full quality.

Two-pass palettegen/paletteuse rather than a naive conversion: a default 256-colour
quantisation wrecks the checkerboard floor with dithering noise and inflates the
file. Frames are also trimmed to the window that matters (just before the push
through the outcome) rather than the full clip.

Usage:  python scripts/make_gifs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FFMPEG = ("/home/sleem/venvs/mjlab/lib/python3.11/site-packages/"
          "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")

# (source mp4, output name, start_s, duration_s, output width)
CLIPS = [
  ("results/videos/v1_problem_20260816-194755/v1_problem.mp4",
   "v1_problem.gif", 1.8, 5.0, 720),
  ("results/videos/v2_p1_vs_p2_20260817-063111/v2_p1_vs_p2.mp4",
   "v2_p1_vs_p2.gif", 1.8, 5.0, 720),
  ("results/videos/v3_hero_20260817-211659/v3_hero.mp4",
   "v3_hero.gif", 1.8, 5.0, 860),
]

# 12 fps and a 128-colour palette keep each GIF near 3-4 MB. At 14 fps / full
# palette these came out at 12 MB each, which is slow to load inline and bloats
# the repository for no visible gain -- the scene is a matte robot on a flat
# floor, not something that needs 256 colours.
FPS = 12
MAX_COLORS = 128


def run(args: list[str]) -> None:
  r = subprocess.run(args, capture_output=True, text=True)
  if r.returncode != 0:
    print(r.stderr[-2000:], file=sys.stderr)
    raise SystemExit(f"ffmpeg failed: {' '.join(args[:6])} ...")


def main() -> None:
  out_dir = REPO / "results/gifs"
  out_dir.mkdir(parents=True, exist_ok=True)

  for src, name, start, dur, width in CLIPS:
    s = REPO / src
    if not s.exists():
      print(f"  SKIP (missing) {src}")
      continue
    palette = out_dir / f"_{name}.png"
    dest = out_dir / name

    filters = f"fps={FPS},scale={width}:-1:flags=lanczos"
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-ss", str(start), "-t", str(dur), "-i", str(s),
         "-vf", f"{filters},palettegen=max_colors={MAX_COLORS}:stats_mode=diff",
         str(palette)])
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-ss", str(start), "-t", str(dur), "-i", str(s), "-i", str(palette),
         "-lavfi", f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
         "-loop", "0", str(dest)])
    palette.unlink(missing_ok=True)
    print(f"  {name:22s} {dest.stat().st_size / 1024 / 1024:5.1f} MB")

  print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
  main()
