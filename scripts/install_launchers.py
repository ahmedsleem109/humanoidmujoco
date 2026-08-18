"""Install shell launchers into $HOME with LF endings.

`install_launchers.sh` does the same job, but it cannot bootstrap itself: it
lives on the Windows filesystem with CRLF endings, so `/usr/bin/env bash\r`
fails before it can strip anything. And the obvious inline workaround --
`bash -c 'tr -d "\r" < x > y'` -- is destroyed by PowerShell, which strips the
backslash and turns it into `tr -d "r"`, silently deleting every letter `r` in
the script (gotcha 19).

Python has neither problem: no shell quoting, no escape mangling.

Usage:  python scripts/install_launchers.py run_train gpu_cooldown detach
        python scripts/install_launchers.py --all
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
HOME = Path(os.path.expanduser("~"))

names = sys.argv[1:]
if not names or names == ["--all"]:
  names = [p.stem for p in sorted(SRC.glob("*.sh"))]

fail = 0
for name in names:
  src = SRC / f"{name}.sh"
  if not src.exists():
    print(f"MISSING  {name}.sh")
    fail = 1
    continue

  data = src.read_bytes()
  data = data.lstrip(b"\xef\xbb\xbf").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
  dest = HOME / f"{name}.sh"
  dest.write_bytes(data)
  dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  proc = subprocess.run(["bash", "-n", str(dest)], capture_output=True, text=True)
  if proc.returncode == 0:
    print(f"OK       {name}.sh -> {dest}")
  else:
    print(f"SYNTAX   {name}.sh\n{proc.stderr}")
    fail = 1

raise SystemExit(fail)
