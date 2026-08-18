"""Hard GPU preflight. Must pass before any training run.

Checking ``torch.cuda.is_available()`` is not sufficient for this project:

  * torch can see the GPU while mujoco-warp silently falls back to CPU,
  * a driver/toolkit mismatch produces a *warning* and a False flag rather than
    an error, so training "works" at unusable speed instead of failing,
  * a device can be visible but have too little free VRAM to be useful.

This script fails loudly and returns a non-zero exit code on any of those.

Usage:
  python scripts/preflight_gpu.py
  python scripts/preflight_gpu.py --task Unitree-G1-Flat --num-envs 64
"""

from __future__ import annotations

import argparse
import sys
import traceback

MIN_FREE_VRAM_GB = 4.0
MIN_FREE_DISK_GB = 15.0

checks: list[tuple[str, bool, str]] = []


def check_disk() -> bool:
  """Free disk on the volume backing the WSL root filesystem.

  Learned the hard way: a multi-GB package install filled the Windows drive
  hosting the WSL VHDX, and WSL then refused to boot at all
  (Wsl/Service/CreateInstance/E_FAIL). Nothing in the GPU checks would have
  caught that. Training also produces checkpoints and videos, so headroom
  matters for the whole run, not just install time.
  """
  import shutil as _shutil

  usage = _shutil.disk_usage("/")
  free_gb = usage.free / 1e9
  ok = record(
    "free disk (WSL root)",
    free_gb >= MIN_FREE_DISK_GB,
    f"{free_gb:.1f} GB free of {usage.total / 1e9:.1f} GB "
    f"(need >= {MIN_FREE_DISK_GB} GB for checkpoints and videos)",
  )
  if not ok:
    print(
      "         NOTE: the ext4.vhdx grows on its host Windows drive. Check "
      "free space there too -- if it fills, WSL will not boot.",
    )
  return ok


def record(name: str, ok: bool, detail: str = "") -> bool:
  checks.append((name, ok, detail))
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
  return ok


def check_torch() -> bool:
  import torch

  ok = record("torch imports", True, torch.__version__)

  avail = torch.cuda.is_available()
  ok &= record(
    "torch.cuda.is_available()",
    avail,
    "CUDA reports unavailable — driver/wheel mismatch. Training would silently "
    "run on CPU." if not avail else f"CUDA {torch.version.cuda}",
  )
  if not avail:
    return False

  name = torch.cuda.get_device_name(0)
  ok &= record("GPU visible", True, name)

  # A real kernel launch. is_available() can be True while launches fail.
  try:
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    ok &= record("CUDA kernel executes", c == c, "2048x2048 matmul on device")
  except Exception as exc:
    ok &= record("CUDA kernel executes", False, f"{type(exc).__name__}: {exc}")
    return False

  free, total = torch.cuda.mem_get_info()
  free_gb, total_gb = free / 1e9, total / 1e9
  ok &= record(
    "free VRAM",
    free_gb >= MIN_FREE_VRAM_GB,
    f"{free_gb:.2f} GB free of {total_gb:.2f} GB "
    f"(need >= {MIN_FREE_VRAM_GB} GB)",
  )
  return ok


def check_warp() -> bool:
  """mujoco-warp is the actual physics backend. It must be on CUDA."""
  try:
    import warp as wp
  except Exception as exc:
    return record("warp imports", False, f"{type(exc).__name__}: {exc}")

  record("warp imports", True, wp.config.version)

  try:
    wp.init()
    devices = [str(d) for d in wp.get_devices()]
    cuda_devices = [d for d in devices if d.startswith("cuda")]
    ok = record("warp sees a CUDA device", bool(cuda_devices), ", ".join(devices))
    if not ok:
      return False

    # Launch a real warp kernel on the GPU.
    dev = cuda_devices[0]
    arr = wp.zeros(1024, dtype=float, device=dev)
    wp.launch(_fill_kernel, dim=1024, inputs=[arr], device=dev)
    wp.synchronize_device(dev)
    total = float(arr.numpy().sum())
    return record(
      "warp kernel executes on GPU", total == 1024.0, f"device={dev}, sum={total}"
    )
  except Exception as exc:
    return record("warp kernel executes on GPU", False, f"{type(exc).__name__}: {exc}")


try:
  import warp as wp

  @wp.kernel
  def _fill_kernel(a: wp.array(dtype=float)):  # type: ignore[valid-type]
    i = wp.tid()
    a[i] = 1.0
except Exception:  # pragma: no cover - reported by check_warp
  _fill_kernel = None  # type: ignore[assignment]


def check_mujoco_warp() -> bool:
  try:
    import mujoco
    import mujoco_warp  # noqa: F401

    return record("mujoco / mujoco_warp import", True, f"mujoco {mujoco.__version__}")
  except Exception as exc:
    return record(
      "mujoco / mujoco_warp import",
      False,
      f"{type(exc).__name__}: {exc} "
      "(check the mujoco/mujoco-warp version pin in setup.py)",
    )


def check_env_steps(task: str, num_envs: int) -> bool:
  """Build a real mjlab env on CUDA and step it. The end-to-end proof."""
  try:
    import torch

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    if task not in list_tasks():
      return record("task registered", False, f"{task} not in registry")
    record("task registered", True, task)

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs

    before = torch.cuda.memory_allocated()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()

    actions = torch.zeros(
      (num_envs, env.action_space.shape[-1]), device="cuda:0"
    )
    for _ in range(10):
      env.step(actions)
    torch.cuda.synchronize()

    used = (torch.cuda.memory_allocated() - before) / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()

    ok = record(
      f"env steps on GPU ({num_envs} envs)",
      True,
      f"torch alloc +{used:.0f} MB, peak {peak:.2f} GB, "
      f"{free / 1e9:.2f} GB free of {total / 1e9:.2f} GB",
    )
    env.close()
    return ok
  except Exception as exc:
    traceback.print_exc()
    return record("env steps on GPU", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-G1-Flat")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--skip-env", action="store_true")
  args = ap.parse_args()

  print("=" * 70)
  print("GPU PREFLIGHT")
  print("=" * 70)

  ok = check_disk()
  ok &= check_torch()
  if ok:
    ok &= check_mujoco_warp()
    ok &= check_warp()
    if ok and not args.skip_env:
      ok &= check_env_steps(args.task, args.num_envs)

  print("=" * 70)
  failed = [n for n, passed, _ in checks if not passed]
  if failed:
    print(f"PREFLIGHT FAILED — {len(failed)} check(s): {', '.join(failed)}")
    print("Do NOT start training. Fix the environment first.")
    print("=" * 70)
    return 1
  print("PREFLIGHT PASSED — GPU is real and mujoco-warp runs on it.")
  print("=" * 70)
  return 0


if __name__ == "__main__":
  sys.exit(main())
