# SPDX-License-Identifier: MIT
"""Runtime bootstrap: GPU detection, venv creation, dep install, LD_LIBRARY_PATH re-exec.

This module is intentionally standalone — it imports nothing from the rest of
the transcriber package so it can run before dependencies are installed.
Call `bootstrap_startup()` exactly once, at the top of the CLI entry point
(`cli.py`). The API entry point skips bootstrap because `uv sync` already
guarantees a correct environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from transcriber.models import _DRIVER_CUDA_MAP, GpuInfo

_DETECTED_GPU_INFO: GpuInfo | None = None


def detect_gpu_info() -> GpuInfo | None:
    """Return the first GPU's compute capability and inferred CUDA version."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        compute_cap = float(parts[0])
        driver_major = int(parts[1].split(".")[0])
        cuda_major = 0
        for min_driver, cuda_ver in _DRIVER_CUDA_MAP:
            if driver_major >= min_driver:
                cuda_major = cuda_ver
                break
        return GpuInfo(compute_cap=compute_cap, cuda_major=cuda_major)
    except Exception:
        return None


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))


def _ensure_venv_and_reexec() -> None:
    """Create .venv in the project root and re-exec into it if not already inside one.

    Ubuntu and other PEP 668 distros mark system Python as externally-managed,
    which blocks pip globally. Creating a project-local venv avoids the restriction
    without requiring the user to set up an environment manually.
    """
    if _in_venv() or os.environ.get("LOCAL_TRANSCRIBER_IN_VENV") == "1":
        return

    script_dir = Path(__file__).resolve().parent.parent.parent
    venv_dir = script_dir / ".venv"
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"

    if not venv_python.exists():
        print(f"Creating virtual environment at {venv_dir} ...", file=sys.stderr)
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    env = os.environ.copy()
    env["LOCAL_TRANSCRIBER_IN_VENV"] = "1"
    env["VIRTUAL_ENV"] = str(venv_dir)
    os.execvpe(str(venv_python), [str(venv_python), *sys.argv], env)


def _pip_installer() -> list[str]:
    """Return the pip command prefix for the current environment.

    Uses `uv pip` when uv is available inside a venv; otherwise falls back to
    the standard pip module. Should only be called from within a venv.
    """
    if shutil.which("uv"):
        return ["uv", "pip"]
    return [sys.executable, "-m", "pip"]


def ensure_dependencies(gpu_info: GpuInfo | None) -> None:
    """Install faster-whisper and CUDA-version-matched libraries if not present."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return

    print(
        "faster-whisper not found. Installing automatically. Pass --no-auto-install to skip.",
        file=sys.stderr,
    )

    packages = ["faster-whisper"]
    if gpu_info and gpu_info.cuda_major >= 12:
        packages += ["nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"]
    elif gpu_info and gpu_info.cuda_major == 11:
        # CUDA 11 + cuDNN 8 requires an older ctranslate2.
        packages += ["ctranslate2==4.4.0"]

    installer = _pip_installer()
    cmd = installer + ["install"] + packages
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Dependency installation failed. Install manually:\n"
            f"  {' '.join(installer)} install {' '.join(packages)}"
        )


def _setup_library_path_and_reexec() -> None:
    """Prepend pip-installed cuBLAS/cuDNN dirs to LD_LIBRARY_PATH and re-exec.

    CTranslate2 wheels expect cuBLAS and cuDNN to be discoverable by the Linux
    dynamic loader. When they are installed via pip inside a venv their directories
    must appear in LD_LIBRARY_PATH before the process starts. This function computes
    the correct path, updates the environment, and re-execs so the loader sees the
    libraries from the very first instruction of the next run.
    """
    if os.name != "posix":
        return

    library_dirs: list[str] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file:
                library_dirs.append(str(Path(module_file).resolve().parent))
        except ImportError:
            pass

    if not library_dirs:
        return

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = list(dict.fromkeys(library_dirs + current))
    if merged == current:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["LOCAL_TRANSCRIBER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def bootstrap_startup() -> None:
    """Prepare the runtime environment for the CLI entry point.

    Call exactly once at the top of ``cli.main()``. Steps:
    1. Detect GPU compute capability and CUDA version via nvidia-smi.
    2. If not inside a virtual environment, create .venv and re-exec into it.
    3. Install faster-whisper and matching CUDA libraries if absent.
    4. Re-exec with an updated LD_LIBRARY_PATH if pip-installed NVIDIA libraries
       need to be visible to the dynamic loader.
    """
    global _DETECTED_GPU_INFO

    _DETECTED_GPU_INFO = detect_gpu_info()

    if "--no-auto-install" not in sys.argv:
        _ensure_venv_and_reexec()

    already_ready = os.environ.get("LOCAL_TRANSCRIBER_LD_READY") == "1"
    if not already_ready and "--no-auto-install" not in sys.argv:
        ensure_dependencies(_DETECTED_GPU_INFO)

    _setup_library_path_and_reexec()
