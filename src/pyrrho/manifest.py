"""manifest.py — Capture the full reproducibility context of a training/eval run.

Every artifact pyrrho produces should be paired with a manifest.json that lets
someone re-create the same result from scratch. Captures:
    - git commit hash + dirty state of pyrrho + fitz-gov
    - python version + full pip freeze
    - hardware (GPU model, VRAM, CPU)
    - seed used
    - the exact config file (snapshot, not just path)
    - start/end timestamps + wall-clock duration
    - dataset commit (fitz-gov data version)
    - free-form `extra` dict for run-specific context
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_state(repo: Path) -> dict[str, Any]:
    """Capture git HEAD + dirty state of `repo`. Returns dict with `commit`, `dirty`, `branch`."""
    repo = Path(repo)
    commit = _run_git(["rev-parse", "HEAD"], repo)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _run_git(["status", "--porcelain"], repo)
    dirty = bool(status) if status is not None else None
    return {
        "repo": str(repo),
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "dirty_files": status.splitlines() if status else [],
    }


def pip_freeze() -> list[str]:
    """Full pip freeze of the active environment. One package per line."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return []
        return [line for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def hardware_info() -> dict[str, Any]:
    """GPU model, VRAM, CPU, OS — enough to know the box. Best-effort, never raises."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        info["torch_probe_error"] = repr(exc)

    return info


def snapshot_config(config_path: Path) -> dict[str, Any]:
    """Read the config file verbatim. Stored as raw text + parsed yaml for downstream tooling."""
    try:
        import yaml

        raw = config_path.read_text(encoding="utf-8")
        return {"path": str(config_path), "raw": raw, "parsed": yaml.safe_load(raw)}
    except Exception as exc:
        return {"path": str(config_path), "error": repr(exc)}


def write_manifest(
    output_dir: Path,
    config_path: Path,
    seed: int,
    pyrrho_repo: Path | None = None,
    fitz_gov_repo: Path | None = None,
    extra: dict[str, Any] | None = None,
    start_time: float | None = None,
) -> Path:
    """Write `output_dir/manifest.json` capturing the full run context.

    Returns the path of the written manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.time()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(seed),
        "config": snapshot_config(Path(config_path)),
        "hardware": hardware_info(),
        "env": {
            "WANDB_DISABLED": os.environ.get("WANDB_DISABLED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "pip_freeze": pip_freeze(),
        "git": {
            "pyrrho": git_state(pyrrho_repo or Path.cwd()) if pyrrho_repo or Path.cwd() else None,
            "fitz_gov": git_state(fitz_gov_repo) if fitz_gov_repo else None,
        },
    }
    if start_time is not None:
        payload["timing"] = {
            "start_unix": start_time,
            "end_unix": end_time,
            "duration_seconds": round(end_time - start_time, 2),
        }
    if extra:
        payload["extra"] = extra

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return manifest_path
