from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_resolved_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


def git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def runtime_metadata(device: torch.device, repository_root: Path) -> dict[str, Any]:
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repository_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def save_model(path: Path, model: torch.nn.Module, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": model.state_dict(), **(extra or {})}, temporary)
    temporary.replace(path)

