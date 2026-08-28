"""Provenance capture and content addressing.

Requirement LC-5 (one-command reproduction) is unenforceable without a record
of what was actually run. This module produces that record and hashes it, so a
claim and the environment that produced it are bound together and any later
edit to either is detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, tight separators, stable float repr.

    Content hashes are only meaningful if serialisation is stable, so every
    hash in the package routes through this one function.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_fallback
    )


def _fallback(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def content_hash(obj: Any, *, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(canonical_json(obj).encode("utf-8"))
    return f"{algo}:{h.hexdigest()}"


def file_hash(path: str | os.PathLike, *, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_state(repo: str | os.PathLike | None = None) -> dict[str, Any]:
    """Commit, dirtiness and remote of the repository that produced a result.

    A dirty tree is not an error, but the validator downgrades any claim
    produced from one: the commit no longer identifies the code that ran.
    """
    cwd = str(repo) if repo else os.getcwd()
    exe = shutil.which("git")

    def g(*args: str) -> str | None:
        if exe is None:
            return None
        try:
            r = subprocess.run(
                [exe, "-C", cwd, *args], capture_output=True, text=True, timeout=20, check=False
            )
        except Exception:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    status = g("status", "--porcelain")
    return {
        "commit": g("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": g("config", "--get", "remote.origin.url"),
        "describe": g("describe", "--always", "--dirty", "--tags"),
    }


def gpu_state() -> dict[str, Any]:
    """Accelerator inventory, from torch when importable and nvidia-smi otherwise."""
    info: dict[str, Any] = {"available": False, "devices": [], "source": None}

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["available"] = True
            info["source"] = "torch"
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "index": i,
                        "name": p.name,
                        "capability": f"{p.major}.{p.minor}",
                        "total_memory_gb": round(p.total_memory / 1e9, 3),
                        "multi_processor_count": p.multi_processor_count,
                    }
                )
            return info
    except Exception:
        pass

    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        info["available"] = True
        info["source"] = "nvidia-smi"
        for i, line in enumerate(smi.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["devices"].append(
                    {
                        "index": i,
                        "name": parts[0],
                        "total_memory_gb": round(float(parts[1]) / 1024, 3),
                        "driver": parts[2] if len(parts) > 2 else None,
                    }
                )
    return info


_TRACKED_PACKAGES = (
    "torch",
    "triton",
    "numpy",
    "vllm",
    "sglang",
    "tensorrt_llm",
    "transformers",
    "flash_attn",
)


def package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str | None]:
    import importlib
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for n in names:
        v: str | None
        try:
            v = md.version(n)
        except Exception:
            try:
                v = getattr(importlib.import_module(n), "__version__", None)
            except Exception:
                v = None
        out[n] = v
    return out


_ENV_PREFIXES = ("CUDA_", "NCCL_", "TORCH_", "OMP_", "VLLM_", "SGLANG_", "TRTLLM_", "LC_")


@dataclass
class Provenance:
    """Everything needed to argue that a number came from where it claims to."""

    captured_at: str = field(default_factory=utcnow)
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    cpu_count: int | None = field(default_factory=os.cpu_count)
    git: dict[str, Any] = field(default_factory=git_state)
    gpu: dict[str, Any] = field(default_factory=gpu_state)
    packages: dict[str, str | None] = field(default_factory=package_versions)
    container_image: str | None = field(
        default_factory=lambda: os.environ.get("LC_CONTAINER_IMAGE")
    )
    operator: str | None = field(default_factory=lambda: os.environ.get("LC_OPERATOR"))
    env_overrides: dict[str, str] = field(
        default_factory=lambda: {
            k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)
        }
    )
    notes: str | None = None

    def hardware_fingerprint(self) -> str:
        """Stable identifier for the machine shape, ignoring hostname and time.

        Two runs with the same fingerprint are comparable; two runs with
        different fingerprints must not be pooled, and the validator says so.
        """
        devs = [
            {k: d.get(k) for k in ("name", "capability", "total_memory_gb")}
            for d in self.gpu.get("devices", [])
        ]
        return content_hash(
            {
                "os": self.platform.split("-")[0],
                "devices": devs,
                "n_devices": len(devs),
                "cuda": self.gpu.get("cuda_version"),
            }
        )[7:23]

    def describe_hardware(self) -> str:
        devs = self.gpu.get("devices") or []
        if not devs:
            return f"CPU only ({self.cpu_count} cores)"
        name = devs[0].get("name", "unknown GPU")
        return f"{len(devs)}x {name}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hardware_fingerprint"] = self.hardware_fingerprint()
        d["hardware"] = self.describe_hardware()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

    @classmethod
    def capture(cls, **kw: Any) -> Provenance:
        return cls(**kw)
