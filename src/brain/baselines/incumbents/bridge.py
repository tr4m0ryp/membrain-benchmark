"""Run an incumbent inside its own virtual environment, JSON in and JSON out.

The incumbents pull dependency stacks this project cannot host: Mem0 2.0.15
wants numpy 2 where this project is pinned below it, and Graphiti pulls a
different pydantic line again. Installing any of them here would silently change
the environment every existing number was measured in, so each incumbent lives
in its own venv and is driven as a subprocess.

That isolation is also what makes the provenance honest: the result file records
the interpreter, the resolved package versions and the exact command, and none
of it can be confused with this project's own stack.

Venv locations are read from the environment so the worker never assumes a
layout:

    BRAIN_INCUMBENT_VENV_MEM0        Mem0 2.x
    BRAIN_INCUMBENT_VENV_MEM0_LEGACY Mem0 0.1.x
    BRAIN_INCUMBENT_VENV_GRAPHITI    graphiti-core
    BRAIN_INCUMBENT_VENV_NOUS        nous-state
    BRAIN_INCUMBENT_VENV_TOKI        the TOKI checkout ('bitemporal')
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["VenvMissing", "WorkerFailed", "Venv", "venv_for", "run_worker", "WORKERS"]

WORKER_ROOT = Path(__file__).resolve().parent / "workers"

#: Probe entry points, by worker name.
WORKERS = {
    "mem0": WORKER_ROOT / "mem0" / "probe.py",
    "graphiti": WORKER_ROOT / "graphiti" / "probe.py",
    "nous": WORKER_ROOT / "nous" / "probe.py",
    "toki": WORKER_ROOT / "toki" / "probe.py",
}

_ENV_VARS = {
    "mem0": "BRAIN_INCUMBENT_VENV_MEM0",
    "mem0_legacy": "BRAIN_INCUMBENT_VENV_MEM0_LEGACY",
    "graphiti": "BRAIN_INCUMBENT_VENV_GRAPHITI",
    "nous": "BRAIN_INCUMBENT_VENV_NOUS",
    "toki": "BRAIN_INCUMBENT_VENV_TOKI",
}


class VenvMissing(RuntimeError):
    """No interpreter for this incumbent. The caller reports BLOCKED, not zero."""


class WorkerFailed(RuntimeError):
    """The worker ran and failed. Its own error text is carried through."""


@dataclass(frozen=True, slots=True)
class Venv:
    """One incumbent interpreter and the packages it resolved to."""

    system: str
    python: Path
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "python": str(self.python),
            "packages": dict(self.packages),
        }


def _freeze(python: Path, names: tuple[str, ...]) -> dict[str, str]:
    """Resolved versions of the packages a result must be attributed to."""
    script = (
        "import json\n"
        "from importlib.metadata import version, PackageNotFoundError\n"
        f"names = {list(names)!r}\n"
        "out = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        out[name] = version(name)\n"
        "    except PackageNotFoundError:\n"
        "        out[name] = None\n"
        "print(json.dumps(out))\n"
    )
    try:
        done = subprocess.run(
            [str(python), "-c", script], capture_output=True, text=True, timeout=180, check=True
        )
        return json.loads(done.stdout.strip().splitlines()[-1])
    except Exception:
        return {}


#: Packages whose versions are recorded per incumbent.
_PINNED = {
    "mem0": ("mem0ai", "sentence-transformers", "faiss-cpu", "spacy", "numpy", "openai"),
    "mem0_legacy": ("mem0ai", "sentence-transformers", "faiss-cpu", "numpy", "openai"),
    "graphiti": ("graphiti-core", "pydantic", "neo4j", "openai"),
    "nous": ("nous-state",),
    "toki": ("bitemporal", "duckdb"),
}


def venv_for(system: str) -> Venv:
    """The interpreter for `system`, or `VenvMissing` naming what is absent."""
    variable = _ENV_VARS.get(system)
    if variable is None:
        raise VenvMissing(f"unknown incumbent {system!r}; known: {sorted(_ENV_VARS)}")
    root = os.environ.get(variable)
    if not root:
        raise VenvMissing(f"{system}: set {variable} to a virtual environment containing it")
    python = Path(root) / "bin" / "python"
    if not python.is_file():
        python = Path(root)  # the variable may name the interpreter directly
    if not python.is_file():
        raise VenvMissing(f"{system}: no interpreter at {root}")
    return Venv(system=system, python=python, packages=_freeze(python, _PINNED.get(system, ())))


def run_worker(
    worker: str,
    system: str,
    spec: Mapping[str, Any],
    *,
    workdir: Path,
    timeout: float = 7200.0,
) -> tuple[dict[str, Any], Venv, dict[str, Any]]:
    """Run one probe. Returns (payload, venv, invocation record)."""
    script = WORKERS.get(worker)
    if script is None or not script.is_file():
        raise VenvMissing(f"no worker script for {worker!r} at {script}")
    venv = venv_for(system)
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "input.json"
    destination = workdir / "output.json"
    source.write_text(json.dumps(dict(spec), indent=2, sort_keys=True), encoding="utf-8")

    command = [str(venv.python), str(script), str(source), str(destination)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    record = {
        "command": command,
        "returncode": done.returncode,
        "stderr_tail": done.stderr[-2000:] if done.stderr else "",
    }
    if not destination.is_file():
        raise WorkerFailed(f"{worker}: worker wrote no output. stderr: {done.stderr[-2000:]}")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise WorkerFailed(f"{worker}: {payload.get('error')}\n{payload.get('traceback', '')[-2000:]}")
    return payload, venv, record
