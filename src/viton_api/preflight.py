"""Preflight: can each stage's conda env import what it needs?

For each stage: the runner script must exist (in <service>/runners/), the ML
tree's model paths must resolve (extra_probe_paths — exists() follows symlinks,
so a moved checkpoint is caught here), and ONE `conda run -n refton python -c
"<imports>"` probe per stage must pass with its own cwd (the ML trees have
strong cwd assumptions). Probes run serially to avoid OOM; results are cached.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .config import Settings
from .models import STAGE_KEYS
from .runners import STAGES

logger = logging.getLogger("viton_api.preflight")
_TIMEOUT = 180  # seconds per probe


class Preflight:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.availability: dict[str, bool] = {k: False for k in STAGE_KEYS}
        self.reasons: dict[str, str] = {}

    async def probe_all(self) -> dict[str, bool]:
        # 1. per-stage: does the runner script exist? do the model paths?
        for spec in STAGES:
            script = Path(self.settings.runners_dir) / spec.runner_script
            if not script.exists():
                self.availability[spec.key] = False
                self.reasons[spec.key] = f"missing runner script: {script}"
                continue
            missing_model = next(
                (str(p) for p in spec.extra_probe_paths(self.settings) if not p.exists()),
                None,
            )
            if missing_model:
                self.availability[spec.key] = False
                self.reasons[spec.key] = f"missing model: {missing_model}"
            else:
                self.reasons[spec.key] = ""  # tentative; import probe decides

        # 2. one import probe per still-unresolved stage (serial: both stages
        #    share the refton env; parallel probes would stack CUDA context).
        for spec in STAGES:
            if self.reasons.get(spec.key):  # already failed (missing script/model)
                continue
            await self._probe_stage(spec)

        for spec in STAGES:
            logger.info(
                "preflight %-10s env=%-8s available=%s %s",
                spec.key, getattr(self.settings, spec.env_attr),
                self.availability[spec.key], self.reasons.get(spec.key, ""),
            )
        return dict(self.availability)

    async def _probe_stage(self, spec) -> None:
        cwd = str(getattr(self.settings, spec.tree_attr))
        cmd = [
            "conda", "run", "--no-capture-output", "-n",
            getattr(self.settings, spec.env_attr),
            "python", "-c", spec.probe_imports,
        ]
        env_ = self.settings.conda_subprocess_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, env=env_,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            self._mark(spec, False, "conda not on PATH")
            return
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            self._mark(spec, False, "probe timeout")
            return
        if proc.returncode == 0:
            self._mark(spec, True, "")
        else:
            tail = (stderr or stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
            tail = tail[-3:] if tail else ["(no output)"]
            self._mark(spec, False, "probe failed: " + " | ".join(tail)[:300])

    def _mark(self, spec, ok: bool, reason: str) -> None:
        self.availability[spec.key] = ok
        if reason:
            self.reasons[spec.key] = reason
