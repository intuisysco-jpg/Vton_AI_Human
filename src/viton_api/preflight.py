"""Serial, cached checks of model paths and the configured Conda runtime."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import signal

from .config import Settings
from .diagnostics import safe_reason
from .models import STAGE_KEYS
from .runners import STAGES

logger = logging.getLogger("viton_api.preflight")
_TIMEOUT = 180


async def _kill_probe(proc) -> None:
    """Kill Conda and its Python child, then reap both output pipes."""
    if proc.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(proc.communicate(), timeout=5)
    except asyncio.TimeoutError:
        logger.warning("Timed out reaping preflight process")


class Preflight:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.availability: dict[str, bool] = {key: False for key in STAGE_KEYS}
        self.reasons: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._revision = 0

    @property
    def ready(self) -> bool:
        return bool(STAGE_KEYS) and all(
            self.availability.get(key, False) for key in STAGE_KEYS
        )

    async def probe_all(self) -> dict[str, bool]:
        # Overlapping requests share a result instead of starting parallel imports.
        revision = self._revision
        async with self._lock:
            if revision != self._revision:
                return dict(self.availability)
            availability, reasons = {}, {}
            for spec in STAGES:
                script = Path(self.settings.runners_dir) / spec.runner_script
                cwd = Path(getattr(self.settings, spec.tree_attr))
                if not script.is_file():
                    ok, reason = False, f"missing runner script: {script}"
                elif not cwd.is_dir():
                    ok, reason = False, f"missing stage directory: {cwd}"
                else:
                    missing = next(
                        (
                            path
                            for path in spec.extra_probe_paths(self.settings)
                            if not path.exists()
                        ),
                        None,
                    )
                    if missing is not None:
                        ok, reason = False, f"missing model: {missing}"
                    else:
                        ok, reason = await self._probe_stage(spec)
                availability[spec.key] = ok
                reasons[spec.key] = safe_reason(reason, self.settings)
                logger.info(
                    "preflight %s env=%s available=%s %s",
                    spec.key,
                    self.settings.conda_env,
                    ok,
                    reasons[spec.key],
                )
            # Publish a complete snapshot; health readers never observe a half-probe.
            self.availability = availability
            self.reasons = reasons
            self._revision += 1
            return dict(availability)

    async def _probe_stage(self, spec) -> tuple[bool, str]:
        command = self.settings.conda_command("python", "-c", spec.probe_imports)
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(getattr(self.settings, spec.tree_attr)),
                env=self.settings.conda_subprocess_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **({"start_new_session": True} if os.name == "posix" else {}),
            )
        except OSError as exc:
            return False, f"could not start Conda probe: {type(exc).__name__}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_TIMEOUT
            )
        except asyncio.TimeoutError:
            await _kill_probe(proc)
            return False, "probe timeout"
        except asyncio.CancelledError:
            await _kill_probe(proc)
            raise
        if proc.returncode == 0:
            return True, ""
        tail = (
            (stderr or stdout or b"")
            .decode("utf-8", errors="replace")
            .strip()
            .splitlines()[-3:]
        )
        return False, "probe failed: " + " | ".join(tail or ["no output"])
