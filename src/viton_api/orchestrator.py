"""Pipeline orchestration.

Each job runs EXACTLY one stage (selected by job_type). The orchestrator:
- picks the stage, checks preflight availability + cancel,
- writes <workdir>/job.json (the runner contract),
- runs the `conda run` subprocess (merged stdout/stderr -> job.log) with the
  stage's cwd and proportional timeout,
- pumps the runner's JSONL progress events into an async callback
  ({"event": "image_done"|"image_generated"|"image_failed", ...}),
- raises StageFailed carrying the runner's exit code (the consumer routes
  2 = poison / 3 = OOM / anything else through the retry budget) or
  JobCancelled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Awaitable, Callable

from .config import Settings
from .logging_setup import JobLog
from .models import JOB_TYPE_TO_STAGE, STAGE_PROGRESS
from .runners import STAGE_BY_KEY, StageContext, StageSpec
from .store import PostgresStore

logger = logging.getLogger("viton_api.orchestrator")

# Seconds to wait after a graceful kill (SIGTERM) before force-killing (SIGKILL)
# a stage subprocess tree that refuses to exit.
_TERM_GRACE_SECONDS = 10

CancelCheck = Callable[[], Awaitable[bool]]
EventCallback = Callable[[dict], Awaitable[None]]


def _stage_popen_kwargs() -> dict[str, object]:
    """Start each stage subprocess in its own process group / session.

    A stage is launched as ``conda run … python …``: ``conda run`` is the process
    we hold (``proc``) and the real model runs in a ``python`` CHILD of it.
    Terminating only ``proc`` can orphan that child (the GPU work keeps running,
    holding VRAM). ``start_new_session=True`` makes ``proc`` a session+group
    leader, so ``os.killpg`` reaches every descendant.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_stage_tree(pid: int, force: bool) -> None:
    """Kill a stage subprocess tree (conda run -> python -> CUDA). Best-effort:
    a process that already exited is not an error."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _sample_tree_cpu(pgid: int) -> int | None:
    """Sum utime+stime (clock ticks) of every process in the stage's process
    group. Returns None if the group can't be read — fail-open, so a /proc hiccup
    can never cause a false stall kill."""
    total = 0
    found = False
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as fh:
                    data = fh.read()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            rpar = data.rfind(b")")
            if rpar < 0:
                continue
            fields = data[rpar + 2:].split()
            if len(fields) < 13:
                continue
            try:
                if int(fields[2]) != pgid:
                    continue
                total += int(fields[11]) + int(fields[12])
                found = True
            except ValueError:
                continue
    except Exception:  # noqa: BLE001 — /proc read failure is non-fatal
        return None
    return total if found else None


class _Liveness:
    """Mutable stall-probe state shared between ``_pump`` (stdout) and the poll
    loop (GPU util + process-tree CPU). A signal is 'alive' if its timestamp
    moved within ``stage_stall_seconds``; the stage is stalled only when all
    three are stale, confirmed across ``stall_debounce_ticks`` ticks."""

    __slots__ = ("last_stdout", "last_gpu", "last_cpu", "prev_cpu", "dead_ticks")

    def __init__(self, now: float) -> None:
        self.last_stdout = now
        self.last_gpu = now
        self.last_cpu = now
        self.prev_cpu: int | None = None
        self.dead_ticks = 0


class JobCancelled(Exception):
    """Raised when cancel_requested is set before/during the stage."""


class StageFailed(Exception):
    """A stage exited non-zero, timed out, stalled, or was unavailable.

    ``exit_code`` is the runner's exit code when the stage RAN and exited
    non-zero (2 = poison, 3 = CUDA OOM); None for timeouts/stalls/unavailable
    stages (the consumer then falls back to the retry budget).
    """

    def __init__(self, stage: str, message: str, exit_code: int | None = None):
        self.stage = stage
        self.message = message
        self.exit_code = exit_code
        super().__init__(f"[{stage}] {message}")


class Orchestrator:
    def __init__(self, settings: Settings, store: PostgresStore, preflight, gpu_reporter=None):
        self.settings = settings
        self.store = store
        self.preflight = preflight
        # GPU reporter feeds the stall probe's utilization signal. None (no pool)
        # => the probe fail-opens on the GPU axis (stdout + CPU still guard).
        self.gpu_reporter = gpu_reporter
        # Set post-connect by the consumer. None => all progress emits are no-ops.
        self.progress_publisher = None

    async def _emit(self, reference: str | None, batch_id: str | None, **fields) -> None:
        """Best-effort progress emission; no-op until the consumer wires a publisher."""
        pub = self.progress_publisher
        if pub is None:
            return
        await pub.emit(reference, batch_id, fields)

    def _effective_timeout(self, job_type: str, n_images: int) -> float:
        """Proportional wall-clock backstop: scales with image count so a large
        job can finish, but never exceeds the broker ceiling (cap). ``max()``
        keeps small jobs at their generous flat budget; the stall probe is the
        real hang-killer."""
        base = self.settings.timeout_base_for(job_type)
        scaled = self.settings.per_image_s_for(job_type) * n_images + self.settings.stage_load_overhead_s
        cap = self.settings.stage_backstop_hard_cap_s
        return min(max(base, scaled), cap)

    async def run(self, job_id, reference: str, params: dict, units: list[dict],
                  workdir, job_type: str, *, batch_id: str | None = None,
                  on_event: EventCallback | None = None) -> None:
        """Run the single stage for one job. Raises StageFailed (with the
        runner's exit code when available) or JobCancelled."""
        try:
            stage_key = JOB_TYPE_TO_STAGE[job_type][0]
        except KeyError:
            raise StageFailed("dispatch", f"unsupported job_type {job_type!r}")
        spec = STAGE_BY_KEY[stage_key]
        ctx = StageContext.for_job(workdir, reference, params, units, self.settings)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        job_json = ctx.write_job_json()

        available = self.preflight.availability.get(spec.key, False)
        if not available and spec.required:
            reason = self.preflight.reasons.get(spec.key, "unavailable")
            raise StageFailed(spec.key, f"stage unavailable in preflight ({reason})")

        joblog = JobLog(ctx.workdir / "job.log")
        joblog.open()
        try:
            if await self.store.is_cancel_requested(job_id):
                raise JobCancelled()
            await self.store.set_stage_progress(job_id, spec.key, STAGE_PROGRESS[spec.key])
            await self.store.add_event(job_id, spec.key, "started")
            await self._emit(reference, batch_id, status="running", current_stage=spec.key,
                             progress=STAGE_PROGRESS[spec.key],
                             message=f"{spec.key} started")

            async def cancel_check() -> bool:
                return await self.store.is_cancel_requested(job_id)

            timeout = self._effective_timeout(job_type, len(units))
            cmd = spec.build_cmd(ctx)
            await self._run_cmd(spec, cmd, spec._cwd(ctx), joblog, timeout,
                                cancel_check, on_event=on_event)
            await self.store.add_event(job_id, spec.key, "done")
            await self._emit(reference, batch_id, status="running", current_stage=spec.key,
                             progress=STAGE_PROGRESS[spec.key], message=f"{spec.key} done")
        finally:
            joblog.close()

    # ------------------------------------------------------------------ #
    async def _run_cmd(
        self,
        spec: StageSpec,
        cmd: list[str],
        cwd: str,
        joblog: JobLog,
        timeout,
        cancel_check: CancelCheck | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> None:
        """Run one conda subprocess, pump merged stdout/stderr to job.log, and
        raise StageFailed on non-zero exit/timeout — or JobCancelled when
        cancel_check() reports the job was cancelled mid-stage.

        JSONL event lines the runner prints ({"event": ...}) are dispatched to
        ``on_event``; callback errors are swallowed (logged) so a DB hiccup can
        never kill the pump. The whole subprocess tree (conda run -> python ->
        CUDA) is killed on cancel/timeout.
        """
        env = self.settings.conda_subprocess_env()
        logger.info("running stage %s: %s (cwd=%s)", spec.key, " ".join(cmd), cwd)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merged -> job.log
            **_stage_popen_kwargs(),
        )
        try:
            stage_pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            stage_pgid = None

        liv = _Liveness(time.monotonic())
        pump_task = asyncio.create_task(
            self._pump(proc, joblog, liv, on_event), name="stage-pump"
        )
        poll = self.settings.cancel_poll_interval
        needs_poll = bool(poll and poll > 0)
        poll_enabled = cancel_check is not None and needs_poll
        deadline = None if not timeout else time.monotonic() + timeout
        stall_s = self.settings.stage_stall_seconds
        try:
            while True:
                if deadline is None:
                    remaining = None
                else:
                    remaining = max(0.0, deadline - time.monotonic())
                if needs_poll:
                    slice_ = poll if remaining is None else min(poll, remaining)
                else:
                    slice_ = remaining

                done, _ = await asyncio.wait({pump_task}, timeout=slice_)
                if pump_task in done:
                    break  # subprocess closed stdout -> it is exiting

                if poll_enabled and await cancel_check():
                    await self._terminate_proc(proc, spec, "CANCELLED")
                    raise JobCancelled()
                if needs_poll and await self._probe_stalled(proc, spec, liv, stage_pgid, stall_s):
                    await self._terminate_proc(
                        proc, spec, f"STALLED after {stall_s:.0f}s with no progress"
                    )
                    raise StageFailed(
                        spec.key,
                        f"stalled after {stall_s:.0f}s with no progress; see job.log",
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    await self._terminate_proc(proc, spec, f"TIMEOUT after {timeout}s")
                    raise StageFailed(spec.key, f"timed out after {timeout}s; see job.log")
        finally:
            if not pump_task.done():
                pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

        rc = await proc.wait()
        if rc != 0:
            raise StageFailed(spec.key, f"exit code {rc}; see job.log", exit_code=rc)

    async def _probe_stalled(
        self,
        proc: asyncio.subprocess.Process,
        spec: StageSpec,
        liv: "_Liveness",
        pgid: int | None,
        stall_s: float,
    ) -> bool:
        """True if the stage looks hung: no sign of life on ANY of stdout, GPU
        utilization, or process-tree CPU for ``stall_s`` seconds, confirmed across
        ``stall_debounce_ticks`` ticks. Each axis fail-opens on telemetry gaps."""
        now = time.monotonic()
        stdout_dead = (now - liv.last_stdout) > stall_s

        try:
            util = await self.gpu_reporter.utilization() if self.gpu_reporter else None
        except Exception:  # noqa: BLE001
            util = None
        if util is None:
            gpu_dead = False  # unknown -> don't count against stall
        elif util >= self.settings.gpu_active_util_pct:
            liv.last_gpu = now
            gpu_dead = False
        else:
            gpu_dead = (now - liv.last_gpu) > stall_s

        try:
            ticks = _sample_tree_cpu(pgid) if pgid else None
        except Exception:  # noqa: BLE001
            ticks = None
        if ticks is None:
            cpu_dead = False
        elif liv.prev_cpu is None or ticks != liv.prev_cpu:
            liv.prev_cpu = ticks
            liv.last_cpu = now
            cpu_dead = False
        else:
            cpu_dead = (now - liv.last_cpu) > stall_s

        if stdout_dead and gpu_dead and cpu_dead:
            liv.dead_ticks += 1
            return liv.dead_ticks >= self.settings.stall_debounce_ticks
        liv.dead_ticks = 0
        return False

    async def _pump(
        self,
        proc: asyncio.subprocess.Process,
        joblog: JobLog,
        liv: "_Liveness",
        on_event: EventCallback | None = None,
    ) -> None:
        """Stream the stage's merged output into job.log, dispatching JSONL
        event lines to ``on_event`` (best-effort: a malformed line or a failing
        callback must never break the pump)."""
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            joblog.write(text)
            liv.last_stdout = time.monotonic()  # any stdout => alive
            stripped = text.rstrip("\n")
            if not stripped:
                continue
            if stripped.lstrip().startswith("{"):
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, dict) and "event" in event and on_event is not None:
                    try:
                        await on_event(event)
                    except Exception:  # noqa: BLE001 — non-critical, keep pumping
                        logger.warning("event callback failed for %s", event.get("event"),
                                       exc_info=True)
                    continue
            logger.info("  | %s", stripped)

    async def _terminate_proc(
        self, proc: asyncio.subprocess.Process, spec: StageSpec, reason: str,
    ) -> None:
        """Kill the whole stage subprocess tree: graceful, then force after a
        grace period. Kills the process GROUP so the python child under
        ``conda run`` — the one actually holding the GPU — stops too."""
        if proc.returncode is not None:
            return
        logger.info("[%s] %s; killing stage subprocess tree", spec.key, reason)
        await asyncio.to_thread(_kill_stage_tree, proc.pid, False)  # graceful SIGTERM
        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERM_GRACE_SECONDS)
        except asyncio.TimeoutError:
            await asyncio.to_thread(_kill_stage_tree, proc.pid, True)  # force SIGKILL
            await proc.wait()
