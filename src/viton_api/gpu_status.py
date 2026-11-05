"""GPU-status heartbeat.

Writes one gpu_status row per box every `gpu_status_interval` seconds (idle or
busy), so /health always reports current free memory. NVML is accessed via
`nvidia-ml-py` (pynvml) — a pure-ctypes binding, no CUDA toolkit. All NVML calls
run in a worker thread (they are blocking syscalls). `import pynvml` is lazy so
the service still imports on GPU-less boxes.

Started whenever the DB pool exists — INDEPENDENT of the consumer gates — so
/health shows free memory even when preflight gates the consumer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from datetime import datetime, timezone

from .config import Settings

logger = logging.getLogger("viton_api.gpu_status")


def gpu_status_id() -> str:
    """gpu_status row key: GPU_STATUS_ID when set (one row per worker service
    sharing a GPU), else the legacy ``<hostname>:0``."""
    return os.environ.get("GPU_STATUS_ID", "").strip() or f"{socket.gethostname()}:0"


def physical_gpu_id() -> str:
    """Optional grouping key for heartbeat rows that share ONE physical GPU
    (all worker containers on this box). Empty string = unset (legacy row;
    central relays such rows ungrouped)."""
    return os.environ.get("GPU_PHYSICAL_ID", "").strip()


class GpuStatusReporter:
    def __init__(self, pool, settings: Settings, app):
        self.pool = pool
        self.settings = settings
        self.app = app
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._handle = None
        self._nvml_ok = False
        # True once the gpu_status table is known to carry the attribution
        # columns (processes/physical_id, migration 0023+). Until then the
        # heartbeat degrades to the legacy INSERT so beats keep flowing.
        self._extended = False

    async def start(self) -> None:
        await self._check_table()
        await asyncio.to_thread(self._init_nvml)
        self._task = asyncio.create_task(self._beat_loop(), name="gpu-status")

    def _init_nvml(self) -> None:
        try:
            import pynvml  # noqa: lazy import
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
            logger.info("NVML initialised; gpu_id=%s", gpu_status_id())
        except Exception as e:  # noqa: BLE001 — any NVML failure is non-fatal
            self._nvml_ok = False
            self._handle = None
            logger.warning("NVML unavailable (gpu status will be null): %s", e)

    async def _check_table(self) -> None:
        """Check the migration-owned table without requesting DDL privileges.

        Detects the optional attribution columns so the beat INSERT can fall
        back to the legacy column set on a not-yet-migrated database.
        """
        await self.pool.execute(
            "SELECT gpu_id, total_mb, used_mb, free_mb, utilization,"
            " current_job, reported_at FROM public.gpu_status LIMIT 0"
        )
        try:
            row = await self.pool.fetchrow(
                "SELECT COUNT(*) AS n FROM information_schema.columns"
                " WHERE table_schema = 'public' AND table_name = 'gpu_status'"
                " AND column_name IN ('processes', 'physical_id')"
            )
            self._extended = bool(row and row["n"] == 2)
        except Exception as e:  # noqa: BLE001 — attribution is optional
            self._extended = False
            logger.info("gpu_status attribution columns not detected: %s", e)

    async def _beat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._beat_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("gpu beat error: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.gpu_status_interval)
            except asyncio.TimeoutError:
                pass

    async def _beat_once(self) -> None:
        gpu_id = gpu_status_id()
        current_job = getattr(self.app.state, "current_job", None)

        if not self._nvml_ok:
            # no-op beat; /health will show gpu: null
            self.app.state.gpu = None
            return

        reading = await asyncio.to_thread(self._read_nvml)
        processes = (
            await asyncio.to_thread(self._read_processes)
            if self._extended
            else None
        )
        physical = physical_gpu_id() if self._extended else None
        reading["gpu_id"] = gpu_id
        reading["current_job"] = current_job
        reading["reported_at"] = "now"

        if self._extended:
            await self.pool.execute(
                """
                INSERT INTO public.gpu_status
                  (gpu_id, total_mb, used_mb, free_mb, utilization, current_job,
                   reported_at, processes, physical_id)
                VALUES ($1, $2, $3, $4, $5, $6, now(), $7::jsonb, $8)
                ON CONFLICT (gpu_id) DO UPDATE SET
                  total_mb=EXCLUDED.total_mb, used_mb=EXCLUDED.used_mb, free_mb=EXCLUDED.free_mb,
                  utilization=EXCLUDED.utilization, current_job=EXCLUDED.current_job, reported_at=now(),
                  processes=EXCLUDED.processes, physical_id=EXCLUDED.physical_id
                """,
                gpu_id, reading["total_mb"], reading["used_mb"], reading["free_mb"],
                reading["utilization"], current_job,
                json.dumps(processes) if processes is not None else None,
                physical,
            )
        else:
            await self.pool.execute(
                """
                INSERT INTO public.gpu_status (gpu_id, total_mb, used_mb, free_mb, utilization, current_job, reported_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                ON CONFLICT (gpu_id) DO UPDATE SET
                  total_mb=EXCLUDED.total_mb, used_mb=EXCLUDED.used_mb, free_mb=EXCLUDED.free_mb,
                  utilization=EXCLUDED.utilization, current_job=EXCLUDED.current_job, reported_at=now()
                """,
                gpu_id, reading["total_mb"], reading["used_mb"], reading["free_mb"],
                reading["utilization"], current_job,
            )
        # cache for O(1) /health
        self.app.state.gpu = {
            "gpu_id": gpu_id,
            "total_mb": reading["total_mb"],
            "used_mb": reading["used_mb"],
            "free_mb": reading["free_mb"],
            "utilization": reading["utilization"],
            "current_job": current_job,
            "reported_at": datetime.now(timezone.utc).isoformat(),
            "processes": processes,
            "physical_id": physical,
        }

    def _read_processes(self) -> list[dict] | None:
        """Per-process VRAM snapshot for attribution, or None when unavailable.

        nvidia-smi is preferred (its process_name is the executable path, which
        names the owning conda env); pynvml is the fallback (pid + MB only, no
        name). Runs in a worker thread like the other NVML reads.
        """
        import subprocess
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if out.returncode == 0:
                procs = []
                for line in out.stdout.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3 and parts[0].isdigit():
                        procs.append(
                            {
                                "pid": int(parts[0]),
                                "name": parts[1],
                                "used_mb": int(parts[2]),
                            }
                        )
                return procs
        except Exception:  # noqa: BLE001 — fall through to the pynvml path
            pass
        try:
            import pynvml  # noqa: lazy import
            procs = []
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                procs.append(
                    {
                        "pid": p.pid,
                        "name": None,
                        "used_mb": int(p.usedMem // (1024 * 1024)),
                    }
                )
            return procs
        except Exception:  # noqa: BLE001 — attribution stays optional
            return None

    def _read_nvml(self) -> dict:
        import pynvml  # noqa: lazy
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        return {
            "total_mb": int(mem.total // (1024 * 1024)),
            "used_mb": int(mem.used // (1024 * 1024)),
            "free_mb": int(mem.free // (1024 * 1024)),
            "utilization": int(util.gpu),
        }

    async def used_mb(self) -> int | None:
        """Fresh on-demand NVML used_mb. None if NVML unavailable.

        Used by the GPU-busy gate (consumer) so the decision reflects *any*
        process on the box, not just this worker's own subprocess. Threaded like
        the heartbeat (NVML calls are blocking syscalls); concurrent reads from
        the gate + heartbeat are safe (NVML handles are thread-safe for queries).
        """
        if not self._nvml_ok or self._handle is None:
            return None
        return await asyncio.to_thread(lambda: self._read_nvml()["used_mb"])

    async def free_mb(self) -> int | None:
        """Fresh on-demand NVML free_mb. None if NVML unavailable.

        Used by the batch-size decision (consumer): how many images can be
        stacked into one stage run given the headroom RIGHT NOW, after the GPU
        mutex is held. Same threading/concurrency guarantees as :meth:`used_mb`.
        """
        if not self._nvml_ok or self._handle is None:
            return None
        return await asyncio.to_thread(lambda: self._read_nvml()["free_mb"])

    async def utilization(self) -> int | None:
        """Fresh on-demand NVML GPU utilization % (0-100). None if unavailable.

        Used by the orchestrator's stall probe. With the cross-container GPU
        mutex held by this worker, only this worker's stage can be driving the
        GPU, so utilization reliably reflects the running stage.
        """
        if not self._nvml_ok or self._handle is None:
            return None
        return await asyncio.to_thread(lambda: self._read_nvml()["utilization"])

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._nvml_ok:
            try:
                import pynvml  # noqa
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
