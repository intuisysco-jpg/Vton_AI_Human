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
import logging
import socket

from .config import Settings

logger = logging.getLogger("viton_api.gpu_status")


class GpuStatusReporter:
    def __init__(self, pool, settings: Settings, app):
        self.pool = pool
        self.settings = settings
        self.app = app
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._handle = None
        self._nvml_ok = False

    async def start(self) -> None:
        await self._check_table()
        self._init_nvml()
        self._task = asyncio.create_task(self._beat_loop(), name="gpu-status")

    def _init_nvml(self) -> None:
        try:
            import pynvml  # noqa: lazy import
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
            logger.info("NVML initialised; gpu_id=%s:0", socket.gethostname())
        except Exception as e:  # noqa: BLE001 — any NVML failure is non-fatal
            self._nvml_ok = False
            self._handle = None
            logger.warning("NVML unavailable (gpu status will be null): %s", e)

    async def _check_table(self) -> None:
        """Check the migration-owned table without requesting DDL privileges."""
        await self.pool.execute("SELECT gpu_id, total_mb, used_mb, free_mb, utilization, current_job, reported_at FROM public.gpu_status LIMIT 0")

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
        gpu_id = f"{socket.gethostname()}:0"
        current_job = getattr(self.app.state, "current_job", None)

        if not self._nvml_ok:
            # no-op beat; /health will show gpu: null
            self.app.state.gpu = None
            return

        reading = await asyncio.to_thread(self._read_nvml)
        reading["gpu_id"] = gpu_id
        reading["current_job"] = current_job
        reading["reported_at"] = "now"

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
        }

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
