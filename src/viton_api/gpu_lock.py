"""Cross-container GPU mutex via flock on the shared /data mount.

The GPU workers bind-mount the same local ext4 path, so a ``flock`` on
``/data/gpu-coord/gpu.lock`` is the same kernel inode in every container ->
true mutual exclusion, auto-released on process death, no network round-trips,
stdlib only (no image rebuild). This is the *authoritative* gate; the NVML
free-memory gate stays as a cheap pre-filter and a safety net for any
non-participating GPU tenant.

The lock is held only around the GPU stage itself (see consumer), not during
input staging, so the other worker is not blocked while a job downloads inputs.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("viton_api.gpu_lock")


class GpuLease:
    """Exclusive flock over the GPU lock file for the duration of a stage.

    Usage::

        lease = GpuLease(settings)
        try:
            await lease.acquire()
        except TimeoutError:
            return AckDecision.NACK_REQUEUE   # treat like a busy GPU
        try:
            ... GPU stage ...
        finally:
            lease.release()
    """

    def __init__(self, settings) -> None:
        self.path = Path(settings.gpu_lock_path)
        self.timeout = float(settings.gpu_lock_timeout)
        # Reuse the cancel-poll cadence for the non-blocking acquire retry loop.
        self.poll_interval = float(settings.cancel_poll_interval) or 1.0
        self._fd: int | None = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    async def acquire(self) -> None:
        """Take LOCK_EX. If ``gpu_lock_timeout > 0`` poll non-blockingly and
        raise ``TimeoutError`` if not acquired in time; otherwise block (in a
        worker thread so the event loop stays responsive)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if self.timeout and self.timeout > 0:
                deadline = time.monotonic() + self.timeout
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"GPU lock not acquired within {self.timeout:.0f}s"
                            )
                        await asyncio.sleep(self.poll_interval)
            else:
                # Blocking flock off-thread so heartbeats / cancel polls keep running.
                await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        logger.info("GPU lease acquired (lock=%s)", self.path)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        finally:
            self._fd = None
            logger.info("GPU lease released (lock=%s)", self.path)

    async def __aenter__(self) -> "GpuLease":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
