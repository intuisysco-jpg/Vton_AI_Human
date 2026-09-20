"""The unified progress ladder: one formula for DB writes and SSE frames.

``image_pct`` maps done/n into the stage window (50 -> 100); every DB write is
monotonic (GREATEST); per-image SSE frames carry ``pct`` (the central SSE
contract names a frame ``progress`` only when it does) and report the SAME
post-increment ``done`` the DB write used. Completed always reads 100.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from viton_api.consumer import AckDecision, JobProcessor
from viton_api.models import STAGE_PROGRESS, image_pct


def test_image_pct_maps_into_stage_window() -> None:
    base = STAGE_PROGRESS["viton"]
    seq = [image_pct(done, 4) for done in range(5)]
    assert seq[0] == base
    assert seq[-1] == 100
    assert seq == sorted(seq)
    assert image_pct(0, 0) == 100


class FakeStore:
    def __init__(self) -> None:
        self.recorded: list[tuple[int, int]] = []   # (ordinal, done)
        self.completed: list[int] = []              # progress values
        self.events: list[tuple] = []

    async def get_job_by_reference(self, reference):
        return {"id": "job-x", "reference": reference, "status": "queued",
                "retry_count": 0, "image_count": 2, "params": "{}",
                "batch_id": "batch-1"}

    async def get_input_files(self, job_id):
        return [
            {"role": "person", "file_path": "ref/in/p0.png", "ordinal": 0,
             "original_filename": "p0.png", "status": "queued"},
            {"role": "garment", "file_path": "ref/in/g0.png", "ordinal": 0,
             "original_filename": "g0.png", "status": "queued"},
            {"role": "person", "file_path": "ref/in/p1.png", "ordinal": 1,
             "original_filename": "p1.png", "status": "queued"},
            {"role": "garment", "file_path": "ref/in/g1.png", "ordinal": 1,
             "original_filename": "g1.png", "status": "queued"},
        ]

    async def get_completed_ordinals(self, job_id):
        return set()

    async def is_cancel_requested(self, job_id):
        return False

    async def mark_running(self, job_id):
        pass

    async def add_event(self, job_id, stage, message):
        self.events.append((job_id, stage, message))

    async def record_image_output(self, job_id, ordinal, rel_path, size, mime,
                                  image_count, done, original_filename=None):
        self.recorded.append((ordinal, done))

    async def mark_image_failed(self, job_id, ordinal):
        pass

    async def mark_completed(self, job_id, progress=100):
        self.completed.append(progress)


class FakeOrchestrator:
    """run() plays the runner's event stream: per image a generated event,
    then image_done with a real produced file (the consumer copies it)."""

    def __init__(self, workroot: Path):
        self.workroot = workroot
        self.emits: list[tuple] = []

    async def _emit(self, reference, batch_id, **fields):
        self.emits.append((reference, batch_id, fields))

    async def run(self, job_id, reference, params, units, workdir, job_type,
                  *, batch_id=None, on_event=None, progress_floor=0):
        for u in units:
            await on_event({"event": "image_generated", "ordinal": u["ordinal"]})
            produced = Path(workdir) / f"{u['ordinal']:04d}.png"
            produced.write_bytes(b"png")
            await on_event({"event": "image_done", "ordinal": u["ordinal"],
                            "path": str(produced)})


async def test_on_event_db_and_sse_agree_and_never_regress(tmp_path):
    storage = tmp_path / "storage"
    for rel in ("ref/in/p0.png", "ref/in/g0.png", "ref/in/p1.png", "ref/in/g1.png"):
        p = storage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    store = FakeStore()
    orch = FakeOrchestrator(tmp_path)
    proc = JobProcessor(
        SimpleNamespace(work_root=str(tmp_path / "work"),
                        storage_root=str(storage), max_retries=3,
                        gpu_lock_path=str(tmp_path / "gpu.lock"),
                        gpu_lock_timeout=0.0, cancel_poll_interval=0.01),
        store, orch, None,
    )
    # No GPU in tests: skip the lease and the settle/free gate.
    proc._await_gpu_free = AsyncMock(return_value=None)
    proc._record_exec_start = AsyncMock()
    proc._record_exec_ms = AsyncMock()

    decision = await proc.process("ref", "viton")

    assert decision is AckDecision.ACK
    # DB writes use the POST-increment count: (ordinal, done) = (0, 1), (1, 2)
    assert store.recorded == [(0, 1), (1, 2)]
    # ...and completion reads full
    assert store.completed == [100]

    # SSE: every running/completed frame carries pct; per-image finalize
    # frames carry the job's batch_id (batch watchers see the ticks); the
    # whole sequence is non-decreasing and ends at 100.
    frames = [f for _, b, f in orch.emits]
    pcts = [f["pct"] for f in frames if "pct" in f]
    assert pcts, "expected pct-bearing frames"
    assert pcts == sorted(pcts), f"progress regressed: {pcts}"
    assert pcts[-1] == 100
    per_image = [f for f in frames if str(f.get("message", "")).endswith("done")]
    assert per_image and all(b == "batch-1" for _, b, f in orch.emits
                             if f in per_image)
    assert [f["pct"] for f in per_image] == [
        image_pct(1, 2), image_pct(2, 2)]
