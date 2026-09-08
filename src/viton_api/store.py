"""PostgresStore — asyncpg access to the central jobs/job_files/job_events tables.

The worker owns no database of its own; it reads/writes the central schema.
Conventions (from the blueprint):
- client-side uuid4() for all PKs (no pgcrypto/uuid-ossp);
- updated_at = now() on every UPDATE (no trigger);
- error messages truncated to 4000 chars;
- every output row carries ``ordinal`` (unique on (job_id, role, ordinal)) and
  the primary input's ``original_filename`` (results map back to what the
  client sent);
- per-image finalization happens LIVE via :meth:`record_image_output` so a
  redelivery skips finished ordinals and a crash never loses a produced image.
"""

from __future__ import annotations

import uuid

import asyncpg

_ERR_TRUNC = 4000


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # --- reads ----------------------------------------------------------- #
    async def get_job_by_reference(self, reference: str) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT id, reference, status, retry_count, cancel_requested, batch_id, "
            "image_count, params FROM jobs WHERE reference = $1",
            reference,
        )
        return dict(row) if row else None

    async def get_input_files(self, job_id) -> list[dict]:
        """Every non-output ``job_files`` row, ordered by image (ordinal, role).

        A multi-image job stores one row per image (same ``role``, distinct
        ``ordinal``), so the consumer groups these back into image units.
        ``original_filename`` rides along so the output row can mirror it.
        """
        rows = await self.pool.fetch(
            "SELECT role, file_path, ordinal, status, original_filename FROM job_files "
            "WHERE job_id = $1 AND role != 'output' ORDER BY ordinal, role",
            job_id,
        )
        return [dict(r) for r in rows]

    async def get_completed_ordinals(self, job_id) -> set[int]:
        """Ordinals that already have a ``completed`` output row.

        Used for idempotent redelivery: a requeue re-runs only the *remaining*
        images (persisted per-image state makes the whole-job retry safe).
        """
        rows = await self.pool.fetch(
            "SELECT DISTINCT ordinal FROM job_files "
            "WHERE job_id = $1 AND role = 'output' AND status = 'completed'",
            job_id,
        )
        return {int(r["ordinal"]) for r in rows}

    async def is_cancel_requested(self, job_id) -> bool:
        row = await self.pool.fetchrow("SELECT cancel_requested FROM jobs WHERE id = $1", job_id)
        return bool(row and row["cancel_requested"])

    # --- status transitions --------------------------------------------- #
    async def mark_running(self, job_id) -> None:
        await self.pool.execute(
            "UPDATE jobs SET status='running', started_at=COALESCE(started_at, now()), "
            "updated_at=now() WHERE id = $1",
            job_id,
        )

    async def set_stage_progress(self, job_id, stage: str, pct: int) -> None:
        await self.pool.execute(
            "UPDATE jobs SET current_stage=$2, progress=$3, updated_at=now() WHERE id = $1",
            job_id, stage, pct,
        )

    async def add_event(self, job_id, stage: str, message: str) -> None:
        await self.pool.execute(
            "INSERT INTO job_events (id, job_id, stage, message, created_at) "
            "VALUES ($1, $2, $3, $4, now())",
            str(uuid.uuid4()), job_id, stage, message,
        )

    async def mark_cancelled(self, job_id) -> None:
        await self.pool.execute(
            "UPDATE jobs SET status='cancelled', completed_at=now(), updated_at=now() WHERE id = $1",
            job_id,
        )

    async def set_failed(self, job_id, err: str) -> None:
        """Set failed WITHOUT bumping retry_count (poison -> DLQ directly)."""
        await self.pool.execute(
            "UPDATE jobs SET status='failed', error_message=$2, updated_at=now() WHERE id = $1",
            job_id, (err or "")[:_ERR_TRUNC],
        )

    async def mark_failed(self, job_id, err: str) -> int:
        """Atomic: set failed AND bump retry_count, returning the new count."""
        row = await self.pool.fetchrow(
            "UPDATE jobs SET status='failed', error_message=$2, retry_count=retry_count+1, "
            "updated_at=now() WHERE id = $1 RETURNING retry_count",
            job_id, (err or "")[:_ERR_TRUNC],
        )
        return int(row["retry_count"]) if row else 0

    # --- per-image status (multi-image jobs) ------------------------------ #
    async def set_image_status(self, job_id, ordinal: int, status: str) -> None:
        """Write one image's per-file status onto its non-output rows."""
        await self.pool.execute(
            "UPDATE job_files SET status = $3 "
            "WHERE job_id = $1 AND ordinal = $2 AND role != 'output'",
            job_id, ordinal, status,
        )

    async def mark_image_failed(self, job_id, ordinal: int) -> None:
        """Mark one image's input rows failed (no output was produced for it)."""
        await self.pool.execute(
            "UPDATE job_files SET status = 'failed' "
            "WHERE job_id = $1 AND ordinal = $2 AND role != 'output'",
            job_id, ordinal,
        )

    async def record_image_output(
        self,
        job_id,
        ordinal: int,
        rel_path: str,
        size: int,
        mime: str,
        image_count: int,
        done: int,
        original_filename: str | None = None,
    ) -> None:
        """Finalize ONE image of a multi-image job (job NOT marked completed).

        One transaction: upsert the image's output row (mirroring the primary
        input's ``original_filename``), mark its input rows completed, and
        advance the aggregate ``jobs.progress`` to ``round(done/image_count*100)``.
        The job row is flipped to ``completed`` separately by
        :meth:`mark_completed` once every image has been processed.
        """
        pct = round(done * 100 / image_count) if image_count else 100
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO job_files (id, job_id, role, ordinal, status, "
                    "file_path, file_size, mime_type, original_filename) "
                    "VALUES ($1, $2, 'output', $3, 'completed', $4, $5, $6, $7) "
                    "ON CONFLICT (job_id, role, ordinal) DO UPDATE SET "
                    "file_path=EXCLUDED.file_path, file_size=EXCLUDED.file_size, "
                    "mime_type=EXCLUDED.mime_type, status='completed', "
                    "original_filename=EXCLUDED.original_filename",
                    str(uuid.uuid4()), job_id, ordinal, rel_path, size, mime,
                    original_filename,
                )
                await conn.execute(
                    "UPDATE job_files SET status='completed' "
                    "WHERE job_id = $1 AND ordinal = $2 AND role != 'output'",
                    job_id, ordinal,
                )
                await conn.execute(
                    "UPDATE jobs SET progress = $2, updated_at = now() WHERE id = $1",
                    job_id, pct,
                )

    async def mark_completed(self, job_id, progress: int = 100) -> None:
        """One transaction: jobs -> completed + completion event.

        ``progress`` defaults to 100 for a fully-successful job; a partially
        failed job passes its real ``round(done/image_count*100)``.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE jobs SET status='completed', current_stage=NULL, "
                    "progress=$2, completed_at=now(), updated_at=now() WHERE id = $1",
                    job_id, progress,
                )
                await conn.execute(
                    "INSERT INTO job_events (id, job_id, stage, message, created_at) "
                    "VALUES ($1, $2, 'completed', 'completed', now())",
                    str(uuid.uuid4()), job_id,
                )
