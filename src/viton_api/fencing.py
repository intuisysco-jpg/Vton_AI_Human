"""Portable Python 3.12+ worker protocol, copied identically into all workers.

Only attempt-owned writes can register outputs. GPU locking remains in each worker.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
import functools
import json
import uuid

attempts = ContextVar("worker_attempts", default={})
references = ContextVar("worker_references", default={})


def attempt_path(reference):
    return references.get().get(reference, "legacy")


class FencedPool:
    def __init__(self, pool):
        self.raw = pool

    @asynccontextmanager
    async def acquire(self):
        async with self.raw.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.attempts',$1,true)",
                    json.dumps(attempts.get()),
                )
                yield conn

    def __getattr__(self, name):
        if name in {"execute", "executemany", "fetch", "fetchrow", "fetchval"}:

            async def invoke(*args, **kwargs):
                async with self.acquire() as conn:
                    return await getattr(conn, name)(*args, **kwargs)

            return invoke
        return getattr(self.raw, name)


def fenced(function):
    @functools.wraps(function)
    async def run(self, ref, *args, **kwargs):
        # In-memory stores used by pipeline-only tests have no database lease layer.
        if not isinstance(getattr(self.store, "pool", None), FencedPool):
            return await function(self, ref, *args, **kwargs)
        raw = self.store.pool.raw
        wanted = ref if isinstance(ref, list) else [ref]
        owned, by_ref = {}, {}
        for reference in wanted:
            async with raw.acquire() as conn:
                async with conn.transaction():
                    job = await conn.fetchrow(
                        "SELECT * FROM jobs WHERE reference=$1 FOR UPDATE", reference
                    )
                    if (
                        not job
                        or job["status"] in {"completed", "cancelled"}
                        or (job["status"] == "failed" and job["completed_at"])
                    ):
                        continue
                    if job["attempt_token"] and await conn.fetchval(
                        "SELECT $1::timestamptz>now()", job["lease_until"]
                    ):
                        continue
                    token = str(uuid.uuid4())
                    await conn.execute(
                        "SELECT set_config('app.coordinator','yes',true)"
                    )
                    await conn.execute(
                        "UPDATE jobs SET attempt_token=$2,lease_until=now()+interval '90 seconds' WHERE id=$1",
                        job["id"],
                        uuid.UUID(token),
                    )
                    owned[str(job["id"])] = token
                    by_ref[reference] = token
        if not owned:
            enum = function.__globals__["AckDecision"]
            return enum.ACK
        at = attempts.set(owned)
        rt = references.set(by_ref)
        lost = asyncio.Event()

        async def renew():
            while True:
                await asyncio.sleep(20)
                try:
                    for job_id, token in owned.items():
                        result = await raw.execute(
                            "UPDATE jobs SET lease_until=now()+interval '90 seconds' WHERE id=$1 AND attempt_token=$2 AND lease_until>now()",
                            uuid.UUID(job_id),
                            uuid.UUID(token),
                        )
                        if result != "UPDATE 1":
                            lost.set()
                            return
                except Exception:
                    lost.set()
                    return

        heartbeat = asyncio.create_task(renew())
        try:
            result = await function(
                self, list(by_ref) if isinstance(ref, list) else ref, *args, **kwargs
            )
            decision = result.value
            async with self.store.pool.acquire() as conn:
                for job_id, token in owned.items():
                    state = await conn.fetchval(
                        "SELECT status FROM jobs WHERE id=$1", uuid.UUID(job_id)
                    )
                    if state not in {"completed", "cancelled", "failed"}:
                        terminal = decision != "nack_requeue"
                        await conn.execute(
                            "UPDATE jobs SET status=$2,completed_at=CASE WHEN $3 THEN now() ELSE NULL END,updated_at=now() WHERE id=$1 AND attempt_token=$4",
                            uuid.UUID(job_id),
                            "failed" if terminal else "retrying",
                            terminal,
                            uuid.UUID(token),
                        )
                    await conn.execute(
                        "UPDATE jobs SET attempt_token=NULL,lease_until=NULL WHERE id=$1 AND attempt_token=$2",
                        uuid.UUID(job_id),
                        uuid.UUID(token),
                    )
            if decision == "nack_requeue":
                await asyncio.sleep(5)
            return result
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            attempts.reset(at)
            references.reset(rt)

    return run
