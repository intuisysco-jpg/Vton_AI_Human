"""Attempt-token fencing (shared fencing.py): duplicate or late deliveries never
rerun the GPU, and a transient DB error does not stop lease renewal."""

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
from types import SimpleNamespace
import uuid

import pytest

from viton_api import fencing
from viton_api.fencing import FencedPool, fenced


class AckDecision(str, Enum):  # fenced() resolves AckDecision from the caller's globals
    ACK = "ack"
    NACK_REQUEUE = "nack_requeue"


class FakeDb:
    """Raw asyncpg-pool stand-in holding one jobs row."""

    def __init__(self, job, lease_live=False, renew_results=()):
        self.job = job
        self.lease_live = lease_live
        self.renew_results = list(renew_results)
        self.renewals = 0
        self.writes = []

    @asynccontextmanager
    async def acquire(self):
        yield FakeConn(self)

    async def execute(self, sql, *args):  # lease renewal goes through the raw pool
        self.renewals += 1
        result = self.renew_results.pop(0) if self.renew_results else "UPDATE 1"
        if isinstance(result, Exception):
            raise result
        return result


class FakeConn:
    def __init__(self, db):
        self.db = db

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetchrow(self, sql, *args):
        return self.db.job

    async def fetchval(self, sql, *args):
        if "timestamptz>now()" in sql:
            return self.db.lease_live
        return self.db.job["status"]

    async def execute(self, sql, *args):
        self.db.writes.append(sql)
        return "UPDATE 1"


def job_row(status="queued", token=None, completed_at=None):
    return {
        "id": uuid.uuid4(),
        "status": status,
        "attempt_token": token,
        "lease_until": None,
        "completed_at": completed_at,
    }


class Processor:
    def __init__(self, db, work=None):
        self.store = SimpleNamespace(pool=FencedPool(db))
        self.calls = 0
        self.work = work

    @fenced
    async def process(self, reference):
        self.calls += 1
        if self.work:
            await self.work()
        return AckDecision.ACK


def test_duplicate_delivery_while_leased_is_dropped():
    db = FakeDb(job_row("running", token=uuid.uuid4()), lease_live=True)
    proc = Processor(db)
    assert asyncio.run(proc.process("ref")) is AckDecision.ACK
    assert proc.calls == 0
    assert db.writes == []


# "failed" without completed_at is what store.set_failed (poison -> DLQ) leaves.
@pytest.mark.parametrize("status", ["completed", "cancelled", "failed"])
def test_terminal_job_is_not_rerun(status):
    db = FakeDb(job_row(status, completed_at=None))
    proc = Processor(db)
    assert asyncio.run(proc.process("ref")) is AckDecision.ACK
    assert proc.calls == 0
    assert db.writes == []


def test_fresh_job_is_claimed_and_run():
    db = FakeDb(job_row("queued"))
    proc = Processor(db)
    assert asyncio.run(proc.process("ref")) is AckDecision.ACK
    assert proc.calls == 1
    assert any("SET attempt_token=$2,lease_until" in sql for sql in db.writes)


def test_lease_renews_well_before_it_expires():
    assert fencing.RENEW_SECONDS * 3 < fencing.LEASE_SECONDS


def test_transient_renewal_error_keeps_renewing(monkeypatch):
    monkeypatch.setattr(fencing, "RENEW_SECONDS", 0.01)
    db = FakeDb(job_row("queued"), renew_results=[ConnectionError("blip")])

    async def work():
        for _ in range(500):
            if db.renewals >= 3:
                return
            await asyncio.sleep(0.01)

    proc = Processor(db, work)
    asyncio.run(proc.process("ref"))
    assert db.renewals >= 3
