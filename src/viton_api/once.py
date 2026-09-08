"""Isolated runner: `python -m viton_api.once <ref> -t <job_type>` (no RabbitMQ).

Runs ONE job by reference against a real DB + storage — the whole JobProcessor
contract (input staging, GPU gate, subprocess, output write-back, retry
accounting) without the broker. Prints the resulting AckDecision; exits
non-zero on anything but ACK.

Examples:
    python -m viton_api.once ah_00001 -t ai_human
    python -m viton_api.once vt_00001 -t viton
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import asyncpg

from .config import get_settings
from .consumer import JobProcessor
from .models import SUPPORTED_JOB_TYPES
from .orchestrator import Orchestrator
from .preflight import Preflight
from .store import PostgresStore

logger = logging.getLogger("viton_api.once")


async def run_once(reference: str, job_type: str) -> str:
    settings = get_settings()
    pool = await asyncpg.create_pool(dsn=settings.database_dsn, min_size=1, max_size=3)
    try:
        store = PostgresStore(pool)
        preflight = Preflight(settings)
        await preflight.probe_all()
        orchestrator = Orchestrator(settings, store, preflight)
        processor = JobProcessor(settings, store, orchestrator)
        decision = await processor.process(reference, job_type)
        print(f"AckDecision = {decision.value}")
        return decision.value
    finally:
        await pool.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Run one ViTOn job by reference (no RabbitMQ).",
    )
    parser.add_argument("reference", help="jobs.reference (the human-readable job id)")
    parser.add_argument(
        "--job-type", "-t", default="ai_human",
        choices=sorted(SUPPORTED_JOB_TYPES),
    )
    args = parser.parse_args()
    decision = asyncio.run(run_once(args.reference, args.job_type))
    # non-zero exit on DLQ/requeue so it's visible in scripts
    sys.exit(0 if decision == "ack" else 1)


if __name__ == "__main__":
    main()
