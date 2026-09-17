"""FastAPI app factory + lifespan.

Startup: ensure work_root, init DB pool (bounded retry), start GPU heartbeat
(whenever the pool exists — independent of the consumer gates), run preflight,
then start the consumer only if store + required stages + storage are all
ready. Otherwise serve /health only (graceful degradation) and run a background
self-heal task that keeps retrying the DB and starts the consumer the moment it
recovers — a transient startup DNS/network blip must never permanently disable
the consumer.

Shutdown: cancel self-heal, stop consumer, stop gpu beat, close pool.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from .config import get_settings
from .consumer import JobProcessor, RabbitConsumer
from .gpu_status import GpuStatusReporter
from .orchestrator import Orchestrator
from .preflight import Preflight
from .routers.health import router as health_router
from .store import PostgresStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("viton_api.main")

_DB_RETRY_DELAY = 1.0  # seconds
_DB_RETRY_MAX_DELAY = 30.0
_DB_STARTUP_ATTEMPTS = 6


async def _create_pool(
    settings, *, max_attempts: int | None = _DB_STARTUP_ATTEMPTS
) -> asyncpg.Pool | None:
    """Create the DB pool with exponential backoff.

    ``max_attempts`` bounds startup retries so /health can still serve degraded;
    pass ``None`` to retry forever (background self-heal). Returns ``None`` only
    when bounded and exhausted.
    """
    delay = _DB_RETRY_DELAY
    attempt = 0
    while True:
        try:
            return await asyncpg.create_pool(
                dsn=settings.database_dsn, min_size=1, max_size=3,
                # The DB is hosted Supabase behind supavisor, which silently
                # kills idle connections; a dead pooled connection costs a
                # ~600ms rebuild (TLS + SCRAM) on the next acquire and can fail
                # the query outright. TCP keepalives keep the pooled
                # connections warm, and the inactive lifetime recycles them
                # before long-idle kills land.
                server_settings={
                    "tcp_keepalives_idle": "30",
                    "tcp_keepalives_interval": "10",
                    "tcp_keepalives_count": "3",
                },
                max_inactive_connection_lifetime=300,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                logger.warning(
                    "DB pool init failed after %d attempts (degraded): %s",
                    attempt,
                    exc,
                )
                return None
            logger.warning(
                "DB pool init attempt %d failed (%s); retrying in %.0fs",
                attempt,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _DB_RETRY_MAX_DELAY)


async def _start_consumer(
    app: FastAPI, settings, store, preflight, gpu_reporter
) -> bool:
    """Build + start the consumer if every readiness gate passes; else ``False``."""
    if store is None:
        return False
    if not all(preflight.availability.values()):
        return False
    if not bool(str(settings.storage_root)):
        return False
    orchestrator = Orchestrator(settings, store, preflight, gpu_reporter)
    processor = JobProcessor(settings, store, orchestrator, gpu_reporter)
    consumer = RabbitConsumer(settings, processor, app)
    app.state.consumer = consumer
    consumer.start()
    app.state.rabbit_connected = True
    return True


async def _self_heal(app: FastAPI, settings, preflight) -> None:
    """Post-degradation recovery: retry the DB forever, then start the consumer.

    Cancelled by the lifespan's ``finally`` on shutdown.
    """
    delay = _DB_RETRY_DELAY
    while True:
        try:
            pool = await _create_pool(settings, max_attempts=None)
        except asyncio.CancelledError:
            return
        if pool is None:
            await asyncio.sleep(delay)
            delay = min(delay * 2, _DB_RETRY_MAX_DELAY)
            continue
        try:
            store = PostgresStore(pool)
            app.state.store = store
            app.state.db_connected = True
            app.state.pool = pool
            gpu_reporter = GpuStatusReporter(pool, settings, app)
            app.state.gpu_reporter = gpu_reporter
            await gpu_reporter.start()
            if await _start_consumer(app, settings, store, preflight, gpu_reporter):
                logger.info("DB recovered after startup degradation; consumer started")
                return
            logger.warning("DB recovered but readiness gates still failing; keeping pool")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("self-heal connect failed: %s", exc)
        await asyncio.sleep(delay)
        delay = min(delay * 2, _DB_RETRY_MAX_DELAY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.work_root.mkdir(parents=True, exist_ok=True)

    app.state.settings = settings
    app.state.store = None
    app.state.consumer = None
    app.state.db_connected = False
    app.state.rabbit_connected = False
    app.state.gpu_reporter = None
    app.state.gpu = None
    app.state.current_job = None
    app.state.pool = None

    preflight = Preflight(settings)
    app.state.preflight = preflight

    # 1. DB pool — bounded retry so a transient startup blip self-heals instead
    #    of permanently disabling the consumer.
    if settings.database_dsn:
        pool = await _create_pool(settings)
    else:
        pool = None
    if pool is not None:
        app.state.pool = pool
        app.state.store = PostgresStore(pool)
        app.state.db_connected = True

        # 2. GPU heartbeat — starts whenever the pool exists, independent of gates
        gpu_reporter = GpuStatusReporter(pool, settings, app)
        app.state.gpu_reporter = gpu_reporter
        await gpu_reporter.start()

    # 3. preflight
    await preflight.probe_all()

    # 4. readiness gates -> consumer
    started = await _start_consumer(
        app, settings, app.state.store, preflight, app.state.gpu_reporter
    )
    if not started:
        reasons = []
        if app.state.store is None:
            reasons.append("no-db")
        if not all(preflight.availability.values()):
            reasons.append("stages-unavailable: " + ", ".join(
                k for k, v in preflight.availability.items() if not v))
        if not bool(str(settings.storage_root)):
            reasons.append("no-storage-root")
        logger.warning(
            "refusing to start consumer (%s); serving /health only",
            "; ".join(reasons),
        )

    # 5. background self-heal — only when we degraded for lack of DB
    heal_task = None
    if app.state.store is None and settings.database_dsn:
        heal_task = asyncio.create_task(
            _self_heal(app, settings, preflight), name="db-self-heal"
        )

    try:
        yield
    finally:
        if heal_task is not None:
            heal_task.cancel()
            try:
                await heal_task
            except asyncio.CancelledError:
                pass
        consumer = getattr(app.state, "consumer", None)
        if consumer is not None:
            await consumer.stop()
        gpu_reporter = getattr(app.state, "gpu_reporter", None)
        if gpu_reporter is not None:
            await gpu_reporter.stop()
        pool = getattr(app.state, "pool", None)
        if pool is not None:
            await pool.close()
        app.state.db_connected = False


def create_app() -> FastAPI:
    app = FastAPI(title="ViTOn Worker", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    return app


app = create_app()
