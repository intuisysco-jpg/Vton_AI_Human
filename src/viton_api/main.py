"""One supervised startup/recovery loop; healthy operation reuses cached checks."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import time

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
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("viton_api.main")
_DB_RETRY_MAX_DELAY = 30.0
_DB_HEALTH_INTERVAL = 15.0
_PREFLIGHT_RETRY_DELAY = 30.0
_PREFLIGHT_RETRY_MAX_DELAY = 300.0


async def _create_pool(settings):
    return await asyncpg.create_pool(
        dsn=settings.database_dsn,
        min_size=1,
        max_size=3,
        timeout=10,
        command_timeout=15,
        max_inactive_connection_lifetime=300,
        server_settings={
            "application_name": "shotwork-viton-worker",
            "tcp_keepalives_idle": "30",
            "tcp_keepalives_interval": "10",
            "tcp_keepalives_count": "3",
        },
    )


async def _start_consumer(app, settings, store, preflight, gpu_reporter) -> bool:
    if store is None or gpu_reporter is None or not preflight.ready:
        return False
    if app.state.consumer is None:
        orchestrator = Orchestrator(settings, store, preflight, gpu_reporter)
        processor = JobProcessor(settings, store, orchestrator, gpu_reporter)
        app.state.consumer = RabbitConsumer(settings, processor, app)
    app.state.consumer.start()  # Idempotent; readiness comes from subscriptions.
    return True


async def _supervise(app, settings) -> None:
    state = app.state
    db_delay = 1.0
    probe_delay = _PREFLIGHT_RETRY_DELAY
    next_probe = next_db_check = 0.0
    while True:
        if state.pool is None:
            try:
                if not settings.database_dsn:
                    raise ValueError("Database is not configured")
                state.pool = await _create_pool(settings)
                state.store = PostgresStore(state.pool)
                state.db_connected = True
                db_delay = 1.0
            except Exception as exc:
                state.db_connected = False
                state.startup_error = f"Database startup failed: {type(exc).__name__}"
                logger.warning("%s; retry in %.0fs", state.startup_error, db_delay)
                await asyncio.sleep(db_delay)
                db_delay = min(_DB_RETRY_MAX_DELAY, db_delay * 2)
                continue
        if time.monotonic() >= next_db_check:
            try:
                await asyncio.wait_for(state.pool.fetchval("SELECT 1"), timeout=5)
                state.db_connected = True
            except Exception as exc:
                state.db_connected = False
                state.startup_error = f"Database check failed: {type(exc).__name__}"
            next_db_check = time.monotonic() + _DB_HEALTH_INTERVAL
        if state.db_connected:
            if state.gpu_reporter is None:
                reporter = GpuStatusReporter(state.pool, settings, app)
                try:
                    await reporter.start()
                    state.gpu_reporter = reporter
                except BaseException:
                    await reporter.stop()
                    raise
            if not state.preflight.ready and time.monotonic() >= next_probe:
                try:
                    await state.preflight.probe_all()
                except Exception as exc:
                    logger.warning("Preflight failed: %s", type(exc).__name__)
                    state.startup_error = f"Preflight failed: {type(exc).__name__}"
                next_probe = time.monotonic() + probe_delay
                probe_delay = min(_PREFLIGHT_RETRY_MAX_DELAY, probe_delay * 2)
            if state.preflight.ready:
                state.startup_error = None
                probe_delay = _PREFLIGHT_RETRY_DELAY
                await _start_consumer(
                    app, settings, state.store, state.preflight, state.gpu_reporter
                )
            elif state.preflight.reasons:
                state.startup_error = (
                    "Pipeline preflight checks failed; see stage_reasons"
                )
        await asyncio.sleep(1)


async def _supervisor_loop(app, settings):
    while True:
        try:
            await _supervise(app, settings)
        except Exception as exc:
            app.state.startup_error = f"Worker startup failed: {type(exc).__name__}"
            logger.warning("%s; retry in 30s", app.state.startup_error)
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.work_root.mkdir(parents=True, exist_ok=True)
    state = app.state
    state.settings = settings
    state.store = state.consumer = state.pool = state.gpu_reporter = None
    state.db_connected = state.rabbit_connected = state.consumer_connected = False
    state.gpu = state.current_job = state.startup_error = None
    state.stopping = False
    state.preflight = Preflight(settings)
    supervisor = asyncio.create_task(
        _supervisor_loop(app, settings), name="worker-supervisor"
    )
    try:
        yield
    finally:
        state.stopping = True
        supervisor.cancel()
        with suppress(asyncio.CancelledError):
            await supervisor
        if state.consumer is not None:
            await state.consumer.stop()
        if state.gpu_reporter is not None:
            await state.gpu_reporter.stop()
        if state.pool is not None:
            try:
                await asyncio.wait_for(state.pool.close(), timeout=5)
            except asyncio.TimeoutError:
                state.pool.terminate()
        state.db_connected = state.rabbit_connected = state.consumer_connected = False


def create_app() -> FastAPI:
    app = FastAPI(title="ViTOn Worker", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    return app


app = create_app()
