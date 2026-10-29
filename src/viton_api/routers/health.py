"""Cached process diagnostics and execution readiness; no I/O per health poll."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..diagnostics import safe_reason
from ..models import GpuHealth, Health, STAGE_KEYS

router = APIRouter()


def snapshot(state) -> Health:
    settings = state.settings
    preflight = state.preflight
    availability = dict(preflight.availability)
    consumer = getattr(state, "consumer", None)
    broker_connected = bool(consumer and consumer.broker_connected)
    consumer_connected = bool(consumer and consumer.connected)
    db_connected = bool(getattr(state, "db_connected", False))
    required_ready = all(availability.get(key, False) for key in STAGE_KEYS)
    ready = (
        required_ready
        and db_connected
        and consumer_connected
        and not getattr(state, "stopping", False)
    )
    return Health(
        status="ok" if ready else "degraded",
        conda_env=settings.conda_env,
        queues=list(settings.queue_names),
        stage_availability=availability,
        stage_reasons={
            key: safe_reason(value, settings)
            for key, value in preflight.reasons.items()
        },
        rabbit_connected=broker_connected,
        consumer_connected=consumer_connected,
        db_connected=db_connected,
        startup_error=getattr(state, "startup_error", None)
        or getattr(consumer, "last_error", None),
        storage_root=str(settings.storage_root),
        gpu=GpuHealth(**state.gpu) if getattr(state, "gpu", None) else None,
    )


@router.get("/health", response_model=Health)
async def health(request: Request, refresh: bool = False) -> Health:
    if refresh:
        await request.app.state.preflight.probe_all()
    return snapshot(request.app.state)


@router.get("/health/ready", response_model=Health)
async def ready(request: Request):
    result = snapshot(request.app.state)
    return JSONResponse(
        result.model_dump(), status_code=200 if result.status == "ok" else 503
    )
