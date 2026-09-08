"""GET /health — O(1), reads cached app.state. Optional ?refresh=1 re-runs preflight."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..models import GpuHealth, Health

router = APIRouter()


@router.get("/health", response_model=Health)
async def health(request: Request, refresh: bool = False) -> Health:
    state = request.app.state
    settings = state.settings

    if refresh:
        await state.preflight.probe_all()

    availability = dict(getattr(state.preflight, "availability", {}))
    required_ready = all(availability.values()) if availability else False
    rabbit_connected = getattr(state, "rabbit_connected", False)
    db_connected = getattr(state, "db_connected", False)

    status = "ok" if (required_ready and rabbit_connected and db_connected) else "degraded"

    return Health(
        status=status,
        conda_env=settings.conda_env,
        queues=list(settings.queue_names),
        stage_availability=availability,
        rabbit_connected=rabbit_connected,
        db_connected=db_connected,
        storage_root=str(settings.storage_root),
        gpu=GpuHealth(**state.gpu) if getattr(state, "gpu", None) else None,
    )
