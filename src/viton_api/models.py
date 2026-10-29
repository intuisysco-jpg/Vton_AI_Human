"""Status constants, stage definitions, param parsing and the /health models."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

logger = logging.getLogger("viton_api.models")


# --------------------------------------------------------------------------- #
# Job status (validated in app code; the DB column is plain TEXT, no enum).
# --------------------------------------------------------------------------- #
class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# `failed` is NOT terminal — it can be retried. Only completed/cancelled stop
# further processing (idempotent redelivery -> ACK without re-run).
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


# --------------------------------------------------------------------------- #
# Stages. Each job type is a single runner invocation; one stage runs per job
# (dispatched by job_type, not a sequence).
# --------------------------------------------------------------------------- #
STAGE_KEYS = ("ai_human", "viton")

# Progress reported while the single stage runs; 100 is set only on finalize.
STAGE_PROGRESS = {
    "ai_human": 50,
    "viton": 50,
}


def image_pct(done: int, n: int) -> int:
    """Map ``done`` finalized images of ``n`` into the stage window (50 -> 100).

    The ONE ladder shared by the DB ``jobs.progress`` writes and the SSE
    frames, so polls and streams always agree and never move backward (the
    stage start writes 50; the first image must land >= 50, not 1/N*100).
    """
    if not n:
        return 100
    base = STAGE_PROGRESS["viton"]
    return min(100, base + round((100 - base) * done / n))

# job_type -> (stage_key, Settings attr for the conda env, Settings attr for the probe cwd tree)
JOB_TYPE_TO_STAGE = {
    "ai_human": ("ai_human", "conda_env", "ai_human_dir"),
    "viton": ("viton", "conda_env", "refton_dir"),
}

SUPPORTED_JOB_TYPES = frozenset(JOB_TYPE_TO_STAGE.keys())


# Runner exit codes (both runner scripts follow this contract).
EXIT_OK = 0            # >= 1 image produced (per-image failures already reported)
EXIT_RETRY = 1         # unexpected failure / nothing produced -> retry budget
EXIT_POISON = 2        # bad params/inputs/path-guard -> failed, no retry, ACK
EXIT_OOM = 3           # CUDA OOM / GPU precondition -> clean requeue, no DB write


# Enums mirrored from app/schemas/ai_features.py (validated there at upload;
# re-checked here so a hand-inserted bad row fails as POISON before GPU work).
PERSON_TYPES = {"man", "woman", "child", "auto"}
POSES = {"straight", "a_pose", "t_pose", "walk", "one_hand_waist",
         "cross_legs", "side_relaxed", "hair_fix_strut"}
ANGLES = {"front", "three_quarter_left", "three_quarter_right",
          "left_side", "right_side", "back"}
OUTFIT_TYPES = {"tshirt", "shirt", "formal_pant", "jeans", "frock", "long_gown",
                "short_skirt_lower", "short_skirt_upper", "long_skirt_lower",
                "long_skirt_upper", "jacket", "short_jacket", "long_jacket", "coat"}
GARMENT_TYPES = {"full_sleeves_shirt", "half_sleeves_shirt", "full_pant",
                 "half_pant", "long_gown", "half_gown", "jacket", "sweater",
                 "short_skirt", "long_skirt"}


def parse_params(raw) -> dict:
    """Normalize the jobs.params column (JSONB) to a dict.

    asyncpg hands JSONB back as a str (no codec registered); be tolerant of an
    already-decoded dict and of NULL/blank (empty params are legal for viton).
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("params column is not valid JSON: %r -> {}", raw[:200])
            return {}
        return value if isinstance(value, dict) else {}
    logger.warning("params column has unexpected type %s -> {}", type(raw).__name__)
    return {}


def validate_ai_human_params(params: dict) -> str | None:
    """Return an error message for a bad ai_human params dict, else None."""
    if params.get("person_type", "man") not in PERSON_TYPES:
        return f"invalid person_type {params.get('person_type')!r}"
    if params.get("pose", "straight") not in POSES:
        return f"invalid pose {params.get('pose')!r}"
    if params.get("angle", "front") not in ANGLES:
        return f"invalid angle {params.get('angle')!r}"
    if params.get("outfit_type") is not None and params["outfit_type"] not in OUTFIT_TYPES:
        return f"invalid outfit_type {params.get('outfit_type')!r}"
    if params.get("outfit") and params.get("outfit_type"):
        return "provide either 'outfit' or 'outfit_type', not both"
    seed = params.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed <= 2_147_483_647):
        return f"invalid seed {seed!r}"
    return None


def validate_viton_params(params: dict) -> str | None:
    """Return an error message for a bad viton params dict, else None."""
    if params.get("garment_type") is not None and params["garment_type"] not in GARMENT_TYPES:
        return f"invalid garment_type {params.get('garment_type')!r}"
    if params.get("garment_type") and params.get("garment_prompt"):
        return "provide either 'garment_type' or 'garment_prompt', not both"
    return None


# --------------------------------------------------------------------------- #
# /health response models.
# --------------------------------------------------------------------------- #
class GpuHealth(BaseModel):
    gpu_id: str | None = None
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    utilization: int | None = None
    current_job: str | None = None
    reported_at: str | None = None


class Health(BaseModel):
    status: str = "ok"
    conda_env: str = ""
    queues: list[str] = []
    stage_availability: dict[str, bool] = {}
    stage_reasons: dict[str, str] = {}
    rabbit_connected: bool = False
    consumer_connected: bool = False
    db_connected: bool = False
    startup_error: str | None = None
    storage_root: str = ""
    gpu: GpuHealth | None = None
