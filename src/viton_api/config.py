"""Settings for the ViTOn worker (ai_human + viton pipelines).

Pattern (from the worker blueprint):
- env_prefix="VT_"; central infra also accepts the bare names (RABBITMQ_URL ...).
- All paths resolve from the service root computed at import time; the ML trees
  (ai_human/, RefTon/) live as siblings of this service inside ViTOn/.
- `conda_subprocess_env()` strips the thin venv (VIRTUAL_ENV) so that
  `conda run` resolves the *target* env's interpreter, not the service venv.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> viton_api -> src -> <service root> (AiVton_Service/)
_PKG_DIR = Path(__file__).resolve().parent
_SERVICE_ROOT = _PKG_DIR.parents[1]
_VITON_ROOT = _SERVICE_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VT_",
        env_file=str(_SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- computed paths ---
    service_root: Path = Field(default=_SERVICE_ROOT)
    work_root: Path = Field(default=_SERVICE_ROOT / "_jobs" / "work")
    runners_dir: Path = Field(default=_SERVICE_ROOT / "runners")

    # --- ML tree directories (where the model weights live; cwd for probes) ---
    ai_human_dir: Path = Field(default=_VITON_ROOT / "ai_human")
    refton_dir: Path = Field(default=_VITON_ROOT / "RefTon")
    vosr_dir: Path = Field(default=_VITON_ROOT / "tools" / "VOSR")

    # Shared ML runtime, verified for both pipelines on the GPU server.
    conda_env: str = Field(
        default="qwen21", min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"
    )

    device: str = "cuda:0"

    # --- per-pipeline base stage timeouts (seconds) ---
    timeout_ai_human: int = 900
    timeout_viton: int = 3600

    # Proportional wall-clock backstop (the hang ceiling; the stall probe is
    # the fast killer). effective = min(max(base, per_image*n + overhead), cap).
    # The cap MUST stay under RabbitMQ consumer_timeout (7200 s) so the broker
    # never force-redelivers a message this worker is still working on.
    stage_per_image_s_ai_human: float = 60.0
    stage_per_image_s_viton: float = 240.0
    stage_load_overhead_s: float = 600.0
    stage_backstop_hard_cap_s: float = 6900.0

    # Mid-stage cancellation poll (kills the subprocess tree within ~this).
    cancel_poll_interval: float = 3.0

    # --- GPU heartbeat ---
    gpu_status_interval: float = 5.0

    # --- GPU-busy gate (shared GPU): after the flock is held and a settle
    # delay (previous tenant's VRAM must hit zero), poll NVML free_mb until it
    # reaches the pipeline's threshold; on timeout give the message back (clean
    # NACK_REQUEUE, no fail/DLQ). ---
    gpu_free_mb_ai_human: int = 16000
    gpu_free_mb_viton: int = 24000
    gpu_settle_seconds: float = 30.0
    gpu_wait_interval: float = 15.0
    gpu_wait_timeout: float = 3600.0

    # --- cross-container GPU mutex (shared /data mount; same inode in both
    # workers -> flock = true mutual exclusion, auto-released on crash). ---
    gpu_lock_path: Path = Field(
        default=Path("/data/gpu-coord/gpu.lock"),
        validation_alias=AliasChoices("VT_GPU_LOCK_PATH", "GPU_LOCK_PATH"),
    )
    gpu_lock_timeout: float = 0.0  # 0 = block; in practice bounded by gpu_wait_timeout

    # --- multi-signal stall detector: stalled only when ALL of (no stdout, GPU
    # idle, process-tree CPU idle) persist for `stage_stall_seconds`, confirmed
    # across `stall_debounce_ticks` poll ticks. ---
    stage_stall_seconds: float = 300.0
    gpu_active_util_pct: int = 5
    stall_debounce_ticks: int = 3

    # --- HTTP ---
    host: str = "0.0.0.0"
    port: int = 8003
    debug: bool = False

    # --- central infra: bare name OR VT_-prefixed ---
    rabbitmq_url: str = Field(
        default="",
        validation_alias=AliasChoices("VT_RABBITMQ_URL", "RABBITMQ_URL"),
    )
    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("VT_DATABASE_URL", "DATABASE_URL"),
    )
    storage_root: Path = Field(
        default=Path(""),
        validation_alias=AliasChoices("VT_STORAGE_ROOT", "STORAGE_ROOT"),
    )
    max_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("VT_MAX_RETRIES", "MAX_RETRIES"),
    )

    # One consumer per queue; the processor lock serializes both pipelines.
    queue_names: list[str] = Field(
        default_factory=lambda: ["ai_human_queue", "viton_queue"]
    )

    # ------------------------------------------------------------------ #
    @property
    def database_dsn(self) -> str:
        """asyncpg DSN (drop the SQLAlchemy `+asyncpg` driver suffix)."""
        return self.database_url.replace("+asyncpg", "")

    def conda_command(self, *arguments: str) -> list[str]:
        """Use one environment selection for probes and actual inference."""
        return ["conda", "run", "--no-capture-output", "-n", self.conda_env, *arguments]

    def gpu_free_mb_for(self, job_type: str) -> int:
        return (
            self.gpu_free_mb_viton if job_type == "viton" else self.gpu_free_mb_ai_human
        )

    def timeout_base_for(self, job_type: str) -> int:
        return self.timeout_viton if job_type == "viton" else self.timeout_ai_human

    def per_image_s_for(self, job_type: str) -> float:
        return (
            self.stage_per_image_s_viton
            if job_type == "viton"
            else self.stage_per_image_s_ai_human
        )

    def conda_subprocess_env(
        self, extra: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Environment for `conda run` subprocesses.

        When launched via `uv run`, VIRTUAL_ENV gives the thin venv precedence
        over the target conda env. Strip it (and scrub its bin/ off PATH) so the
        stage runs under the right interpreter.
        """
        env = dict(os.environ)
        virtual_env = env.pop("VIRTUAL_ENV", None)
        env.pop("VIRTUAL_ENV_PROMPT", None)
        env.pop("UV_RUN_RECURSION_DEPTH", None)
        if virtual_env:
            virtual_bin = os.path.normpath(str(Path(virtual_env) / "bin"))
            env["PATH"] = os.pathsep.join(
                p
                for p in env.get("PATH", "").split(os.pathsep)
                if os.path.normpath(p) != virtual_bin
            )
        # The runner scripts locate the ML trees through these (single source:
        # the .env / compose already carry the VT_ values).
        env["VT_CONDA_ENV"] = self.conda_env
        env["VT_AI_HUMAN_DIR"] = str(self.ai_human_dir)
        env["VT_REFTON_DIR"] = str(self.refton_dir)
        env["VT_VOSR_DIR"] = str(self.vosr_dir)
        if extra:
            env.update(extra)
        # expandable allocator reduces fragmentation/OOM on the shared GPU; an
        # operator-set value always wins.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — settings are read once on first access."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def dlq_name_for(queue: str) -> str:
    """`ai_human_queue` -> `ai_human_dlq`."""
    base = queue[: -len("_queue")] if queue.endswith("_queue") else queue
    return f"{base}_dlq"
