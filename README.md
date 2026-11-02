# ViTOn Worker (AiVton_Service)

Headless RabbitMQ GPU worker: `ai_human` (text/reference -> image, FLUX.2-klein)
and `viton` (Qwen-Image-2.1 clothing and Garments2Look accessory try-on, native PNGs).

Set `VT_CONDA_ENV=qwen21` in the service `.env`. Both pipeline preflight checks
were verified against this existing environment on the GPU server. Production
Compose supplies the same variable from the backend `.env` (default `qwen21`),
overriding stale service settings. The `refton` Python package and model tree
names are separate from the Conda environment name.

VITON checks Qwen model shards, processor files, LoRA and both inference imports.
The production model folder must be mounted at its absolute host path.
`VT_QWEN_MODEL_DIR` and `VT_ACCESSORY_LORA_PATH` override model locations.

Preflight and inference share one command builder. Child processes inherit
`VT_CONDA_ENV`, `VT_AI_HUMAN_DIR`, `VT_REFTON_DIR` and `VT_VOSR_DIR`. The backend's
`scripts/configure_viton_runtime.py` updates the service `.env` and AI-human nested
Conda launchers (`--engine refton` additionally repairs the legacy entrypoint); it validates all sources
before writing and preserves model/package names. Use `--check` for inspection.

`GET /health` returns cached diagnostics; `GET /health/ready` returns 503 until
the database, both preflight checks and both confirmed queue subscriptions are
ready. Diagnostics include `stage_reasons`, `consumer_connected` and a bounded
startup error. A free GPU heartbeat alone does not establish execution readiness.

Startup recovery owns one database pool (maximum three connections), one GPU
reporter and one consumer. Failed preflight retries back off from 30 seconds to
five minutes; healthy operation reuses successful checks. Broker setup failures
close partial subscriptions and retry with jitter from one to 30 seconds.
Native broker reconnects restore the same subscriptions without duplicates.
Restoration stalled for 30 seconds replaces the broker session. This limit
concerns broker recovery, not how long a queued job can wait for the GPU.

Queued jobs do not expire because the GPU is busy. Claimed GPU waits renew their
leases; a GPU-wait timeout requeues without increasing the processing failure
count. Accepted-job inputs remain protected until terminal retention applies.

Run tests with `uv run pytest`. `tests/test_broker_integration.py` uses unique
temporary queues on the backend's isolated local RabbitMQ port 56739 and skips
when that test stack is absent. GPU locking/inference require Linux; lifecycle
tests can also run on Windows without exercising POSIX flock.

```bash
uv sync
uv run uvicorn viton_api.main:app --host 0.0.0.0 --port 8003   # service + consumer
uv run python -m viton_api.once ah_00001 -t ai_human           # DB-direct smoke run
```

Qwen runs explicit positional pairs, one selected pipeline per job process. Clothing
uses 1792×2400 / 40 steps / seed 42; accessories use a ≤1 MP aligned canvas /
40 steps / seed 0. No VOSR is used for Qwen jobs. Omitted tryon_mode means clothing.
The shared delivery deadline is 6900 seconds including resource waits.
