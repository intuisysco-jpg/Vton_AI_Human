# ViTOn Worker (AiVton_Service)

Headless RabbitMQ GPU worker: `ai_human` (text/reference -> image, FLUX.2-klein)
and `viton` (virtual try-on, RefTon/FLUX.1-Kontext + VOSR2 4x upscale).

```bash
uv sync
uv run uvicorn viton_api.main:app --host 0.0.0.0 --port 8003   # service + consumer
uv run python -m viton_api.once ah_00001 -t ai_human           # DB-direct smoke run
```
