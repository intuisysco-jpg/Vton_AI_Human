#!/usr/bin/env python
"""ai_model runner — person photo -> identity-replaced photo via ai_model/pipeline.py.

Invoked by the orchestrator as:
    conda run --no-capture-output -n aimodel python <runners>/ai_model_runner.py \
        --job-json <workdir>/job.json

Contract (exit codes the consumer routes on):
    0  >= 1 image produced (per-image failures already reported)
    1  unexpected failure / nothing produced        -> retry budget
    2  poison (bad params/inputs/tree)              -> failed, no retry
    3  CUDA OOM                                     -> clean requeue

Progress protocol: JSONL events on stdout (the orchestrator dispatches them):
    {"event": "image_done", "ordinal": N, "path": "..."}
    {"event": "image_failed", "ordinal": N, "error": "..."}
    {"event": "fatal_poison", "error": "..."}
Everything else printed is transcript (goes to job.log).

Params (central /jobs/ai-model schema, mirrored in viton_api.models):
    seed    base seed; ordinal i uses seed+i (default 42)

pipeline.process() status mapping:
    accepted                          -> image_done (final_image at the input
                                         photo's original resolution)
    manual_review                     -> image_failed (QC reasons included)
    skip_no_person / skip_no_head     -> image_failed (deterministic per input;
                                         a retry of the same photo cannot help)
    failed_* / qc_unavailable         -> image_failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import NoReturn

# located at <service>/runners/ai_model_runner.py -> parents[2] = ViTOn/
_DEFAULT_AI_MODEL_DIR = Path(__file__).resolve().parents[2] / "new" / "ai_model"

_DEFAULT_SEED = 42
_DEFAULT_MAX_RETRIES = 2


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def die_poison(message: str) -> NoReturn:
    emit({"event": "fatal_poison", "error": message})
    sys.exit(2)


def validate_params(params: dict) -> str | None:
    seed = params.get("seed", _DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed <= 2_147_483_647):
        return f"invalid seed {seed!r}"
    return None


def _describe(result: dict) -> str:
    """Human-readable one-liner for a non-accepted pipeline result."""
    status = result.get("status", "unknown")
    reasons = result.get("qc_reasons") or result.get("reasons") or []
    if isinstance(reasons, (list, tuple)):
        reasons = "; ".join(str(r) for r in reasons)
    return f"{status}: {reasons}" if reasons else str(status)


def _save_final(final_image: str, dest: Path) -> None:
    """Copy the pipeline composite to the contract path atomically."""
    from PIL import Image

    temporary = dest.with_suffix(".partial.png")
    with Image.open(final_image) as im:
        im.load()
        im.save(temporary, format="PNG")
    with Image.open(temporary) as saved:
        saved.verify()
    temporary.replace(dest)


def run(job: dict, process_fn=None) -> int:
    """Execute the job; returns the runner exit code (see module docstring).

    ``process_fn`` overrides ``pipeline.process`` (tests inject a fake here).
    """
    params = job.get("params") or {}
    units = job.get("units") or []
    err = validate_params(params)
    if err:
        die_poison(err)
    if not units:
        die_poison("no image units in job.json")

    ai_model_dir = Path(os.environ.get("VT_AI_MODEL_DIR") or _DEFAULT_AI_MODEL_DIR)
    if not (ai_model_dir / "pipeline.py").exists():
        die_poison(f"ai_model tree not found at {ai_model_dir}")

    for unit in units:
        person = unit.get("person")
        if not person or not Path(person).exists():
            # _stage_units already guarantees the files; missing here is a
            # staging bug, not a per-photo quality problem -> poison.
            die_poison(f"person input missing for ordinal {unit.get('ordinal')}")

    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # import AFTER validation (no GPU work for a poison job)
    if process_fn is None:
        sys.path.insert(0, str(ai_model_dir))
        try:
            import pipeline
        except Exception as e:  # noqa: BLE001
            emit({"event": "fatal_poison", "error": f"import pipeline failed: {e}"})
            return 2
        process_fn = pipeline.process

    max_retries = int(os.environ.get("VT_AI_MODEL_MAX_RETRIES", _DEFAULT_MAX_RETRIES))
    base_seed = int(params.get("seed", _DEFAULT_SEED))

    produced = 0
    oom = False
    for unit in units:
        ordinal = int(unit["ordinal"])
        dest = out_dir / f"{ordinal:04d}.png"
        try:
            result = process_fn(
                input_path=str(unit["person"]),
                seed=base_seed + ordinal,
                max_retries=max_retries,
            )
        except Exception as e:  # noqa: BLE001 -- translate ML failures into runner exit codes
            traceback.print_exc()
            if type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower():
                oom = True
                emit({"event": "image_failed", "ordinal": ordinal,
                      "error": f"CUDA OOM: {e}"})
                break  # VRAM is gone; further images would fail too
            emit({"event": "image_failed", "ordinal": ordinal, "error": str(e)})
            continue

        if not isinstance(result, dict):
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": f"pipeline returned {type(result).__name__}, expected dict"})
            continue
        status = str(result.get("status", ""))
        print(f"[ai_model] ordinal {ordinal}: status={status} "
              f"seconds={result.get('total_seconds', '?')}", flush=True)
        if status != "accepted":
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": _describe(result)})
            continue
        final_image = result.get("final_image")
        if not final_image or not Path(final_image).exists():
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": f"accepted but final_image missing: {final_image!r}"})
            continue
        try:
            _save_final(str(final_image), dest)
        except Exception as e:  # noqa: BLE001
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": f"could not persist final image: {e}"})
            continue
        produced += 1
        emit({"event": "image_done", "ordinal": ordinal, "path": str(dest)})

    if oom:
        return 3
    if produced == 0:
        print("no images produced", file=sys.stderr, flush=True)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()
    try:
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        die_poison(f"unreadable job.json: {e}")
    sys.exit(run(job))


if __name__ == "__main__":
    main()
