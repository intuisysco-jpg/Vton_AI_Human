#!/usr/bin/env python
"""ai_human runner — text/reference -> image via ai_human/generate_human.py.

Invoked by the orchestrator as:
    conda run --no-capture-output -n refton python <runners>/ai_human_runner.py \
        

Contract (exit codes the consumer routes on):
    0  >= 1 image produced (per-image failures already reported)
    1  unexpected failure / nothing produced        -> retry budget
    2  poison (bad params/inputs)                    -> failed, no retry
    3  CUDA OOM                                      -> clean requeue

Progress protocol: JSONL events on stdout (the orchestrator dispatches them):
    {"event": "image_done", "ordinal": N, "path": "..."}
    {"event": "image_failed", "ordinal": N, "error": "..."}
    {"event": "fatal_poison", "error": "..."}
Everything else printed is transcript (goes to job.log).

Params (central /jobs/ai-human schema, mirrored in viton_api.models):
    person_type man|woman|child|auto   (reference jobs: ALWAYS ignored — gender
                                       derives from the reference image;
                                       text jobs: auto + no reference -> man)
    pose, angle                        (pipeline enums)
    outfit XOR outfit_type             (free text or preset)
    extra_prompt                       free text
    seed                               base seed; ordinal i uses seed+i
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NoReturn

# located at <service>/runners/ai_human_runner.py -> parents[2] = ViTOn/
_DEFAULT_AI_HUMAN_DIR = Path(__file__).resolve().parents[2] / "ai_human"

VALID_PERSON_TYPES = {"man", "woman", "child", "auto"}
VALID_POSES = {"straight", "a_pose", "t_pose", "walk", "one_hand_waist",
               "cross_legs", "side_relaxed", "hair_fix_strut"}
VALID_ANGLES = {"front", "three_quarter_left", "three_quarter_right",
                "left_side", "right_side", "back"}


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def die_poison(message: str) -> NoReturn:
    emit({"event": "fatal_poison", "error": message})
    sys.exit(2)


def validate_params(params: dict) -> str | None:
    person_type = params.get("person_type", "auto")
    if person_type not in VALID_PERSON_TYPES:
        return f"invalid person_type {person_type!r}"
    if params.get("pose", "straight") not in VALID_POSES:
        return f"invalid pose {params.get('pose')!r}"
    if params.get("angle", "front") not in VALID_ANGLES:
        return f"invalid angle {params.get('angle')!r}"
    if params.get("outfit") and params.get("outfit_type"):
        return "provide either 'outfit' or 'outfit_type', not both"
    seed = params.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        return f"invalid seed {seed!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()

    job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    params = job.get("params") or {}
    units = job.get("units") or []
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    err = validate_params(params)
    if err:
        die_poison(err)
    if not units:
        die_poison("no image units in job.json")

    ai_human_dir = Path(os.environ.get("VT_AI_HUMAN_DIR") or _DEFAULT_AI_HUMAN_DIR)
    if not (ai_human_dir / "generate_human.py").exists():
        die_poison(f"ai_human tree not found at {ai_human_dir}")

    # import AFTER validation (no GPU work for a poison job)
    sys.path.insert(0, str(ai_human_dir))
    try:
        import torch
        import generate_human as gh
    except Exception as e:  # noqa: BLE001
        emit({"event": "fatal_poison", "error": f"import generate_human failed: {e}"})
        sys.exit(2)

    base_seed = int(params.get("seed", 0))
    # default "auto": with a reference the gender/age is derived from it;
    # without one, the auto -> man fallback below keeps the old text-mode default
    person_type_param = params.get("person_type", "auto")
    pose = params.get("pose", "straight")
    angle = params.get("angle", "front")
    outfit = params.get("outfit", "") or ""
    outfit_type = params.get("outfit_type", "") or ""
    extra_prompt = params.get("extra_prompt", "") or ""
    steps = int(params.get("steps", 8))  # <8 invites extra-limb artifacts on raised-arm poses

    produced = 0
    oom = False
    try:
        gh.load_model()
    except Exception as e:  # noqa: BLE001
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            print(f"CUDA OOM while loading model: {e}", file=sys.stderr, flush=True)
            sys.exit(3)
        raise

    for unit in units:
        ordinal = int(unit["ordinal"])
        ref = unit.get("reference")
        if ref and not Path(ref).exists():
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": f"reference file missing: {ref}"})
            continue
        # image-to-image: gender/age ALWAYS derives from the reference image —
        # person_type is deliberately ignored here (clients were hardcoding
        # "man" for reference jobs, forcing male output).
        # text mode + auto has no reference to derive from -> man
        if ref:
            person_type = "auto"
        else:
            person_type = "man" if person_type_param == "auto" else person_type_param

        out_path = out_dir / f"{ordinal:04d}.png"
        try:
            gh.generate_image(
                person_type,
                seed=base_seed + ordinal,
                output_path=str(out_path),
                pose=pose,
                angle=angle,
                extra_prompt=extra_prompt,
                outfit=outfit,
                outfit_type=outfit_type,
                steps=steps,
                reference_image=str(ref) if ref else None,
                upscale="none",
            )
        except Exception as e:  # noqa: BLE001
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                oom = True
                emit({"event": "image_failed", "ordinal": ordinal,
                      "error": f"CUDA OOM: {e}"})
                break  # VRAM is gone; further images would fail too
            emit({"event": "image_failed", "ordinal": ordinal, "error": str(e)})
            continue
        if out_path.exists():
            produced += 1
            emit({"event": "image_done", "ordinal": ordinal, "path": str(out_path)})
        else:
            emit({"event": "image_failed", "ordinal": ordinal,
                  "error": "generate_image returned without writing the output file"})

    if oom:
        sys.exit(3)
    if produced == 0:
        print("no images produced", file=sys.stderr, flush=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
