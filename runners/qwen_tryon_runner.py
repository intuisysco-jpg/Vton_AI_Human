#!/usr/bin/env python
"""Production adapter of qwen21_tryon_test.py and product_tryon.py/run_g2l.py.

Keeps the verified model settings, but consumes explicit ordinal pairs rather
than folder Cartesian products. One selected pipeline is loaded per job.

Clothing keeps the diffusers Qwen 2.1 pipeline in-process. Accessory jobs
shell out per unit to the audited crop-context implementation
(qwen21_tryon_test/try_on_gar/run_accessory_tryon.py: MediaPipe landmark
placement + Qwen 2.1 crop generation, LoRA off) and translate its artifacts
and exit codes back into this runner's JSONL/exit-code contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from viton_api.tryon_params import CATEGORIES, validate_params

PIPELINE_PROMPT = "Edit image 1 so the person wears the garment from image 2. Preserve the person's face, body shape, pose, hands, background, and framing. Preserve the garment's original color, fabric texture, pattern, embroidery, logos, neckline, sleeves, and length. Fit it naturally with realistic folds. Do not redesign or simplify the garment. Return one realistic photograph."
GARMENT_TYPE_PROMPTS = {
    "top": "The garment in image 2 is a TOP (shirt, t-shirt, blouse, kurti, sweater, or similar upper-body clothing). Replace only the person's upper-body clothing with it; keep the person's lower-body clothing unchanged. ",
    "bottom": "The garment in image 2 is BOTTOM wear (pants, trousers, jeans, shorts, or a skirt). Replace only the person's lower-body clothing with it; keep the person's upper-body clothing unchanged. ",
    "dress": "The garment in image 2 is a DRESS (one-piece garment covering torso and legs). Replace the person's entire outfit with it. ",
    "outerwear": "The garment in image 2 is OUTERWEAR (jacket, coat, hoodie, blazer, or similar). Put it on over the person's current outfit as the outermost layer. ",
}
CUSTOM_TEMPLATE = "Edit Image 1 so the person wears the {category} shown in Image 2. Preserve the product's design, color, material, and proportions. Place it naturally with appropriate scale and occlusion. Keep the person's identity, pose, existing clothing, and background unchanged."

_RUNNER_DIR = Path(__file__).resolve().parent
_DEFAULT_ACCESSORY_SCRIPT = (
    _RUNNER_DIR.parent.parent
    / "qwen21_tryon_test/try_on_gar/run_accessory_tryon.py"
)
# Human-readable attachment instructions are advisory under the crop-context
# implementation: it builds its own placement-aware prompt per category.
_ACCESSORY_COMPOSITE = "10_final_composite.png"
_ACCESSORY_META = "meta.json"


class UnitFailure(Exception):
    """This person/accessory pair cannot be processed; remaining pairs continue."""


class PoisonJobError(Exception):
    """The job itself is invalid (e.g. unsupported category); fail fast."""


class ResourceWaitError(Exception):
    """GPU/RAM pressure; the orchestrator should clean-requeue the delivery."""


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def prepare(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def load_pipeline(mode, model, lora):
    import torch

    if mode == "clothing":
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(
            str(model), torch_dtype=torch.bfloat16, local_files_only=True
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling(
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
        )
    else:
        # The accessory implementation loads its own pipeline per invocation
        # (subprocess per unit); there is no shared in-process pipeline.
        return None
    return pipe


def _visible_device_index() -> str:
    """Explicit CUDA index for CUDA_VISIBLE_DEVICES (the script's GPU gate).

    The crop-context script refuses to generate unless CUDA_VISIBLE_DEVICES
    names explicit valid indices, so derive one from VT_DEVICE (default cuda:0).
    """
    device = os.environ.get("VT_DEVICE", "cuda:0")
    digits = "".join(ch for ch in device.split(":")[-1] if ch.isdigit())
    return digits or "0"


def _spawn_accessory(cmd, env, cwd):
    """Run the accessory script, streaming its output through our stdout.

    Streaming (not capture-then-print) keeps the orchestrator's multi-signal
    stall detector fed during long generation phases. Returns the exit code
    and the combined transcript for post-mortem OOM detection.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
    )
    transcript = []
    assert proc.stdout is not None
    for line in proc.stdout:
        transcript.append(line)
        print(f"[accessory] {line.rstrip()}", flush=True)
    proc.wait()
    return proc.returncode, "".join(transcript)


def run_accessory_script(params, unit, out_dir):
    """Drive try_on_gar/run_accessory_tryon.py for one (person, accessory) pair.

    Exit-code contract of the script: 0 ok | 1 prepare failed | 2 GPU gate
    refused | 3 unsupported category | 4 landmarks needs-input | 5 preservation
    rejected | 6 sam3 check failed. Success artifact: 10_final_composite.png
    (full person, native resolution) + meta.json with status "ok".
    """
    script = Path(os.environ.get("VT_ACCESSORY_SCRIPT") or _DEFAULT_ACCESSORY_SCRIPT)
    mode = os.environ.get("VT_ACCESSORY_MODE", "crop-context")
    category = params["accessory_category"].strip()
    if params.get("contact_sentence"):
        print(
            "note: contact_sentence is advisory under the crop-context "
            "implementation and is not sent to the model: "
            f"{params['contact_sentence']!r}",
            flush=True,
        )
    # The script never overwrites an output dir: _fresh_dir() bumps the name
    # (suffix _1, _2, ...) whenever the path exists — including an empty dir.
    # So clean any stale scratch but do NOT create it; the passed path must be
    # non-existent for the script to use it verbatim (harvest relies on that).
    scratch = Path(out_dir) / f"accessory_{unit['ordinal']:04d}"
    if scratch.exists():
        shutil.rmtree(scratch)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _visible_device_index()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        mode,
        "--person",
        str(unit["person"]),
        "--accessory",
        str(unit["garment"]),
        "--category",
        category,
        "--output-dir",
        str(scratch),
    ]
    print(f"[accessory] launching: {' '.join(cmd)}", flush=True)
    code, transcript = _spawn_accessory(cmd, env, str(script.parent))
    lowered = transcript.lower()
    if "out of memory" in lowered or "outofmemoryerror" in lowered:
        raise ResourceWaitError("accessory script reported CUDA OOM")
    if code == 0:
        composite = scratch / _ACCESSORY_COMPOSITE
        meta_path = scratch / _ACCESSORY_META
        if not composite.is_file():
            raise UnitFailure(
                f"accessory script exited 0 but {_ACCESSORY_COMPOSITE} is missing"
            )
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        except ValueError:
            meta = {}
        if str(meta.get("status", "ok")).lower() != "ok":
            raise UnitFailure(
                f"accessory script exited 0 but meta status is {meta.get('status')!r}"
            )
        from PIL import Image

        with Image.open(composite) as result:
            result.load()  # detach from the scratch dir before cleanup
            image = result.copy()
        shutil.rmtree(scratch, ignore_errors=True)
        return image
    if code == 4:
        raise UnitFailure(
            "needs-input: MediaPipe could not place the accessory on this "
            "person photo (landmark detection failed; a closer photo may help)"
        )
    if code == 5:
        raise UnitFailure(
            "rejected: preservation/quality checks failed for this pair"
        )
    if code == 1:
        raise UnitFailure("prepare failed: invalid person/accessory image")
    if code == 3:
        emit(
            "fatal_poison",
            error=(
                f"unsupported accessory category {category!r} for the "
                "crop-context implementation"
            ),
        )
        raise PoisonJobError(category)
    if code == 2:
        raise ResourceWaitError("accessory script GPU gate refused to run")
    raise RuntimeError(f"accessory script exited with unexpected code {code}")


def infer(pipe, params, person, product, unit=None, out_dir=None):
    mode = params.get("tryon_mode", "clothing")
    if mode == "clothing":
        import torch

        category = CATEGORIES[params.get("garment_type") or "auto"]
        prompt = GARMENT_TYPE_PROMPTS.get(category, "") + PIPELINE_PROMPT
        if params.get("garment_prompt"):
            prompt += " " + params["garment_prompt"].strip()
        return pipe(
            prompt=prompt,
            image=[person, product],
            width=1792,
            height=2400,
            num_inference_steps=40,
            true_cfg_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
    if unit is None or out_dir is None:
        raise UnitFailure("accessory mode requires unit paths (unit/out_dir)")
    return run_accessory_script(params, unit, out_dir)


def run(job, loader=load_pipeline, generate=infer):
    params = job.get("params") or {}
    error = validate_params(params)
    units = job.get("units") or []
    if error or not units:
        emit("fatal_poison", error=error or "no image units")
        return 2
    ordinals = [u.get("ordinal") for u in units]
    if any(type(o) is not int or o < 0 for o in ordinals) or len(set(ordinals)) != len(
        ordinals
    ):
        emit("fatal_poison", error="invalid or duplicate ordinals")
        return 2
    import psutil

    if psutil.virtual_memory().available < 45 * 1024**3:
        emit("resource_wait", error="Qwen needs at least 45 GiB available RAM")
        return 3
    out = Path(job["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mode = params.get("tryon_mode", "clothing")
    model = Path(os.environ["VT_QWEN_MODEL_DIR"])
    lora = Path(os.environ["VT_ACCESSORY_LORA_PATH"])
    pipe = None
    done = 0
    for unit in units:
        ordinal = unit["ordinal"]
        try:
            person, product = prepare(unit["person"]), prepare(unit["garment"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            emit("image_failed", ordinal=ordinal, error=f"invalid image: {exc}")
            continue
        try:
            if pipe is None:
                pipe = loader(mode, model, lora)
            image = generate(pipe, params, person, product, unit, out)
            dest = out / f"{ordinal:04d}.png"
            temporary = out / f"{ordinal:04d}.partial.png"
            image.save(temporary, format="PNG")
            from PIL import Image

            with Image.open(temporary) as saved:
                saved.verify()
            temporary.replace(dest)
            emit("image_done", ordinal=ordinal, path=str(dest))
            done += 1
        except UnitFailure as exc:
            # One bad pair (needs-input / rejected / invalid input) must not
            # sink the remaining pairs in the batch.
            emit("image_failed", ordinal=ordinal, error=str(exc))
            continue
        except PoisonJobError:
            # fatal_poison was already emitted by the helper.
            return 2
        except ResourceWaitError as exc:
            emit("resource_wait", error=str(exc))
            return 3
        except Exception as exc:  # noqa: BLE001 -- translate ML failures into runner exit codes
            traceback.print_exc()
            if (
                "out of memory" in str(exc).lower()
                or type(exc).__name__ == "OutOfMemoryError"
            ):
                return 3
            # A broken pipeline must not turn every remaining pair into a failure.
            return 1
        finally:
            # Match the supplied clothing batch runner: keep allocator caches
            # from accumulating across a long job, including after CUDA OOM.
            torch = sys.modules.get("torch")
            if torch is not None:
                torch.cuda.empty_cache()
    return 0 if done else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()
    try:
        code = run(json.loads(Path(args.job_json).read_text()))
    except Exception:  # noqa: BLE001 -- CLI boundary must return a retryable exit code
        traceback.print_exc()
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
