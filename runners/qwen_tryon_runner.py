#!/usr/bin/env python
"""Production adapter of qwen21_tryon_test.py and product_tryon.py/run_g2l.py.

Keeps the verified model settings, but consumes explicit ordinal pairs rather
than folder Cartesian products. One selected pipeline is loaded per job.
"""

from __future__ import annotations

import argparse
import json
import os
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
        from diffsynth.pipelines.qwen_image_21 import ModelConfig, QwenImage21Pipeline

        configs = []
        for pattern in (
            "transformer/diffusion_pytorch_model*.safetensors",
            "text_encoder/model*.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ):
            paths = sorted(model.glob(pattern))
            if not paths:
                raise FileNotFoundError(f"No weights matching {pattern}")
            configs.append(
                ModelConfig(
                    path=[str(p) for p in paths] if len(paths) > 1 else str(paths[0]),
                    offload_device="cpu",
                    offload_dtype=torch.bfloat16,
                )
            )
        pipe = QwenImage21Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=configs,
            vram_limit=24.0,
            processor_config=ModelConfig(path=str(model / "processor")),
        )
        pipe.load_lora(pipe.dit, str(lora))
    return pipe


def infer(pipe, params, person, product):
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
    from diffsynth.core.data.operators import ImageCropAndResize

    for im in (person, product):
        if max(im.size) > 2048:
            from PIL import Image

            im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    person = ImageCropAndResize(
        max_pixels=1048576, height_division_factor=32, width_division_factor=32
    )(person)
    prompt = CUSTOM_TEMPLATE.format(category=params["accessory_category"].strip())
    if params.get("contact_sentence"):
        prompt += " " + params["contact_sentence"].strip()
    return pipe(
        prompt,
        edit_image=[person, product],
        seed=0,
        num_inference_steps=40,
        cfg_scale=1.0,
        height=person.height,
        width=person.width,
    )


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
            image = generate(pipe, params, person, product)
            dest = out / f"{ordinal:04d}.png"
            temporary = out / f"{ordinal:04d}.partial.png"
            image.save(temporary, format="PNG")
            from PIL import Image

            with Image.open(temporary) as saved:
                saved.verify()
            temporary.replace(dest)
            emit("image_done", ordinal=ordinal, path=str(dest))
            done += 1
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
