#!/usr/bin/env python
"""refton runner — virtual try-on via RefTon/inference.py (FLUX.1-Kontext + VOSR2 4x).

Invoked by the orchestrator as:
    conda run --no-capture-output -n refton python <runners>/refton_runner.py \
        --job-json <workdir>/job.json

What it does:
1. validates + normalizes the staged person/cloth pairs into an `instance/`
   tree (PIL -> RGB .png, names {ordinal:04d}.png so cloth/person pair by
   stem — MyDataset pairs by identical filename, .jpg/.png only);
2. optionally writes garments.txt (free-text garment_prompt, keyed by stem);
3. spawns `python inference.py --instance_data_dir <instance>` with
   cwd = RefTon (the model/LoRA paths there are cwd-relative) and streams its
   output through (transcript for job.log) while parsing progress;
4. copies each finished 4K file to <workdir>/output/{ordinal:04d}.png.

Resolution/quality knobs are inference.py's defaults (1024x768, cond_scale
2.0, 1024_768 LoRA, seed 0, cpu_offload, VOSR2 x4) — nothing to pass.

Guard: inference.py routes datasets by CASE-SENSITIVE substrings
("viton"/"DressCode") of the instance path. Our paths never contain them, but
if one ever would, the routing would silently switch — refuse instead (exit 2).

Exit codes / progress protocol: identical to ai_human_runner.py, plus
    {"event": "image_generated", "ordinal": N}   (cumulative "Generated N images" line)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# located at <service>/runners/refton_runner.py -> parents[2] = ViTOn/
_DEFAULT_REFTON_DIR = Path(__file__).resolve().parents[2] / "RefTon"

_GEN_RE = re.compile(r"Generated (\d+) images")
_SAVED_RE = re.compile(r"Saved:\s*(.+?)\s*\|\s*resolution:")


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def die_poison(message: str) -> NoReturn:
    emit({"event": "fatal_poison", "error": message})
    sys.exit(2)


def normalize_to(src: str | None, dest: Path) -> bool:
    """PIL-open any input format, force RGB, save as .png. False if src missing."""
    if not src or not Path(src).exists():
        return False
    from PIL import Image
    with Image.open(src) as im:
        im.convert("RGB").save(dest, format="PNG")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", required=True)
    args = parser.parse_args()

    job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    params = job.get("params") or {}
    units = job.get("units") or []
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir.parent
    instance = workdir / "instance"

    if not units:
        die_poison("no image units in job.json")
    garment_type = params.get("garment_type") or ""
    garment_prompt = params.get("garment_prompt") or ""
    if garment_type and garment_prompt:
        die_poison("provide either garment_type or garment_prompt, not both")

    refton_dir = Path(os.environ.get("VT_REFTON_DIR") or _DEFAULT_REFTON_DIR)
    if not (refton_dir / "inference.py").exists():
        die_poison(f"RefTon tree not found at {refton_dir}")

    # case-sensitive raw-path guard: inference.py sniffs these substrings to
    # pick a dataset class; a hit would silently change the pipeline.
    raw = str(instance)
    if "viton" in raw or "DressCode" in raw:
        die_poison(f"instance path contains a dataset-routing keyword: {raw}")

    # 1. normalize pairs
    (instance / "cloth").mkdir(parents=True, exist_ok=True)
    (instance / "person").mkdir(parents=True, exist_ok=True)
    for unit in units:
        ordinal = int(unit["ordinal"])
        name = f"{ordinal:04d}.png"
        if not normalize_to(unit.get("person"), instance / "person" / name):
            die_poison(f"ordinal {ordinal}: person input missing")
        if not normalize_to(unit.get("garment"), instance / "cloth" / name):
            die_poison(f"ordinal {ordinal}: garment input missing")

    # 2. per-image free-text prompts (presets go through --garment instead)
    if garment_prompt:
        (instance / "garments.txt").write_text(
            "\n".join(f"{int(u['ordinal']):04d} {garment_prompt}" for u in units) + "\n",
            encoding="utf-8",
        )

    # 3. run inference.py from the RefTon tree
    cmd = [sys.executable, "inference.py", "--instance_data_dir", str(instance)]
    if garment_type:
        cmd += ["--garment", garment_type]
    if len(units) > 1 or garment_prompt:
        # garments.txt may carry per-image prompts -> never batch across images
        cmd += ["--inference_batch_size", "1"]
    seed = params.get("seed")
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    print(f"[refton_runner] cwd={refton_dir}", flush=True)
    print(f"[refton_runner] cmd={cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(refton_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None

    copied: set[int] = set()
    oom = False
    tail: list[str] = []

    def copy_final(src: Path, ordinal: int) -> None:
        dest = out_dir / f"{ordinal:04d}.png"
        shutil.copy2(str(src), str(dest))
        copied.add(ordinal)
        emit({"event": "image_done", "ordinal": ordinal, "path": str(dest)})

    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)  # transcript for job.log (orchestrator pumps us)
        tail.append(line)
        tail = tail[-40:]
        m = _GEN_RE.search(line)
        if m:
            n = int(m.group(1))
            if n > 0:
                emit({"event": "image_generated", "ordinal": n - 1})
            continue
        m = _SAVED_RE.search(line)
        if m:
            src = Path(m.group(1))
            try:
                ordinal = int(src.stem)
            except ValueError:
                continue
            if src.exists() and ordinal not in copied:
                copy_final(src, ordinal)
            continue
        low = line.lower()
        if "out of memory" in low or "outofmemoryerror" in low:
            oom = True

    rc = proc.wait()

    # 4. sweep: any finished file the "Saved:" lines missed (e.g. VOSR skipped
    #    because the file was already upscaled on a redelivery).
    sample_dir = instance / "sample_person"
    if sample_dir.exists():
        for src in sorted(sample_dir.glob("*.png")):
            try:
                ordinal = int(src.stem)
            except ValueError:
                continue
            if ordinal not in copied:
                copy_final(src, ordinal)

    if oom:
        sys.exit(3)
    if rc != 0:
        print(f"inference.py exited with code {rc}", file=sys.stderr, flush=True)
        sys.exit(1)
    if not copied:
        print("no images produced", file=sys.stderr, flush=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
