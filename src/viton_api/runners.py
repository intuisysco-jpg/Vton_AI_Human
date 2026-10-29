"""Stage specifications: one per pipeline (ai_human, viton).

Each stage knows:
- its conda env (env_attr -> Settings field),
- its subprocess cwd / probe tree (tree_attr -> Settings field),
- its runner script (in <service>/runners/, so the ML trees stay pristine),
- how to build the `conda run ... python <runner> --job-json <file>` command.

Per-job context (reference, params, staged unit inputs, output dir) is handed
to the runner through a JSON file the orchestrator writes into the workdir —
the runner emits JSONL progress events back on stdout (see orchestrator).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings


@dataclass
class StageContext:
    workdir: Path               # per-job scratch dir (inputs/, output/, job.json)
    reference: str
    params: dict
    units: list[dict]           # staged per-ordinal inputs (abs paths)
    settings: Settings
    job_json: Path | None = None

    @classmethod
    def for_job(cls, workdir, reference: str, params: dict, units: list[dict],
                settings: Settings) -> "StageContext":
        return cls(workdir=Path(workdir), reference=reference, params=params,
                   units=units, settings=settings)

    @property
    def out_dir(self) -> Path:
        return self.workdir / "output"

    def write_job_json(self) -> Path:
        """Serialize the job contract for the runner subprocess."""
        payload = {
            "reference": self.reference,
            "params": self.params,
            "output_dir": str(self.out_dir),
            "units": self.units,
        }
        path = self.workdir / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self.job_json = path
        return path


class StageSpec:
    # subclass overrides
    key: str = ""
    required: bool = True
    runner_script: str = ""       # filename under Settings.runners_dir
    probe_imports: str = ""       # import snippet for preflight
    env_attr: str = "conda_env"   # Settings field -> this stage's conda env
    tree_attr: str = "ai_human_dir"  # Settings field -> subprocess cwd / probe tree

    def extra_probe_paths(self, settings: Settings) -> list[Path]:
        """Model paths preflight verifies before the env import probe."""
        return []

    def _script(self, ctx: StageContext) -> str:
        return str(Path(ctx.settings.runners_dir) / self.runner_script)

    def _cwd(self, ctx: StageContext) -> str:
        return str(getattr(ctx.settings, self.tree_attr))

    def build_cmd(self, ctx: StageContext) -> list[str]:
        job_json = ctx.job_json or ctx.write_job_json()
        return ctx.settings.conda_command("python", self._script(ctx), "--job-json", str(job_json))


class AiHumanStage(StageSpec):
    """Text/reference -> image (FLUX.2-klein in ai_human/, generate_human.py)."""
    key = "ai_human"
    runner_script = "ai_human_runner.py"
    env_attr = "conda_env"
    tree_attr = "ai_human_dir"  # cwd = ai_human/ (model dir resolved relative)
    probe_imports = (
        "import sys; sys.path.insert(0, '.'); import generate_human; "
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    )

    def extra_probe_paths(self, settings: Settings) -> list[Path]:
        base = Path(settings.ai_human_dir)
        return [
            base / "generate_human.py",
            base / "models" / "FLUX.2-klein-4B",
        ]


class VitonStage(StageSpec):
    """Virtual try-on (FLUX.1-Kontext + LoRA in RefTon/inference.py, VOSR2 4x)."""
    key = "viton"
    runner_script = "refton_runner.py"
    env_attr = "conda_env"
    tree_attr = "refton_dir"  # cwd = RefTon/ (relative model/LoRA paths!)
    probe_imports = (
        "import sys; sys.path.insert(0, '.'); "
        "from refton.pipelines import FluxKontextPipelineI2I; "
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
    )

    def extra_probe_paths(self, settings: Settings) -> list[Path]:
        base = Path(settings.refton_dir)
        return [
            base / "inference.py",
            base / "models" / "FLUX.1-Kontext-dev",
            base / "checkpoints" / "1024_768_pytorch_lora_weights.safetensors",
            Path(settings.vosr_dir) / "inference_vosr_onestep.py",
            Path(settings.vosr_dir) / "preset" / "ckpts" / "VOSR2",
        ]


STAGES: list[StageSpec] = [AiHumanStage(), VitonStage()]
STAGE_BY_KEY: dict[str, StageSpec] = {s.key: s for s in STAGES}
