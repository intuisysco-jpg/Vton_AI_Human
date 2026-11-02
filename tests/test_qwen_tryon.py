import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from viton_api.config import Settings
from viton_api.runners import VitonStage
from viton_api.tryon_params import validate_params

spec = importlib.util.spec_from_file_location(
    "qwen_runner", Path(__file__).parents[1] / "runners/qwen_tryon_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"garment_type": "top"},
        {"garment_prompt": "Keep stripes"},
        {"tryon_mode": "accessory", "accessory_category": "earring"},
    ],
)
def test_valid_params(params):
    assert validate_params(params) is None


@pytest.mark.parametrize(
    "params",
    [
        {"tryon_mode": "bad"},
        {"tryon_mode": "accessory"},
        {"accessory_category": "ring"},
        {
            "tryon_mode": "accessory",
            "accessory_category": "ring",
            "garment_type": "top",
        },
        {"garment_type": "top", "garment_prompt": "x"},
    ],
)
def test_invalid_params(params):
    assert validate_params(params)


def test_explicit_pairs_partial_and_noncontiguous_ordinals(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("VT_QWEN_MODEL_DIR", "/models")
    monkeypatch.setenv("VT_ACCESSORY_LORA_PATH", "/lora")
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    for name in ("p", "g"):
        Image.new("RGB", (32, 64)).save(tmp_path / f"{name}.png")
    units = [
        {
            "ordinal": o,
            "person": str(tmp_path / "p.png"),
            "garment": str(tmp_path / "g.png"),
        }
        for o in (2, 7)
    ]
    units.insert(1, {"ordinal": 4, "person": "/missing", "garment": "/missing"})
    loads = []

    def load(*args):
        loads.append(args)
        return object()

    assert (
        runner.run(
            {"output_dir": str(tmp_path / "out"), "units": units},
            load,
            lambda *args: Image.new("RGB", (32, 64)),
        )
        == 0
    )
    events = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    assert [(e["event"], e["ordinal"]) for e in events] == [
        ("image_done", 2),
        ("image_failed", 4),
        ("image_done", 7),
    ]
    assert len(loads) == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "0002.png",
        "0007.png",
    ]


def test_duplicate_ordinal_rejected_before_load(tmp_path):
    assert (
        runner.run(
            {"units": [{"ordinal": 0}, {"ordinal": 0}], "output_dir": str(tmp_path)},
            lambda *a: pytest.fail("loaded"),
        )
        == 2
    )


def test_missing_model_shard_blocks_preflight(tmp_path):
    (tmp_path / "transformer").mkdir()
    (
        tmp_path / "transformer/diffusion_pytorch_model.safetensors.index.json"
    ).write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    config = Settings(_env_file=None, qwen_model_dir=tmp_path)
    assert (
        tmp_path / "transformer/missing.safetensors"
        in VitonStage().extra_probe_paths(config)
    )


def test_clothing_preset_mapping_and_preservation_prompt():
    calls = []

    class Fake:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=["image"])

    # CPU-only substitute for the generator; no model is loaded.
    import sys

    fake = SimpleNamespace(
        Generator=lambda **k: SimpleNamespace(manual_seed=lambda n: n)
    )
    old = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        assert (
            runner.infer(Fake(), {"garment_type": "full_pant"}, None, None) == "image"
        )
    finally:
        if old is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = old
    assert calls[0]["prompt"].startswith(runner.GARMENT_TYPE_PROMPTS["bottom"])
    assert calls[0]["image"] == [None, None]
    assert calls[0]["width"] == 1792 and calls[0]["height"] == 2400


def test_24_pairs_emit_exactly_24_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VT_QWEN_MODEL_DIR", "/models")
    monkeypatch.setenv("VT_ACCESSORY_LORA_PATH", "/lora")
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    Image.new("RGB", (32, 64)).save(tmp_path / "input.png")
    units = [
        {
            "ordinal": i,
            "person": str(tmp_path / "input.png"),
            "garment": str(tmp_path / "input.png"),
        }
        for i in range(24)
    ]
    assert (
        runner.run(
            {"units": units, "output_dir": str(tmp_path / "output")},
            lambda *a: object(),
            lambda *a: Image.new("RGB", (32, 64)),
        )
        == 0
    )
    assert len([json.loads(s) for s in capsys.readouterr().out.splitlines()]) == 24
    assert len(list((tmp_path / "output").glob("*.png"))) == 24


@pytest.mark.asyncio
async def test_cancel_reaps_subprocess_tree(tmp_path):
    import asyncio
    import os
    import sys

    from viton_api.logging_setup import JobLog
    from viton_api.orchestrator import Orchestrator

    config = Settings(_env_file=None, cancel_poll_interval=0.01)
    orchestrator = Orchestrator(config, None, None)
    log = JobLog(tmp_path / "job.log")
    log.open()
    pidfile = tmp_path / "pid"
    cmd = [
        sys.executable,
        "-c",
        f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)',
    ]
    task = asyncio.create_task(
        orchestrator._run_cmd(
            SimpleNamespace(key="viton"), cmd, str(tmp_path), log, 120
        )
    )
    try:
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.01)
        pid = int(pidfile.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        log.close()


@pytest.mark.asyncio
async def test_delivery_deadline_returns_clean_requeue_inside_fencing(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from viton_api.consumer import _DELIVERY_DEADLINE, AckDecision, JobProcessor

    processor = JobProcessor(Settings(_env_file=None), SimpleNamespace(), None)
    processor._process_job = AsyncMock(side_effect=lambda *a: None)

    async def slow(*args):
        await asyncio.sleep(10)

    processor._process_job = slow
    token = _DELIVERY_DEADLINE.set(asyncio.get_running_loop().time() + 0.01)
    try:
        assert await processor.process("vt_12345", "viton") == AckDecision.NACK_REQUEUE
    finally:
        _DELIVERY_DEADLINE.reset(token)


@pytest.mark.asyncio
async def test_gpu_wait_observes_user_cancel_and_releases_descriptor(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    from viton_api.gpu_lock import GpuLease, GpuWaitCancelled

    settings = Settings(
        _env_file=None, gpu_lock_path=tmp_path / "gpu.lock", cancel_poll_interval=0.01
    )
    holder = GpuLease(settings)
    await holder.acquire()
    waiter = GpuLease(settings)
    check = AsyncMock(side_effect=[False, True])
    try:
        with pytest.raises(GpuWaitCancelled):
            await waiter.acquire(cancel_check=check)
        assert not waiter.acquired
    finally:
        holder.release()
    await asyncio.wait_for(waiter.acquire(), timeout=0.1)
    waiter.release()


def test_allocator_cache_is_released_after_each_success(tmp_path, monkeypatch):
    import sys
    from unittest.mock import Mock

    empty = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty))
    )
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    monkeypatch.setenv("VT_QWEN_MODEL_DIR", "/models")
    monkeypatch.setenv("VT_ACCESSORY_LORA_PATH", "/lora")
    Image.new("RGB", (32, 64)).save(tmp_path / "input.png")
    units = [
        {
            "ordinal": i,
            "person": str(tmp_path / "input.png"),
            "garment": str(tmp_path / "input.png"),
        }
        for i in (0, 1)
    ]
    assert (
        runner.run(
            {"units": units, "output_dir": str(tmp_path / "output")},
            lambda *a: object(),
            lambda *a: Image.new("RGB", (32, 64)),
        )
        == 0
    )
    assert empty.call_count == 2
