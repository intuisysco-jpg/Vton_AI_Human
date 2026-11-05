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


# --- accessory mode: the crop-context subprocess adapter ----------------- #


def _accessory_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VT_QWEN_MODEL_DIR", "/models")
    monkeypatch.setenv("VT_ACCESSORY_LORA_PATH", "/lora")
    monkeypatch.setenv("VT_ACCESSORY_SCRIPT", str(tmp_path / "run_accessory_tryon.py"))
    monkeypatch.setenv("VT_ACCESSORY_MODE", "crop-context")
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    Image.new("RGB", (40, 80)).save(tmp_path / "person.png")
    Image.new("RGB", (20, 20)).save(tmp_path / "accessory.png")


def _events(capsys):
    out = []
    for line in capsys.readouterr().out.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # plain log lines (e.g. "[accessory] ...") are not events
    return out


def _fake_spawn(codes_by_ordinal, meta_status="ok"):
    """Fake _spawn_accessory: writes success artifacts, returns scripted codes."""
    calls = []

    def spawn(cmd, env, cwd):
        calls.append((cmd, env))
        ordinal = int(Path(cmd[cmd.index("--output-dir") + 1]).name.split("_")[1])
        scratch = Path(cmd[cmd.index("--output-dir") + 1])
        scratch.mkdir(parents=True, exist_ok=True)
        code = codes_by_ordinal.get(ordinal, 0)
        if code == 0:
            Image.new("RGB", (40, 80)).save(scratch / "10_final_composite.png")
            (scratch / "meta.json").write_text(json.dumps({"status": meta_status}))
        return code, f"fake transcript for {ordinal}\n"

    return spawn, calls


def _accessory_units(tmp_path, ordinals):
    return [
        {
            "ordinal": o,
            "person": str(tmp_path / "person.png"),
            "garment": str(tmp_path / "accessory.png"),
        }
        for o in ordinals
    ]


def test_accessory_success_harvests_composite_and_cleans_scratch(
    tmp_path, monkeypatch, capsys
):
    _accessory_env(monkeypatch, tmp_path)
    spawn, calls = _fake_spawn({5: 0})
    monkeypatch.setattr(runner, "_spawn_accessory", spawn)
    units = _accessory_units(tmp_path, [5])
    job = {
        "params": {"tryon_mode": "accessory", "accessory_category": "earring"},
        "units": units,
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 0
    events = _events(capsys)
    assert [(e["event"], e["ordinal"]) for e in events] == [("image_done", 5)]
    # The contract PNG lands in the output dir; the script scratch is removed.
    assert (tmp_path / "out" / "0005.png").is_file()
    assert not (tmp_path / "out" / "accessory_0005").exists()
    # The script is invoked exactly once with the canonical shape.
    cmd = calls[0][0]
    assert cmd[cmd.index("--category") + 1] == "earring"
    assert cmd[cmd.index("--person") + 1] == str(tmp_path / "person.png")
    assert cmd[cmd.index("--accessory") + 1] == str(tmp_path / "accessory.png")
    assert cmd[cmd.index("--mode") + 1] == "crop-context"
    # The child env carries an explicit CUDA device (the script's GPU gate).
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "0"


def test_accessory_needs_input_fails_pair_but_batch_continues(
    tmp_path, monkeypatch, capsys
):
    _accessory_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner, "_spawn_accessory", _fake_spawn({2: 4, 7: 0})[0]
    )
    units = _accessory_units(tmp_path, [2, 7])
    job = {
        "params": {"tryon_mode": "accessory", "accessory_category": "nose_pin"},
        "units": units,
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 0
    events = _events(capsys)
    assert [(e["event"], e["ordinal"]) for e in events] == [
        ("image_failed", 2),
        ("image_done", 7),
    ]
    assert "needs-input" in events[0]["error"]


def test_accessory_unsupported_category_is_poison(tmp_path, monkeypatch, capsys):
    _accessory_env(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_spawn_accessory", _fake_spawn({0: 3})[0])
    job = {
        "params": {"tryon_mode": "accessory", "accessory_category": "monocle"},
        "units": _accessory_units(tmp_path, [0]),
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 2
    events = _events(capsys)
    assert events[0]["event"] == "fatal_poison"
    assert "monocle" in events[0]["error"]


def test_accessory_gpu_gate_refusal_clean_requeues(tmp_path, monkeypatch, capsys):
    _accessory_env(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_spawn_accessory", _fake_spawn({0: 2})[0])
    job = {
        "params": {"tryon_mode": "accessory", "accessory_category": "bangle"},
        "units": _accessory_units(tmp_path, [0]),
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 3
    assert _events(capsys)[0]["event"] == "resource_wait"


def test_accessory_contact_sentence_is_advisory_and_not_forwarded(
    tmp_path, monkeypatch, capsys
):
    _accessory_env(monkeypatch, tmp_path)
    spawn, calls = _fake_spawn({1: 0})
    monkeypatch.setattr(runner, "_spawn_accessory", spawn)
    sentence = "The earring post passes through the earlobe piercing."
    job = {
        "params": {
            "tryon_mode": "accessory",
            "accessory_category": "earring",
            "contact_sentence": sentence,
        },
        "units": _accessory_units(tmp_path, [1]),
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 0
    plain = capsys.readouterr().out
    assert "advisory" in plain
    assert sentence in plain
    cmd = " ".join(calls[0][0])
    assert sentence not in cmd  # never reaches the model prompt


def test_accessory_non_ok_meta_status_fails_the_pair(tmp_path, monkeypatch, capsys):
    _accessory_env(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_spawn_accessory", _fake_spawn({0: 0}, meta_status="needs-input")[0])
    job = {
        "params": {"tryon_mode": "accessory", "accessory_category": "cap"},
        "units": _accessory_units(tmp_path, [0]),
        "output_dir": str(tmp_path / "out"),
    }
    assert runner.run(job, lambda *a: None) == 2  # nothing produced
    events = _events(capsys)
    assert events[0]["event"] == "image_failed"
    assert "needs-input" in events[0]["error"]
