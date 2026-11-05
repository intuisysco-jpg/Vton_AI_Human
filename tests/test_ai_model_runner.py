"""ai_model runner contract tests — process_fn injected, no GPU/import of pipeline."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "ai_model_runner", Path(__file__).parents[1] / "runners/ai_model_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _unit(tmp_path, ordinal=0, missing=False):
    person = tmp_path / f"person_{ordinal}.png"
    if not missing:
        Image.new("RGB", (40, 80)).save(person)
    return {
        "ordinal": ordinal,
        "original_filename": person.name,
        "person": str(person),
    }


def _job(tmp_path, units, params=None):
    return {
        "reference": "am_00001",
        "params": params or {},
        "output_dir": str(tmp_path / "out"),
        "units": units,
    }


def _events(capsys):
    out = []
    for line in capsys.readouterr().out.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # plain transcript lines are not events
    return out


def _ok_process(final_from):
    calls = []

    def process(**kwargs):
        calls.append(kwargs)
        final = final_from(kwargs["input_path"])
        return {"status": "accepted", "final_image": str(final),
                "total_seconds": 80.0}

    return process, calls


def test_accepted_unit_emits_done_and_writes_contract_png(tmp_path, capsys):
    Image.new("RGB", (40, 80)).save(tmp_path / "final.png")
    process, calls = _ok_process(lambda p: tmp_path / "final.png")
    job = _job(tmp_path, [_unit(tmp_path, 0)])
    assert runner.run(job, process) == 0
    events = _events(capsys)
    assert [(e["event"], e["ordinal"]) for e in events] == [("image_done", 0)]
    assert (tmp_path / "out" / "0000.png").is_file()
    # seed defaults to 42 and is offset by the ordinal
    assert calls[0]["seed"] == 42
    assert calls[0]["max_retries"] == 2


def test_seed_offset_per_ordinal_and_env_retries(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VT_AI_MODEL_MAX_RETRIES", "4")
    Image.new("RGB", (40, 80)).save(tmp_path / "final.png")
    process, calls = _ok_process(lambda p: tmp_path / "final.png")
    job = _job(tmp_path, [_unit(tmp_path, i) for i in (0, 3)],
               params={"seed": 100})
    assert runner.run(job, process) == 0
    assert [c["seed"] for c in calls] == [100, 103]
    assert all(c["max_retries"] == 4 for c in calls)


def test_manual_review_and_skip_fail_units_but_batch_continues(tmp_path, capsys):
    Image.new("RGB", (40, 80)).save(tmp_path / "final.png")
    statuses = {0: ("manual_review", ["face mismatch", "hair seam"]),
                1: ("accepted", None)}

    def process(**kwargs):
        status, reasons = statuses[len(statuses_done)]
        statuses_done.append(1)
        if status == "accepted":
            return {"status": "accepted",
                    "final_image": str(tmp_path / "final.png")}
        return {"status": status, "qc_reasons": reasons}

    statuses_done = []
    job = _job(tmp_path, [_unit(tmp_path, 0), _unit(tmp_path, 1)])
    assert runner.run(job, process) == 0
    events = _events(capsys)
    assert [(e["event"], e["ordinal"]) for e in events] == [
        ("image_failed", 0),
        ("image_done", 1),
    ]
    assert "manual_review" in events[0]["error"]
    assert "face mismatch" in events[0]["error"]


def test_all_units_fail_returns_retry_code(tmp_path, capsys):
    def process(**kwargs):
        return {"status": "skip_no_head"}

    job = _job(tmp_path, [_unit(tmp_path, 0), _unit(tmp_path, 1)])
    assert runner.run(job, process) == 1
    events = _events(capsys)
    assert all(e["event"] == "image_failed" for e in events)


def test_oom_breaks_loop_and_returns_requeue_code(tmp_path, capsys):
    seen = []

    def process(**kwargs):
        seen.append(kwargs["input_path"])
        raise RuntimeError("CUDA out of memory during inpaint")

    job = _job(tmp_path, [_unit(tmp_path, 0), _unit(tmp_path, 1)])
    assert runner.run(job, process) == 3
    assert len(seen) == 1  # loop breaks after the first OOM
    events = _events(capsys)
    assert events[0]["event"] == "image_failed"
    assert "OOM" in events[0]["error"]


def test_bad_params_is_poison(tmp_path):
    job = _job(tmp_path, [_unit(tmp_path, 0)], params={"seed": "forty-two"})
    with pytest.raises(SystemExit) as exc:
        runner.run(job, lambda **k: {})
    assert exc.value.code == 2


def test_missing_person_input_is_poison(tmp_path):
    job = _job(tmp_path, [_unit(tmp_path, 0, missing=True)])
    with pytest.raises(SystemExit) as exc:
        runner.run(job, lambda **k: {})
    assert exc.value.code == 2


def test_accepted_without_final_image_fails_unit(tmp_path, capsys):
    def process(**kwargs):
        return {"status": "accepted", "final_image": "/nonexistent/final.png"}

    job = _job(tmp_path, [_unit(tmp_path, 0)])
    assert runner.run(job, process) == 1
    events = _events(capsys)
    assert events[0]["event"] == "image_failed"
    assert "final_image missing" in events[0]["error"]
