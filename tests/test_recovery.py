import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
import httpx
import pytest

from viton_api import main, preflight as preflight_module
from viton_api.config import Settings
from viton_api.consumer import AckDecision, JobProcessor, RabbitConsumer
from viton_api.diagnostics import safe_reason
from viton_api.preflight import Preflight
from viton_api.routers.health import router
from viton_api.runners import STAGES, StageContext


def settings(**values):
    return Settings(_env_file=None, **values)


def state(config=None, check=None, consumer=None):
    config = config or settings()
    return SimpleNamespace(
        settings=config,
        preflight=check or Preflight(config),
        consumer=consumer,
        db_connected=True,
        rabbit_connected=False,
        consumer_connected=False,
        gpu=None,
        pool=None,
        store=None,
        gpu_reporter=None,
        startup_error=None,
        stopping=False,
    )


def test_one_environment_drives_probes_runners_and_child_environment(
    tmp_path, monkeypatch
):
    config = settings()
    assert config.conda_env == "qwen21"
    monkeypatch.setenv("VT_CONDA_ENV", "old-runtime")
    monkeypatch.setenv("VT_AI_HUMAN_DIR", "old-tree")
    env = config.conda_subprocess_env()
    assert env["VT_CONDA_ENV"] == "qwen21"
    assert env["VT_AI_HUMAN_DIR"] == str(config.ai_human_dir)
    assert env["VT_VOSR_DIR"] == str(config.vosr_dir)
    context = StageContext.for_job(tmp_path, "test-job", {}, [], config)
    for stage in STAGES:
        assert stage.build_cmd(context)[:6] == [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "qwen21",
            "python",
        ]


def test_diagnostics_redact_secrets():
    config = settings(DATABASE_URL="postgresql://admin:test-password@db/prod")
    message = safe_reason(
        "postgresql://admin:test-password@db/prod hf_1234567890 token=secret", config
    )
    assert "test-password" not in message and "1234567890" not in message
    assert "token=secret" not in message
    assert len(safe_reason("x" * 1000, config)) == 300


async def test_cached_readiness_requires_actual_subscriptions(monkeypatch):
    config = settings()
    check = Preflight(config)
    check.availability = {"ai_human": True, "viton": True}
    check.reasons = {"ai_human": "", "viton": ""}
    check.probe_all = AsyncMock(side_effect=AssertionError("health polled preflight"))
    consumer = SimpleNamespace(
        broker_connected=True, connected=False, last_error="Subscription pending"
    )
    app = FastAPI()
    app.include_router(router)
    for key, value in vars(state(config, check, consumer)).items():
        setattr(app.state, key, value)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["rabbit_connected"] is True
        assert response.json()["consumer_connected"] is False
        consumer.connected = True
        assert (await client.get("/health/ready")).status_code == 200
        app.state.db_connected = False
        assert (await client.get("/health/ready")).status_code == 503
        app.state.db_connected = True
        app.state.stopping = True
        assert (await client.get("/health/ready")).status_code == 503
    check.probe_all.assert_not_awaited()


def fake_stages(tmp_path):
    output = []
    for key, tree in (("ai_human", "ai_human_dir"), ("viton", "refton_dir")):
        (tmp_path / f"{key}.py").touch()
        output.append(
            SimpleNamespace(
                key=key,
                tree_attr=tree,
                runner_script=f"{key}.py",
                extra_probe_paths=lambda config: [],
            )
        )
    return output


async def test_preflight_serializes_and_coalesces_overlapping_checks(
    tmp_path, monkeypatch
):
    config = settings(runners_dir=tmp_path, ai_human_dir=tmp_path, refton_dir=tmp_path)
    check = Preflight(config)
    monkeypatch.setattr(preflight_module, "STAGES", fake_stages(tmp_path))
    active = maximum = calls = 0

    async def probe(stage):
        nonlocal active, maximum, calls
        active += 1
        maximum = max(maximum, active)
        calls += 1
        await asyncio.sleep(0.01)
        active -= 1
        return True, ""

    monkeypatch.setattr(check, "_probe_stage", probe)
    snapshots = await asyncio.gather(*(check.probe_all() for _ in range(3)))
    assert calls == 2 and maximum == 1
    assert all(snapshot == {"ai_human": True, "viton": True} for snapshot in snapshots)
    assert check.ready


async def test_preflight_checks_missing_directory_before_spawning_conda(
    tmp_path, monkeypatch
):
    config = settings(
        runners_dir=tmp_path,
        ai_human_dir=tmp_path / "missing",
        refton_dir=tmp_path / "missing",
    )
    check = Preflight(config)
    monkeypatch.setattr(preflight_module, "STAGES", fake_stages(tmp_path))
    probe = AsyncMock(side_effect=AssertionError("spawned probe with missing cwd"))
    monkeypatch.setattr(check, "_probe_stage", probe)
    assert not any((await check.probe_all()).values())
    assert all("missing stage directory" in reason for reason in check.reasons.values())
    probe.assert_not_awaited()


async def test_probe_timeout_reaps_process_group(monkeypatch, tmp_path):
    config = settings(ai_human_dir=tmp_path)
    check = Preflight(config)
    proc = SimpleNamespace(pid=12345, returncode=None)
    proc.communicate = AsyncMock(side_effect=[asyncio.TimeoutError, (b"", b"")])
    proc.kill = lambda: None
    spawn = AsyncMock(return_value=proc)
    monkeypatch.setattr(preflight_module.asyncio, "create_subprocess_exec", spawn)
    killed = []
    if preflight_module.os.name == "posix":
        monkeypatch.setattr(
            preflight_module.os, "killpg", lambda pid, sig: killed.append(pid)
        )
    else:
        proc.kill = lambda: killed.append(proc.pid)
    ok, reason = await check._probe_stage(STAGES[0])
    assert not ok and reason == "probe timeout" and killed == [12345]
    assert proc.communicate.await_count == 2
    assert spawn.call_args.args[:6] == (
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "qwen21",
        "python",
    )


class FakeConsumer:
    def __init__(self, *args):
        self.started = False

    def start(self):
        self.started = True


async def test_supervisor_recovers_preflight_without_recreating_pool(
    monkeypatch, tmp_path
):
    config = settings(
        DATABASE_URL="postgresql://test:test@localhost/test", work_root=tmp_path
    )
    check = Preflight(config)
    calls = 0

    async def probe():
        nonlocal calls
        calls += 1
        check.availability = {"ai_human": calls >= 2, "viton": calls >= 2}
        check.reasons = {"ai_human": "" if calls >= 2 else "missing environment"}

    monkeypatch.setattr(check, "probe_all", probe)
    pool = SimpleNamespace(fetchval=AsyncMock(return_value=1))
    create = AsyncMock(return_value=pool)
    monkeypatch.setattr(main, "_create_pool", create)
    monkeypatch.setattr(main, "PostgresStore", lambda value: object())
    monkeypatch.setattr(
        main,
        "GpuStatusReporter",
        lambda *args: SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
    )
    monkeypatch.setattr(main, "Orchestrator", lambda *args: object())
    monkeypatch.setattr(main, "JobProcessor", lambda *args: object())
    monkeypatch.setattr(main, "RabbitConsumer", FakeConsumer)
    monkeypatch.setattr(main, "_PREFLIGHT_RETRY_DELAY", 0.01)
    monkeypatch.setattr(main, "_PREFLIGHT_RETRY_MAX_DELAY", 0.03)
    original_sleep = asyncio.sleep
    clock = [0.0]
    monkeypatch.setattr(main, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def fast_sleep(delay):
        clock[0] += delay
        await original_sleep(0)

    monkeypatch.setattr(main.asyncio, "sleep", fast_sleep)
    app = SimpleNamespace(state=state(config, check))
    task = asyncio.create_task(main._supervise(app, config))
    try:
        for _ in range(100):
            if app.state.consumer is not None:
                break
            await original_sleep(0.005)
        assert app.state.consumer is not None and app.state.consumer.started
        consumer = app.state.consumer
        await original_sleep(0.03)
        assert calls == 2 and app.state.consumer is consumer
        create.assert_awaited_once()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_gpu_busy_timeout_requeues_without_failure_or_cancellation():
    store = SimpleNamespace(
        add_event=AsyncMock(),
        is_cancel_requested=AsyncMock(return_value=False),
        mark_failed=AsyncMock(),
        mark_cancelled=AsyncMock(),
    )
    config = SimpleNamespace(
        gpu_free_mb_for=lambda kind: 24000,
        gpu_settle_seconds=0,
        gpu_wait_interval=0.001,
        gpu_wait_timeout=0.01,
    )
    processor = JobProcessor(
        config, store, object(), SimpleNamespace(free_mb=AsyncMock(return_value=1000))
    )
    assert (
        await processor._await_gpu_free("job-id", "vt-test", "viton")
        is AckDecision.NACK_REQUEUE
    )
    store.mark_failed.assert_not_awaited()
    store.mark_cancelled.assert_not_awaited()


async def test_gpu_wait_proceeds_when_memory_becomes_available():
    store = SimpleNamespace(
        add_event=AsyncMock(), is_cancel_requested=AsyncMock(return_value=False)
    )
    config = SimpleNamespace(
        gpu_free_mb_for=lambda kind: 24000,
        gpu_settle_seconds=0,
        gpu_wait_interval=0.001,
        gpu_wait_timeout=1,
    )
    processor = JobProcessor(
        config,
        store,
        object(),
        SimpleNamespace(free_mb=AsyncMock(side_effect=[1000, 32096])),
    )
    assert await processor._await_gpu_free("job-id", "vt-test", "viton") is None


async def test_start_consumer_is_idempotent(monkeypatch):
    config = settings()
    processor = SimpleNamespace(orchestrator=SimpleNamespace(progress_publisher=None))
    consumer = RabbitConsumer(
        config, processor, SimpleNamespace(state=SimpleNamespace())
    )
    waiting = asyncio.Event()
    monkeypatch.setattr(consumer, "_run", waiting.wait)
    first = consumer.start()
    assert consumer.start() is first
    await consumer.stop()
    assert first.done()


async def test_reporter_initializes_nvml_off_the_event_loop(monkeypatch):
    import threading
    from viton_api.gpu_status import GpuStatusReporter

    loop_thread = threading.get_ident()
    threads = []
    reporter = GpuStatusReporter(
        SimpleNamespace(execute=AsyncMock()),
        settings(),
        SimpleNamespace(state=SimpleNamespace()),
    )
    monkeypatch.setattr(
        reporter, "_init_nvml", lambda: threads.append(threading.get_ident())
    )
    monkeypatch.setattr(reporter, "_beat_loop", asyncio.Event().wait)
    await reporter.start()
    try:
        assert len(threads) == 1 and threads[0] != loop_thread
        assert reporter._task is not None
    finally:
        await reporter.stop()
