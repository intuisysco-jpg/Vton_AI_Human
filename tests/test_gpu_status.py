"""gpu_status row key: one stable row per worker service (GPU_STATUS_ID)."""

import socket

from viton_api.gpu_status import gpu_status_id


def test_gpu_status_id_uses_env(monkeypatch):
    monkeypatch.setenv("GPU_STATUS_ID", "viton-worker:0")
    assert gpu_status_id() == "viton-worker:0"


def test_gpu_status_id_falls_back_to_hostname(monkeypatch):
    monkeypatch.delenv("GPU_STATUS_ID", raising=False)
    assert gpu_status_id() == f"{socket.gethostname()}:0"
    monkeypatch.setenv("GPU_STATUS_ID", "  ")
    assert gpu_status_id() == f"{socket.gethostname()}:0"


# --- attribution heartbeat (processes / physical_id, migration 0023+) ------

import json as _json

from viton_api.gpu_status import GpuStatusReporter, physical_gpu_id


def test_physical_gpu_id_reads_env(monkeypatch):
    monkeypatch.setenv("GPU_PHYSICAL_ID", " gpu0 ")
    assert physical_gpu_id() == "gpu0"
    monkeypatch.delenv("GPU_PHYSICAL_ID", raising=False)
    assert physical_gpu_id() == ""


class _FakePool:
    """Records execute() calls; simulates the 0023 column check."""

    def __init__(self, has_columns=True):
        self.calls = []
        self._has_columns = has_columns

    async def execute(self, sql, *args):
        self.calls.append((sql, args))

    async def fetchrow(self, sql):
        return {"n": 2 if self._has_columns else 0}


class _FakeSettings:
    gpu_status_interval = 5


class _FakeApp:
    class state:
        pass


def _reporter(pool):
    r = GpuStatusReporter(pool, _FakeSettings(), _FakeApp())
    r._nvml_ok = True
    r._handle = object()
    return r


def _fake_nvidia_smi(monkeypatch, stdout, returncode=0):
    import subprocess as sp

    class _R:
        pass

    def run(*a, **k):
        r = _R()
        r.returncode = returncode
        r.stdout = stdout
        return r

    monkeypatch.setattr(sp, "run", run)


def test_beat_writes_attribution_columns(monkeypatch):
    import asyncio

    pool = _FakePool(has_columns=True)
    r = _reporter(pool)
    r._extended = True
    monkeypatch.setattr(r, "_read_nvml", lambda: {
        "total_mb": 32607, "used_mb": 28000, "free_mb": 4607,
        "utilization": 40})
    _fake_nvidia_smi(
        monkeypatch,
        "123, /nvme0n1-disk/conda/envs/aimodel/bin/python, 9000\n"
        "456, /nvme0n1-disk/conda/envs/some_training/bin/python, 18000\n",
    )
    monkeypatch.setenv("GPU_PHYSICAL_ID", "gpu0")
    monkeypatch.setattr("viton_api.gpu_status.gpu_status_id",
                        lambda: "viton-worker:0")
    asyncio.run(r._beat_once())
    sql, args = pool.calls[0]
    assert "processes" in sql and "physical_id" in sql
    procs = _json.loads(args[6])
    assert procs == [
        {"pid": 123, "name": "/nvme0n1-disk/conda/envs/aimodel/bin/python",
         "used_mb": 9000},
        {"pid": 456, "name": "/nvme0n1-disk/conda/envs/some_training/bin/python",
         "used_mb": 18000},
    ]
    assert args[7] == "gpu0"
    assert r.app.state.gpu["physical_id"] == "gpu0"
    assert len(r.app.state.gpu["processes"]) == 2


def test_beat_degrades_to_legacy_insert_without_columns(monkeypatch):
    import asyncio

    pool = _FakePool(has_columns=False)
    r = _reporter(pool)
    monkeypatch.setattr(r, "_read_nvml", lambda: {
        "total_mb": 32607, "used_mb": 511, "free_mb": 32096,
        "utilization": 0})
    asyncio.run(r._beat_once())
    sql, _ = pool.calls[0]
    assert "processes" not in sql  # legacy INSERT, beats keep flowing


def test_process_parsing_ignores_malformed_lines(monkeypatch):
    _fake_nvidia_smi(
        monkeypatch,
        "notapid, /x/python, 100\n999, /x/conda/envs/qwen21/bin/python, 250\n",
    )
    r = _reporter(_FakePool())
    procs = r._read_processes()
    assert procs == [{"pid": 999,
                      "name": "/x/conda/envs/qwen21/bin/python",
                      "used_mb": 250}]


def test_column_detection_sets_extended(monkeypatch):
    import asyncio

    for has, expected in ((True, True), (False, False)):
        r = _reporter(_FakePool(has_columns=has))
        asyncio.run(r._check_table())
        assert r._extended is expected
