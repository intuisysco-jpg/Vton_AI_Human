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
