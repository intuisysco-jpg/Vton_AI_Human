import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from viton_api import consumer as module
from viton_api.config import Settings
from viton_api.consumer import RabbitConsumer


class Callbacks:
    def __init__(self, owner):
        self.owner, self.callbacks = owner, set()

    def add(self, callback):
        self.callbacks.add(callback)

    def emit(self):
        for callback in self.callbacks:
            callback(self.owner)


class Channel:
    def __init__(self, connection, failure=None):
        self.connection, self.failure = connection, failure
        self.is_closed = False
        self.close_callbacks = Callbacks(self)
        self.reopen_callbacks = Callbacks(self)
        self.subscriptions = []

    async def set_qos(self, **options):
        assert options == {"prefetch_count": 1, "global_": False}

    async def declare_queue(self, name, **options):
        async def consume(callback, **options):
            if name == self.failure:
                raise RuntimeError("simulated subscription rejection")
            self.subscriptions.append((name, callback))
            return name

        return SimpleNamespace(consume=consume)

    async def close(self):
        self.is_closed = True
        self.subscriptions.clear()
        self.close_callbacks.emit()


class Connection:
    def __init__(self, failure=None):
        self.is_closed = False
        self.connected = asyncio.Event()
        self.connected.set()
        self.close_callbacks = Callbacks(self)
        self.reconnect_callbacks = Callbacks(self)
        self.consumer_channel = Channel(self, failure)

    async def channel(self):
        return self.consumer_channel

    async def close(self):
        self.is_closed = True
        self.connected.clear()
        self.close_callbacks.emit()


def make_consumer(monkeypatch):
    settings = Settings(_env_file=None)
    app = SimpleNamespace(state=SimpleNamespace())
    processor = SimpleNamespace(
        orchestrator=SimpleNamespace(progress_publisher=None), process=AsyncMock()
    )
    monkeypatch.setattr(
        module,
        "ProgressPublisher",
        lambda: SimpleNamespace(start=AsyncMock(), close=AsyncMock()),
    )
    return RabbitConsumer(settings, processor, app)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


async def test_partial_subscription_failure_closes_session_and_retries(monkeypatch):
    consumer = make_consumer(monkeypatch)
    failed = Connection("viton_queue")
    recovered = Connection()
    connect = AsyncMock(side_effect=[failed, recovered])
    monkeypatch.setattr(module.aio_pika, "connect_robust", connect)
    retry_entered, allow_retry = asyncio.Event(), asyncio.Event()
    retry_delays = []

    async def retry(delay):
        if consumer.connected:
            await consumer._stop_event.wait()
            return
        retry_delays.append(delay)
        retry_entered.set()
        await allow_retry.wait()

    monkeypatch.setattr(consumer, "_wait_retry", retry)
    consumer.start()
    try:
        await asyncio.wait_for(retry_entered.wait(), timeout=2)
        assert failed.is_closed and failed.consumer_channel.is_closed
        assert not failed.consumer_channel.subscriptions
        assert not consumer.connected and not consumer.app.state.consumer_connected
        assert 0.8 <= retry_delays[0] <= 1.2
        allow_retry.set()
        await until(lambda: consumer.connected)
        assert connect.await_count == 2
        assert len(recovered.consumer_channel.subscriptions) == 2
        assert consumer.app.state.consumer_connected
    finally:
        allow_retry.set()
        await consumer.stop()


async def test_disconnect_and_restore_track_real_subscription_state(monkeypatch):
    consumer = make_consumer(monkeypatch)
    connection = Connection()
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(module.aio_pika, "connect_robust", connect)
    consumer.start()
    try:
        await until(lambda: consumer.connected)
        original = list(connection.consumer_channel.subscriptions)
        connection.connected.clear()
        connection.close_callbacks.emit()
        assert not consumer.connected and not consumer.app.state.rabbit_connected
        assert not consumer._session_ready.is_set()
        connection.connected.set()
        assert not consumer.connected  # Transport alone is insufficient.
        connection.consumer_channel.reopen_callbacks.emit()
        assert consumer.connected and consumer.app.state.consumer_connected
        assert consumer._session_ready.is_set()
        assert connection.consumer_channel.subscriptions == original
        connect.assert_awaited_once()
    finally:
        await consumer.stop()


async def test_stalled_native_restoration_is_bounded_without_cancelling_job(monkeypatch):
    consumer = make_consumer(monkeypatch)
    consumer._disconnected_at = 0
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 31))
    consumer.processor.process = AsyncMock()
    with pytest.raises(ConnectionError, match="did not recover"):
        await consumer._watch_connection()
    consumer.processor.process.assert_not_awaited()


async def test_failed_setup_delivery_never_runs_processor(monkeypatch):
    consumer = make_consumer(monkeypatch)
    connection = Connection()
    ready = asyncio.Event()
    consumer._connection = connection
    consumer._channel = connection.consumer_channel
    message = SimpleNamespace(
        body=b'{"job_id":"vt-test","job_type":"viton"}',
        ack=AsyncMock(),
        nack=AsyncMock(),
    )
    task = asyncio.create_task(
        consumer._on_message(message, connection=connection, ready=ready)
    )
    await asyncio.sleep(0)
    assert not task.done()
    await consumer._close()
    ready.set()
    await asyncio.wait_for(task, timeout=1)
    consumer.processor.process.assert_not_awaited()
    message.nack.assert_awaited_once_with(requeue=True)


async def test_both_queue_handlers_serialize_and_shutdown_preserves_waiting_delivery(
    monkeypatch,
):
    consumer = make_consumer(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    references = []

    async def process(reference, job_type):
        references.append(reference)
        entered.set()
        await release.wait()
        return module.AckDecision.ACK

    consumer.processor.process = process
    messages = [
        SimpleNamespace(
            body=(f'{{"job_id":"job-{i}","job_type":"viton"}}').encode(),
            ack=AsyncMock(),
            nack=AsyncMock(),
        )
        for i in range(2)
    ]
    tasks = [asyncio.create_task(consumer._on_message(message)) for message in messages]
    await entered.wait()
    await asyncio.sleep(0)
    assert references == ["job-0"] and len(consumer._inflight) == 2
    stopping = asyncio.create_task(consumer.stop())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks, stopping), timeout=2)
    assert references == ["job-0"]
    messages[1].nack.assert_awaited_once_with(requeue=True)
    assert not consumer._inflight


async def test_optional_progress_failure_does_not_disable_consumption(monkeypatch):
    consumer = make_consumer(monkeypatch)
    connection = Connection()
    monkeypatch.setattr(
        module.aio_pika, "connect_robust", AsyncMock(return_value=connection)
    )
    progress = SimpleNamespace(
        start=AsyncMock(side_effect=RuntimeError("optional")), close=AsyncMock()
    )
    monkeypatch.setattr(module, "ProgressPublisher", lambda: progress)
    consumer.start()
    try:
        await until(lambda: progress.close.await_count == 1)
        assert consumer.connected
        assert consumer.processor.orchestrator.progress_publisher is None
    finally:
        await consumer.stop()
