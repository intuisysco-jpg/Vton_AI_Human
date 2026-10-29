"""Real broker integration against the backend's isolated local test stack."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import aio_pika
import pytest

from viton_api.config import Settings
from viton_api.consumer import AckDecision, RabbitConsumer
from viton_api import consumer as consumer_module

TEST_URL = "amqp://platform_test:local_test_only@127.0.0.1:56739/"


async def wait_until(predicate, timeout=8):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=timeout)


@pytest.mark.parametrize("interrupt_startup", [False, True])
async def test_real_broker_restores_exactly_one_subscription_per_queue(
    monkeypatch, interrupt_startup
):
    try:
        observer = await aio_pika.connect(TEST_URL, timeout=2)
    except (OSError, asyncio.TimeoutError):
        pytest.skip("Isolated RabbitMQ test stack is not running on port 56739")

    original_connect = aio_pika.connect_robust

    async def connect(*args, **kwargs):
        return await original_connect(*args, **kwargs, reconnect_interval=0.1)

    monkeypatch.setattr(aio_pika, "connect_robust", connect)
    # Exercise the production watchdog too if a disconnect races the optional
    # progress channel's initial setup, without waiting 30 seconds in the test.
    monkeypatch.setattr(consumer_module, "_RESTORE_TIMEOUT", 0.5)
    monkeypatch.setattr(consumer_module, "_WATCH_INTERVAL", 0.05)
    progress_entered, release_progress = asyncio.Event(), asyncio.Event()
    if interrupt_startup:
        original_publisher = consumer_module.ProgressPublisher

        class DelayedProgressPublisher(original_publisher):
            async def start(self, connection):
                progress_entered.set()
                await release_progress.wait()
                await super().start(connection)

        monkeypatch.setattr(
            consumer_module, "ProgressPublisher", DelayedProgressPublisher
        )
    prefix = f"viton-recovery-test-{uuid.uuid4().hex}"
    names = [f"{prefix}-human_queue", f"{prefix}-viton_queue"]
    settings = Settings(_env_file=None, RABBITMQ_URL=TEST_URL, queue_names=names)
    processor = SimpleNamespace(
        orchestrator=SimpleNamespace(progress_publisher=None),
        process=AsyncMock(return_value=AckDecision.ACK),
    )
    consumer = RabbitConsumer(
        settings, processor, SimpleNamespace(state=SimpleNamespace())
    )
    channel = await observer.channel()
    consumer.start()
    try:
        await wait_until(lambda: consumer.connected)
        if interrupt_startup:
            await asyncio.wait_for(progress_entered.wait(), timeout=2)
        else:
            await wait_until(
                lambda: processor.orchestrator.progress_publisher is not None
            )
        for name in names:
            queue = await channel.declare_queue(name, passive=True)
            assert queue.declaration_result.consumer_count == 1
        await channel.default_exchange.publish(
            aio_pika.Message(b'{"job_id":"synthetic-test","job_type":"viton"}'),
            routing_key=names[1],
        )
        await wait_until(lambda: processor.process.await_count == 1)
        await consumer._connection.transport.connection.close()
        assert not consumer.connected
        release_progress.set()
        await wait_until(lambda: consumer.connected, timeout=15)
        for name in names:
            queue = await channel.declare_queue(name, passive=True)
            assert queue.declaration_result.consumer_count == 1
        await channel.default_exchange.publish(
            aio_pika.Message(b'{"job_id":"after-reconnect","job_type":"viton"}'),
            routing_key=names[0],
        )
        await wait_until(lambda: processor.process.await_count == 2)
    finally:
        release_progress.set()
        await consumer.stop()
        for name in names:
            await channel.queue_delete(name, if_unused=False, if_empty=False)
            await channel.queue_delete(
                name.removesuffix("_queue") + "_dlq", if_unused=False, if_empty=False
            )
        await observer.close()
