"""Non-critical progress event publisher.

Emits real-time progress snapshots onto a fanout exchange (``job-progress``) so
the central API's SSE endpoints can stream live job/batch status. These events
are EPHEMERAL and MUST NEVER fail a job: ``emit`` swallows every error and just
logs. Postgres (the store transitions) remains the source of truth; this is
best-effort enrichment.

The exchange is ``durable=False``, FANOUT. Every declarer (each worker + the
central listener) must pass identical arguments or RabbitMQ raises
``PRECONDITION_FAILED`` and closes the shared channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

logger = logging.getLogger(__name__)


class ProgressPublisher:
    """Publishes progress snapshots to the ``job-progress`` fanout exchange.

    Opens its OWN channel on the shared robust connection (auto-reconnect is
    free: aio-pika robust channels replay declarations on reconnect). Created
    lazily after the connection exists and assigned onto the orchestrator, which
    calls :meth:`emit` from within the job pipeline. A broker hiccup during emit
    is swallowed — progress is never worth failing a job over.
    """

    EXCHANGE = "job-progress"

    def __init__(self) -> None:
        self._channel: AbstractChannel | None = None
        self._exchange = None

    async def start(self, connection: AbstractRobustConnection) -> None:
        """Declare the exchange on a 2nd channel of the existing connection."""
        self._channel = await connection.channel()
        self._exchange = await self._channel.declare_exchange(
            self.EXCHANGE,
            aio_pika.ExchangeType.FANOUT,
            durable=False,
        )

    async def close(self) -> None:
        channel = self._channel
        self._channel = None
        self._exchange = None
        if channel is not None and not channel.is_closed:
            try:
                await channel.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    async def emit(
        self,
        job_id: str | None,
        batch_id: str | None,
        payload: dict,
    ) -> None:
        """Publish one progress event (fire-and-forget; never raises)."""
        if self._exchange is None:
            return
        body = {
            "job_id": job_id,
            "batch_id": batch_id,
            **payload,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await self._exchange.publish(
                aio_pika.Message(
                    json.dumps(body).encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key="",  # fanout ignores routing_key
            )
        except Exception:  # noqa: BLE001 - non-critical: swallow and log
            logger.warning("progress publish failed: %s", body.get("status"), exc_info=True)