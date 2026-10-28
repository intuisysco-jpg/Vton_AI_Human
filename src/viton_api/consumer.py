"""RabbitMQ consumer + per-job processor.

RabbitConsumer  — broker lifecycle (one shared channel, prefetch=1 global_=True
  => at most one unacked message => one GPU job at a time across BOTH queues;
  robust reconnect; DLQ-first declare; poison = ACK).
JobProcessor    — the per-job contract (DB + storage + orchestrator); no broker
  I/O.

AckDecision semantics (worker blueprint Appendix D):
  ACK          poison (bad JSON / missing fields / unknown ref / terminal /
               bad params / missing per-ordinal inputs / pre-pipeline cancel),
               pipeline success, JobCancelled
  NACK_DLQ     unsupported job_type that DOES exist in DB; OR failure with
               retry_count >= max_retries
  NACK_REQUEUE failure with retry_count < max_retries; runner exit 3 (CUDA
               OOM, no DB write); GPU busy past timeout
"""

from __future__ import annotations
from .fencing import fenced, attempt_path

import asyncio
import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import aio_pika

from .config import Settings, dlq_name_for
from .gpu_lock import GpuLease
from .models import (
    EXIT_OOM,
    EXIT_POISON,
    SUPPORTED_JOB_TYPES,
    TERMINAL_STATUSES,
    image_pct,
    parse_params,
    validate_ai_human_params,
    validate_viton_params,
)
from .orchestrator import JobCancelled, Orchestrator, StageFailed
from .progress_publisher import ProgressPublisher
from .store import PostgresStore

logger = logging.getLogger("viton_api.consumer")


class MissingInputError(Exception):
    """A job's input file is gone from the storage tree (retention purged it).

    Permanent failure: the job is set ``failed`` with an 'inputs expired'
    message and the message goes straight to the DLQ — never a retry loop.
    """


class AckDecision(str, Enum):
    ACK = "ack"
    NACK_REQUEUE = "nack_requeue"
    NACK_DLQ = "nack_dlq"


@dataclass(slots=True)
class Unit:
    """One image unit of a job: its ordinal, the storage-relative input paths
    (``*_rel``) and the staged absolute paths filled in by :meth:`_stage_units`."""

    ordinal: int
    original_filename: str | None
    reference_rel: str | None = None  # ai_human mode B (text): None
    person_rel: str | None = None     # viton
    garment_rel: str | None = None    # viton
    reference: Path | None = None
    person: Path | None = None
    garment: Path | None = None

    def rels(self) -> list[tuple[str, str]]:
        """(role, storage-relative path) pairs to stage for this unit."""
        out = []
        if self.reference_rel:
            out.append(("reference", self.reference_rel))
        if self.person_rel:
            out.append(("person", self.person_rel))
        if self.garment_rel:
            out.append(("garment", self.garment_rel))
        return out

    def to_job_unit(self) -> dict:
        """The per-unit entry of the runner's job.json contract."""
        return {
            "ordinal": self.ordinal,
            "original_filename": self.original_filename,
            "reference": str(self.reference) if self.reference else None,
            "person": str(self.person) if self.person else None,
            "garment": str(self.garment) if self.garment else None,
        }


class RabbitConsumer:
    def __init__(self, settings: Settings, processor: "JobProcessor", app):
        self.settings = settings
        self.processor = processor
        self.app = app
        self._connection = None
        self._channel = None
        self._consumer_tags: list[str] = []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._stop_requested = False
        self._inflight: asyncio.Future | None = None
        self._connected = False
        self._progress: ProgressPublisher | None = None
        # Serializes job processing across BOTH consumed queues (see
        # _setup_channel: the channel-level global prefetch cap is gone).
        self._job_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run(), name="rabbit-consumer")
        return self._task

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                if self._progress is not None:
                    await self._progress.close()
                    self._progress = None
                    self.processor.orchestrator.progress_publisher = None
                self._connection = await aio_pika.connect_robust(self.settings.rabbitmq_url)
                self._channel = await self._connection.channel()
                ok = await self._setup_channel()
                self._connected = ok
                if ok:
                    logger.info("consumer connected (prefetch=1, global_=False)")
                    backoff = 1.0
                    self._progress = ProgressPublisher()
                    await self._progress.start(self._connection)
                    self.processor.orchestrator.progress_publisher = self._progress
                await self._stop_event.wait()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("consumer connect error (retry in %.1fs): %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        await self._close()

    async def _setup_channel(self) -> bool:
        # Per-consumer prefetch (global_=False): RabbitMQ 4 deprecates the
        # channel-wide cap (global_=True — deprecation errors today, removal
        # in a future major). The channel no longer enforces "one unacked
        # TOTAL across both queues"; the _on_message job lock now enforces
        # the single-GPU-job invariant instead.
        await self._channel.set_qos(prefetch_count=1, global_=False)
        required_ok = True
        # BOTH queues are required: dropping one would silently split the
        # single-unacked-message invariant across pipelines.
        for queue in self.settings.queue_names:
            dlq = dlq_name_for(queue)
            try:
                # declare the DLQ first — a dead-letter to a missing DLQ is dropped.
                await self._channel.declare_queue(dlq, durable=True)
                main = await self._channel.declare_queue(
                    queue,
                    durable=True,
                    arguments={
                        "x-dead-letter-exchange": "",
                        "x-dead-letter-routing-key": dlq,
                    },
                )
                tag = await main.consume(self._on_message, no_ack=False)
                self._consumer_tags.append(tag)
                logger.info("consuming %s (dlq=%s)", queue, dlq)
            except Exception as e:  # noqa: BLE001
                logger.error("FAILED to consume required queue %s: %s", queue, e)
                required_ok = False
        return required_ok

    async def _on_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        if self._stop_requested:
            await message.nack(requeue=True)
            return
        self._inflight = asyncio.get_running_loop().create_future()
        try:
            # One job at a time across BOTH queues: with per-consumer prefetch
            # (global_=False) the channel no longer enforces the single-unacked
            # invariant, so this lock does. A second delivered message just
            # waits here in worker memory (broker sees <=2 unacked).
            async with self._job_lock:
                decision = await self._handle_message(message)
        except Exception as e:  # noqa: BLE001 — never let the consumer task die
            logger.exception("unhandled error processing message: %s", e)
            decision = AckDecision.NACK_REQUEUE
        finally:
            if not self._inflight.done():
                self._inflight.set_result(None)

        try:
            if decision == AckDecision.ACK:
                await message.ack()
            elif decision == AckDecision.NACK_REQUEUE:
                await message.nack(requeue=True)
            elif decision == AckDecision.NACK_DLQ:
                await message.nack(requeue=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("ack/nack failed: %s", e)

    async def _handle_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> AckDecision:
        try:
            payload = json.loads(message.body.decode("utf-8"))
        except Exception:
            logger.warning("poison: malformed JSON -> ACK")
            return AckDecision.ACK
        if not isinstance(payload, dict):
            return AckDecision.ACK

        job_type = (payload.get("job_type") or "").strip()
        if not job_type:
            logger.warning("poison: missing job_type -> ACK")
            return AckDecision.ACK

        reference = (payload.get("job_id") or "").strip()
        if not reference:
            logger.warning("poison: no job_id -> ACK")
            return AckDecision.ACK

        self.app.state.current_job = reference
        try:
            return await self.processor.process(reference, job_type)
        finally:
            self.app.state.current_job = None

    async def stop(self) -> None:
        self._stop_requested = True
        self._stop_event.set()
        # Bounded on purpose: a GPU job mid-flight is NOT drained (too long);
        # its message stays unacked and the broker requeues it once the
        # connection drops. Previously this wait was unbounded (timeout=None),
        # so a long job at SIGTERM meant docker's stop timeout SIGKILLed us.
        if self._inflight is not None and not self._inflight.done():
            try:
                await asyncio.wait_for(self._inflight, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("job still in flight at shutdown; abandoning (message requeues)")
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # The cancelled _run task never reaches its trailing _close() (the
        # CancelledError propagates past it), so tear the AMQP side down here.
        # Bounded so a broker already gone at shutdown can't hang us past the
        # compose stop_grace_period (budget: 5+5+5 = 15s of a 30s window).
        await self._close()

    async def _close(self) -> None:
        # Every close is bounded + swallowed: these must never hang or raise
        # past shutdown (a dead broker transport can stall a close forever).
        if self._progress is not None:
            try:
                await asyncio.wait_for(self._progress.close(), timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            self._progress = None
            self.processor.orchestrator.progress_publisher = None
        try:
            if self._channel is not None and self._channel.is_closed is False:
                await asyncio.wait_for(self._channel.close(), timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._connection is not None and self._connection.is_closed is False:
                await asyncio.wait_for(self._connection.close(), timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        self._connected = False


class JobProcessor:
    def __init__(self, settings: Settings, store: PostgresStore, orchestrator: Orchestrator,
                 gpu_reporter=None):
        self.settings = settings
        self.store = store
        self.orchestrator = orchestrator
        self.gpu_reporter = gpu_reporter

    # ------------------------------------------------------------------ #
    @fenced
    async def process(self, reference: str, job_type: str) -> AckDecision:
        # 1. supported job_type?
        if job_type not in SUPPORTED_JOB_TYPES:
            return await self._reject_unimplemented(reference, job_type)

        # 2. job exists?
        job = await self.store.get_job_by_reference(reference)
        if job is None:
            logger.warning("poison: reference %s not in DB -> ACK", reference)
            return AckDecision.ACK

        # 3. already terminal? (idempotent redelivery)
        if job["status"] in TERMINAL_STATUSES:
            logger.info("reference %s already %s -> ACK (idempotent)", reference, job["status"])
            return AckDecision.ACK

        # 4. params (DB is the single source of truth) — bad values are POISON,
        #    but only after they've been marked failed so the client can see why.
        params = parse_params(job.get("params"))
        if job_type == "ai_human":
            err = validate_ai_human_params(params)
        else:
            err = validate_viton_params(params)
        if err:
            logger.warning("poison: reference %s bad params (%s) -> ACK", reference, err)
            await self.store.set_failed(job["id"], f"invalid params: {err}")
            await self.store.add_event(job["id"], "dispatch", f"invalid params: {err}")
            return AckDecision.ACK

        # 5. input discovery by ordinal -> image units.
        try:
            units = await self._build_units(job, job_type)
        except _PoisonError as e:
            logger.warning("poison: reference %s %s -> ACK", reference, e)
            await self.store.set_failed(job["id"], str(e))
            await self.store.add_event(job["id"], "dispatch", str(e))
            return AckDecision.ACK

        return await self._process_units(job, job_type, params, units)

    # ------------------------------------------------------------------ #
    async def _build_units(self, job: dict, job_type: str) -> list[Unit]:
        """Resolve non-output job_files rows into per-ordinal image units.

        ai_human: text mode has ZERO input rows (legal) — units are synthesized
        from jobs.image_count. Reference rows (role='reference') attach per
        ordinal. viton: every ordinal MUST have both 'person' and 'garment'.
        """
        job_id = job["id"]
        inputs = await self.store.get_input_files(job_id)
        by_ordinal: dict[int, dict[str, dict]] = {}
        for row in inputs:
            by_ordinal.setdefault(int(row.get("ordinal") or 0), {})[row["role"]] = row

        if job_type == "ai_human":
            image_count = int(job.get("image_count") or 0)
            if not by_ordinal and image_count <= 0:
                raise _PoisonError("text job with image_count=0 and no inputs")
            # one unit per text variation, or one per reference row when present
            ordinals = sorted(by_ordinal) if by_ordinal else list(range(image_count))
            if len(ordinals) > max(image_count, len(by_ordinal)):
                raise _PoisonError("more reference rows than image_count")
            units = []
            for ordinal in ordinals:
                roles = by_ordinal.get(ordinal, {})
                ref_row = roles.get("reference")
                units.append(Unit(
                    ordinal=ordinal,
                    original_filename=ref_row["original_filename"] if ref_row else None,
                    reference_rel=ref_row["file_path"] if ref_row else None,
                ))
            return units

        # viton
        if not by_ordinal:
            raise _PoisonError("no non-output job_files rows")
        units = []
        for ordinal in sorted(by_ordinal):
            roles = by_ordinal[ordinal]
            person = roles.get("person")
            garment = roles.get("garment")
            if person is None or garment is None:
                raise _PoisonError(f"ordinal {ordinal} missing person/garment input")
            primary = person["original_filename"] or garment["original_filename"]
            units.append(Unit(
                ordinal=ordinal,
                original_filename=primary,
                person_rel=person["file_path"],
                garment_rel=garment["file_path"],
            ))
        return units

    # ------------------------------------------------------------------ #
    async def _process_units(self, job: dict, job_type: str, params: dict,
                             units: list[Unit]) -> AckDecision:
        job_id = job["id"]
        reference = job["reference"]
        batch_id = job.get("batch_id")
        image_count = int(job.get("image_count") or len(units)) or len(units)

        # 1. idempotent redelivery: skip ordinals that already have a completed
        #    output row — only the *remaining* images are (re-)processed.
        completed = await self.store.get_completed_ordinals(job_id)
        todo = [u for u in units if u.ordinal not in completed]
        done = len(completed)
        if not todo:
            logger.info("reference %s: all %d images already completed -> ACK",
                        reference, image_count)
            if job["status"] != "completed":
                await self.store.mark_completed(job_id, 100)
            return AckDecision.ACK

        # 2. pre-pipeline cancel
        if await self.store.is_cancel_requested(job_id):
            await self.store.mark_cancelled(job_id)
            await self.store.add_event(job_id, "cancel", "cancelled before pipeline")
            await self.orchestrator._emit(reference, batch_id, status="cancelled",
                                          error_message="cancelled before pipeline")
            logger.info("reference %s cancelled before pipeline -> ACK", reference)
            return AckDecision.ACK

        # 3. stage inputs into a local workdir (raw copies; the runner does any
        #    normalization — the thin service venv has no PIL).
        workdir = Path(self.settings.work_root) / f"job_{uuid.uuid4().hex}"
        workdir.mkdir(parents=True, exist_ok=True)
        lease = GpuLease(self.settings)
        try:
            self._stage_units(todo, workdir)

            # 4. running
            await self.store.mark_running(job_id)
            await self.store.add_event(
                job_id, "pipeline",
                f"running ({len(todo)}/{image_count} images)")
            await self.orchestrator._emit(
                reference, batch_id, status="running",
                pct=image_pct(done, image_count),
                progress=image_pct(done, image_count),
                ordinal=0, images_total=image_count, images_done=done,
                message=f"running ({len(todo)}/{image_count} images)")

            # 5. shared GPU: exclusive flock first, then settle, then the NVML
            #    free-memory gate (run_tryon.sh §2.5 ordering). Lease-busy or a
            #    gate timeout is a CLEAN requeue — busyness is not a job error.
            try:
                await lease.acquire()
            except TimeoutError:
                logger.info("reference %s: GPU lease busy -> NACK_REQUEUE", reference)
                return AckDecision.NACK_REQUEUE
            exec_t0: float | None = None  # set once the gate passes; None = never started
            try:
                gate = await self._await_gpu_free(job_id, reference, job_type)
                if gate is not None:
                    return gate
                # Execution timing: clock starts once the GPU is ours AND the
                # settle gate has passed — the moment execution actually begins
                # (stops when the inference run returns, any outcome).
                exec_t0 = time.monotonic()
                await self._record_exec_start(job_id)
                unit_map = {u.ordinal: u for u in todo}

                async def on_event(event: dict) -> None:
                    nonlocal done
                    kind = event.get("event")
                    ordinal = int(event.get("ordinal", 0) or 0)
                    if kind == "image_done":
                        # Increment FIRST so _finalize_image persists and emits
                        # the same post-increment count (DB and SSE must agree,
                        # and the store's ladder needs done INCLUDING this image).
                        done += 1
                        unit = unit_map.get(ordinal)
                        await self._finalize_image(
                            reference, job_id, ordinal, event.get("path"),
                            unit.original_filename if unit else None,
                            image_count, done, batch_id)
                    elif kind == "image_generated":
                        pct = image_pct(done, image_count)
                        await self.orchestrator._emit(
                            reference, batch_id, status="running",
                            pct=pct, progress=pct,
                            ordinal=ordinal, images_total=image_count,
                            images_done=done,
                            message=f"image {ordinal + 1}/{image_count} generated")
                    elif kind == "image_failed":
                        await self.store.mark_image_failed(job_id, ordinal)
                        err = str(event.get("error") or "runner reported failure")
                        await self.store.add_event(
                            job_id, "pipeline", f"image {ordinal} failed: {err}")
                        pct = image_pct(done, image_count)
                        await self.orchestrator._emit(
                            reference, batch_id, status="running",
                            pct=pct, progress=pct,
                            ordinal=ordinal, images_total=image_count,
                            images_done=done,
                            message=f"image {ordinal + 1}/{image_count} failed")
                        logger.warning("reference %s: image %d failed (%s)",
                                       reference, ordinal, err)
                    # unknown / fatal_* events: already in job.log, nothing to do

                await self.orchestrator.run(
                    job_id, reference, params,
                    [u.to_job_unit() for u in todo],
                    workdir, job_type, batch_id=batch_id, on_event=on_event,
                    progress_floor=image_pct(done, image_count))
            finally:
                if exec_t0 is not None:
                    await self._record_exec_ms(
                        job_id, int((time.monotonic() - exec_t0) * 1000))
                lease.release()

            # 6. finalize: at least one image must have been recorded.
            if done == 0:
                raise StageFailed("pipeline", "exit 0 but no image_done events")

            # Completed reads 100 even when some images failed: a completed
            # job's bar must read full — failures show in the images[] manifest.
            await self.store.mark_completed(job_id, 100)
            await self.store.add_event(
                job_id, "pipeline", f"completed ({done}/{image_count} images)")
            await self.orchestrator._emit(
                reference, batch_id, status="completed",
                pct=100, progress=100, current_stage=None,
                ordinal=0, images_total=image_count, images_done=done,
                message=f"completed ({done}/{image_count} images)")
            return AckDecision.ACK
        except MissingInputError as e:
            # Retention purged the inputs while this job was waiting -> permanent
            # failure straight to the DLQ, never a retry loop.
            await self.store.set_failed(job_id, f"inputs expired: {e}")
            await self.store.add_event(job_id, "pipeline", f"inputs expired: {e}")
            await self.orchestrator._emit(reference, batch_id, status="failed",
                                          error_message=f"inputs expired: {e}")
            logger.error("reference %s inputs missing (expired) -> DLQ", reference)
            return AckDecision.NACK_DLQ
        except JobCancelled:
            await self.store.mark_cancelled(job_id)
            await self.store.add_event(job_id, "cancel", "cancelled during pipeline")
            await self.orchestrator._emit(reference, batch_id, status="cancelled",
                                          error_message="cancelled during pipeline")
            logger.info("reference %s cancelled during pipeline -> ACK", reference)
            return AckDecision.ACK
        except StageFailed as e:
            # Runner contract: exit 2 = POISON (bad params/inputs/guard) — failed
            # without bumping the retry budget, ACK to the DLQ. Exit 3 = CUDA
            # OOM — clean requeue, no DB write. Anything else: retry budget.
            if e.exit_code == EXIT_POISON:
                await self.store.set_failed(job_id, repr(e))
                await self.store.add_event(job_id, "pipeline", f"poison: {e}")
                await self.orchestrator._emit(reference, batch_id, status="failed",
                                              error_message=repr(e))
                logger.warning("reference %s poison (%s) -> DLQ", reference, e)
                return AckDecision.NACK_DLQ
            if e.exit_code == EXIT_OOM:
                logger.info("reference %s CUDA OOM -> clean NACK_REQUEUE (no DB write)",
                            reference)
                return AckDecision.NACK_REQUEUE
            return await self._fail_with_budget(job, reference, batch_id, e, image_count, done)
        except Exception as e:  # noqa: BLE001
            return await self._fail_with_budget(job, reference, batch_id, e, image_count, done)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _fail_with_budget(self, job: dict, reference: str, batch_id: str | None,
                                err: Exception, image_count: int, done: int) -> AckDecision:
        retry_count = await self.store.mark_failed(job["id"], repr(err))
        await self.orchestrator._emit(reference, batch_id, status="failed",
                                      error_message=repr(err), retry_count=retry_count,
                                      ordinal=0, images_total=image_count, images_done=done)
        logger.warning("reference %s failed (%s); retry_count=%d",
                       reference, err, retry_count)
        if retry_count < self.settings.max_retries:
            return AckDecision.NACK_REQUEUE
        logger.error("reference %s exhausted retries -> DLQ", reference)
        return AckDecision.NACK_DLQ

    # ------------------------------------------------------------------ #
    def _stage_units(self, todo: list[Unit], workdir: Path) -> None:
        """Copy each unit's referenced inputs from storage into
        <workdir>/inputs/{ordinal:04d}_{role}.<ext> (raw copies; the runner
        does any normalization — the thin service venv has no PIL)."""
        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        storage = Path(self.settings.storage_root)
        for unit in todo:
            for role, rel in unit.rels():
                src = storage / rel
                if not src.exists():
                    raise MissingInputError(rel)
                dest = inputs_dir / f"{unit.ordinal:04d}_{role}{src.suffix or '.png'}"
                shutil.copy2(str(src), str(dest))
                setattr(unit, role, dest)

    async def _await_gpu_free(self, job_id, reference: str, job_type: str) -> AckDecision | None:
        """After the flock is held: let the previous tenant's VRAM drain
        (settle), then poll NVML free_mb until it reaches this pipeline's
        threshold. Timeout -> clean NACK_REQUEUE (the lease is released by the
        caller's finally). Cancel is honored while waiting. Fails open when NVML
        is unavailable."""
        threshold = self.settings.gpu_free_mb_for(job_type)
        await self.store.add_event(job_id, "dispatch", "waiting for free GPU")
        logger.info("reference %s: GPU settle %.0fs then gate free_mb >= %d",
                    reference, self.settings.gpu_settle_seconds, threshold)
        await asyncio.sleep(self.settings.gpu_settle_seconds)

        reporter = self.gpu_reporter
        if reporter is None:
            return None  # no telemetry -> fail-open (proceed)
        started = time.monotonic()
        while True:
            if await self.store.is_cancel_requested(job_id):
                # raise so the caller's JobCancelled handler finalizes the cancel
                raise JobCancelled()
            free = await reporter.free_mb()
            if free is None:
                logger.warning("NVML unavailable; skipping GPU-free gate for %s", reference)
                return None
            if free >= threshold:
                logger.info("reference %s: GPU free (free_mb=%d) -> proceeding", reference, free)
                return None
            if self.settings.gpu_wait_timeout and \
                    (time.monotonic() - started) >= self.settings.gpu_wait_timeout:
                logger.info("reference %s: GPU busy after %.0fs -> NACK_REQUEUE (no fail)",
                            reference, self.settings.gpu_wait_timeout)
                return AckDecision.NACK_REQUEUE
            await asyncio.sleep(self.settings.gpu_wait_interval)

    async def _record_exec_start(self, job_id) -> None:
        """Best-effort: stamp ``exec_started_at`` right before the pipeline call
        (GPU lease held, settle gate passed). Swallowed — timing telemetry must
        never abort the job (e.g. before the central migration has landed).
        """
        try:
            await self.store.mark_exec_started(job_id)
        except Exception:
            logger.warning("could not stamp exec_started_at for job %s; continuing",
                           job_id, exc_info=True)

    async def _record_exec_ms(self, job_id, ms: int) -> None:
        """Best-effort: persist the pipeline-call duration. Swallowed — callers
        invoke this from a ``finally`` so it must never mask an in-flight
        exception either.
        """
        try:
            await self.store.set_execution_ms(job_id, ms)
        except Exception:
            logger.warning("could not record execution_ms for job %s; continuing",
                           job_id, exc_info=True)

    async def _finalize_image(self, reference: str, job_id, ordinal: int,
                              produced_path: str | None, original_filename: str | None,
                              image_count: int, done: int,
                              batch_id: str | None = None) -> None:
        """Copy the runner's produced file into storage and record the output
        row + per-image completion (idempotent upsert)."""
        if not produced_path or not Path(produced_path).exists():
            await self.store.mark_image_failed(job_id, ordinal)
            await self.store.add_event(job_id, "pipeline",
                                       f"image {ordinal}: runner reported done but file missing")
            logger.warning("reference %s: image %d output missing", reference, ordinal)
            return
        storage = Path(self.settings.storage_root)
        out_root = storage / reference / "output" / attempt_path(reference)
        await asyncio.to_thread(out_root.mkdir, parents=True, exist_ok=True)
        dest = out_root / f"{ordinal:04d}.png"
        await asyncio.to_thread(shutil.copy2, produced_path, str(dest))
        rel = f"{reference}/output/{attempt_path(reference)}/{dest.name}"
        size = dest.stat().st_size
        await self.store.record_image_output(
            job_id, ordinal, rel, size, "image/png", image_count, done,
            original_filename=original_filename)
        # batch_id is threaded through so BATCH SSE watchers also receive the
        # per-image ticks (the hub routes on both job_id and batch_id).
        pct = image_pct(done, image_count)
        await self.orchestrator._emit(
            reference, batch_id, status="running",
            pct=pct, progress=pct,
            ordinal=ordinal, images_total=image_count, images_done=done,
            message=f"image {ordinal + 1}/{image_count} done")

    async def _reject_unimplemented(self, reference: str, job_type: str) -> AckDecision:
        job = await self.store.get_job_by_reference(reference)
        if job is None:
            return AckDecision.ACK  # poison
        await self.store.set_failed(job["id"], f"unsupported job_type {job_type!r}")
        await self.store.add_event(job["id"], "dispatch", f"unsupported job_type {job_type!r}")
        logger.warning("reference %s unsupported job_type %r -> DLQ", reference, job_type)
        return AckDecision.NACK_DLQ


class _PoisonError(Exception):
    """Bad per-ordinal inputs (viton with zero rows / missing role) — set_failed + ACK."""
