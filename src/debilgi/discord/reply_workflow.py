"""Preparation and post-send bookkeeping for durable gateway reply jobs."""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime

from debilgi.events.models import Event
from debilgi.experience.models import PreparedReply
from debilgi.language.clarification_policy import requests_word_definition
from debilgi.language.tokenizer import tokenize

logger = logging.getLogger(__name__)


class ReplyWorkflow:
    def __init__(self, bot, pipeline, reply_service, context_gate, response_traces):
        self._bot = bot
        self._pipeline = pipeline
        self._reply_service = reply_service
        self._context_gate = context_gate
        self._response_traces = response_traces

    async def prepare(self, event: Event) -> PreparedReply | None:
        upstream_failure = None
        bot_addressed = bool(event.payload.get("bot_mentioned"))
        reply_to = event.payload.get("reply_to_message_id")
        if not await self._pipeline.wait_until_processed(
            event.event_id, timeout=5.0
        ):
            logger.warning("Message event was not processed in time: %s", event.event_id)
            upstream_failure = "event_processing_timeout"
        await self._response_traces.begin(
            event,
            prompt=str(event.payload.get("content", "")),
        )
        await self._response_traces.step(
            event.event_id,
            stage="event_pipeline",
            component="event_listener",
            status="completed" if upstream_failure is None else "failed",
            summary=(
                "event observers completed"
                if upstream_failure is None
                else "event observers did not complete normally"
            ),
            details={
                "bot_addressed": bot_addressed,
                "upstream_failure": upstream_failure,
                "reply_to_message_id": reply_to,
            },
        )
        bound_explanation = (
            await self._reply_service.prepare_bound_explanation(event)
            if upstream_failure is None
            else None
        )
        bound_short_act = (
            await self._reply_service.prepare_bound_short_act(event)
            if upstream_failure is None and bound_explanation is None
            else None
        )
        if bound_explanation is not None:
            prepared = bound_explanation
        elif bound_short_act is not None:
            prepared = bound_short_act
        elif bot_addressed:
            prepared = await self._reply_service.prepare_guaranteed_reply(
                event,
                upstream_failure=upstream_failure,
            )
        elif upstream_failure is None:
            try:
                prepared = await self._reply_service.prepare_initiative(event)
            except Exception:
                logger.exception("Failed to prepare initiative for event %s", event.event_id)
                prepared = None
        else:
            prepared = None
        if prepared:
            await self._response_traces.step(
                event.event_id,
                stage="route",
                component="event_listener",
                status="selected",
                summary="reply route selected",
                details={
                    "response_source": prepared.response_source,
                    "fallback": prepared.fallback,
                    "initiative": prepared.initiative,
                    "candidate_count": prepared.candidate_count,
                    "text": prepared.text,
                },
            )
            return prepared
        elif bot_addressed:
            await self._response_traces.finish(
                event.event_id,
                status=(
                    "withheld"
                    if self._reply_service.productive_only_replies
                    else "failed"
                ),
                failure_stage="preparation",
                failure_reason=(
                    "productive_only_no_surface"
                    if self._reply_service.productive_only_replies
                    else "no_prepared_reply"
                ),
            )
        else:
            await self._response_traces.finish(
                event.event_id,
                status="no_reply",
                selected_route="initiative_not_selected",
            )

        return None

    async def validate(self, event: Event, prepared: PreparedReply) -> PreparedReply | None:
        if requests_word_definition(prepared.text):
            await self._response_traces.finish(
                event.event_id, status="withheld", failure_stage="autonomous_research",
                failure_reason="automatic_definition_question_disabled",
            )
            return None
        decision = await self._context_gate.evaluate(
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            candidate_text=prepared.text,
            source=prepared.response_source,
            now=datetime.now(UTC),
        )
        await self._response_traces.step(
            event.event_id,
            stage="context_gate",
            component="working_memory_context_gate",
            status="allowed" if decision.allowed else "withheld",
            summary=decision.reason,
            details={
                "candidate_text": prepared.text,
                "output_text": decision.text,
                "source": prepared.response_source,
            },
        )
        if not decision.allowed or decision.text is None:
            logger.info(
                "Context gate withheld event=%s source=%s reason=%s",
                event.event_id,
                prepared.response_source,
                decision.reason,
            )
            await self._response_traces.finish(
                event.event_id,
                status="withheld",
                selected_route=prepared.response_source,
                final_text=prepared.text,
                arbitration_decision_id=prepared.arbitration_decision_id,
                failure_stage="context_gate",
                failure_reason=decision.reason,
            )
            return
        if decision.text != prepared.text:
            prepared = replace(
                prepared,
                text=decision.text,
                tokens=tokenize(decision.text),
                fallback=True,
                response_source="fallback",
            )
        return prepared

    async def record_sent(self, event: Event, prepared: PreparedReply, sent) -> None:
        await self._response_traces.step(
            event.event_id,
            stage="send",
            component="discord_api",
            status="sent",
            summary="Discord accepted the reply",
            details={
                "response_message_id": sent.id,
                "text": prepared.text,
            },
            created_at=sent.created_at,
        )
        await self._context_gate.commit_outgoing(
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            message_id=sent.id,
            text=prepared.text,
            bot_user_id=self._bot.user.id if self._bot.user is not None else None,
            target_user_id=event.actor_id,
            observed_at=sent.created_at,
        )
        await self._reply_service.record_sent(
            event,
            response_message_id=sent.id,
            reply=prepared,
            created_at=sent.created_at,
        )
        await self._response_traces.finish(
            event.event_id,
            status="sent",
            selected_route=prepared.response_source,
            final_text=prepared.text,
            response_message_id=sent.id,
            arbitration_decision_id=prepared.arbitration_decision_id,
            completed_at=sent.created_at,
        )
