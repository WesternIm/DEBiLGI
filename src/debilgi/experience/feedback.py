"""One conservative feedback attribution contract for every learning consumer."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.events.models import Event, EventType
from debilgi.language.repair_signals import signals_prior_answer_correction
from debilgi.language.tokenizer import tokenize
from debilgi.social.signals import AFFIRMATION, APPROVAL, AVERSION, DISAPPROVAL, REFUSAL

_POSITIVE_REACTIONS = frozenset({"❤️", "❤", "👍", "✅", "🔥", "💯", "🥰", "😍", "🤩"})
_NEGATIVE_REACTIONS = frozenset({"👎", "🤮", "💩", "❌", "🤢"})
_NEGATIVE_PHRASES = (
    "не так",
    "неверно",
    "неправильно",
    "не понял",
    "не поняла",
    "ничего не понял",
    "ничего не поняла",
    "ошиб",
)


@dataclass(frozen=True, slots=True)
class AttributedFeedback:
    """A single interpretation of a user event, shared by all learners."""

    feedback_event_id: str
    target_message_id: int | None
    kind: str
    reward: float
    attribution_confidence: float
    explicit: bool

    @property
    def is_neutral(self) -> bool:
        return self.reward == 0.0

    def is_learnable(self, *, minimum_confidence: float = 0.60) -> bool:
        return not self.is_neutral and self.attribution_confidence >= minimum_confidence


class FeedbackAttributor:
    """Classify feedback without treating attention as approval."""

    def attribute(
        self,
        event: Event,
        *,
        fallback_target_message_id: int | None = None,
    ) -> AttributedFeedback:
        target = feedback_target(event)
        if target is None:
            target = fallback_target_message_id
        if event.event_type in {EventType.REACTION_ADDED, EventType.REACTION_REMOVED}:
            return self._reaction(event, target)
        if event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
            return self._neutral(event, target, "unsupported_event")
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return self._neutral(event, target, "empty_reply")
        normalized = content.casefold().replace("ё", "е")
        words = set(tokenize(normalized))
        if any(phrase in normalized for phrase in _NEGATIVE_PHRASES) or words & (
            DISAPPROVAL | AVERSION
        ):
            return AttributedFeedback(
                event.event_id,
                target,
                "explicit_rejection",
                -1.0,
                0.95,
                True,
            )
        if target is not None and signals_prior_answer_correction(content):
            return AttributedFeedback(
                event.event_id,
                target,
                "contextual_correction",
                -0.85,
                0.86,
                True,
            )
        if words & (APPROVAL | AFFIRMATION):
            return AttributedFeedback(
                event.event_id,
                target,
                "explicit_confirmation",
                1.0,
                0.95,
                True,
            )
        if words & REFUSAL:
            return AttributedFeedback(
                event.event_id,
                target,
                "refusal",
                -0.65,
                0.75,
                True,
            )
        return self._neutral(event, target, "conversation_continued", confidence=0.10)

    @staticmethod
    def _reaction(event: Event, target: int | None) -> AttributedFeedback:
        emoji = str(event.payload.get("emoji", ""))
        added = event.event_type is EventType.REACTION_ADDED
        direction = 1.0 if added else -1.0
        if emoji in _POSITIVE_REACTIONS:
            return AttributedFeedback(
                event.event_id,
                target,
                "positive_reaction" if added else "positive_reaction_removed",
                0.8 * direction,
                0.90,
                True,
            )
        if emoji in _NEGATIVE_REACTIONS:
            return AttributedFeedback(
                event.event_id,
                target,
                "negative_reaction" if added else "negative_reaction_removed",
                -0.8 * direction,
                0.90,
                True,
            )
        return FeedbackAttributor._neutral(event, target, "neutral_reaction")

    @staticmethod
    def _neutral(
        event: Event,
        target: int | None,
        kind: str,
        *,
        confidence: float = 0.0,
    ) -> AttributedFeedback:
        return AttributedFeedback(event.event_id, target, kind, 0.0, confidence, False)


def feedback_target(event: Event) -> int | None:
    if event.event_type in {EventType.REACTION_ADDED, EventType.REACTION_REMOVED}:
        value = event.payload.get("message_id")
    elif event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
        value = event.payload.get("reply_to_message_id")
    else:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


DEFAULT_FEEDBACK_ATTRIBUTOR = FeedbackAttributor()
