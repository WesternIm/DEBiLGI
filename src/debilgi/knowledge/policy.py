"""Event-driven policy for deciding when an external concept lookup is useful."""

from __future__ import annotations

import re

from debilgi.cognition.models import ActionIntent
from debilgi.events.models import Event, EventType
from debilgi.knowledge.models import KnowledgeDecision
from debilgi.knowledge.text import normalize_key, words

_EXPLICIT_RE = re.compile(
    r"(?:что\s+такое|кто\s+(?:такой|такая|такие)|что\s+за|"
    r"расскажи\s+(?:про|о)|объясни(?:\s+что\s+такое)?)\s+(.+)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "это",
        "как",
        "что",
        "кто",
        "где",
        "когда",
        "зачем",
        "почему",
        "тебя",
        "тебе",
        "меня",
        "мне",
        "тоже",
        "только",
        "сейчас",
        "потом",
        "просто",
        "очень",
        "вообще",
        "если",
        "или",
        "для",
        "про",
        "там",
        "тут",
        "уже",
        "еще",
        "ещё",
        "блять",
        "нахуй",
        "дебил",
        "еблан",
        "the",
        "and",
        "this",
        "that",
        "with",
        "from",
        "what",
        "who",
    }
)


class KnowledgeQueryPolicy:
    def __init__(
        self,
        *,
        enabled: bool = False,
        uncertainty_threshold: float = 0.68,
        interest_threshold: float = 0.48,
    ) -> None:
        self.enabled = enabled
        self.uncertainty_threshold = uncertainty_threshold
        self.interest_threshold = interest_threshold

    def decide(self, event: Event, intent: ActionIntent | None) -> KnowledgeDecision:
        denied = KnowledgeDecision(False, None, None, 0.0)
        if (
            not self.enabled
            or event.channel_id is None
            or event.actor_id is None
            or event.payload.get("author_is_bot") is True
            or event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
        ):
            return denied
        content = event.payload.get("content")
        if not isinstance(content, str):
            return denied
        explicit = _EXPLICIT_RE.search(content)
        uncertainty = _context_number(intent, "state_uncertainty")
        if explicit:
            query = normalize_key(explicit.group(1))
            query = " ".join(query.split()[:6])
            return KnowledgeDecision(bool(query), query or None, "explicit", uncertainty)
        interest = _context_number(intent, "state_interest")
        unexpectedness = _context_number(intent, "state_unexpectedness")
        if (
            intent is None
            or uncertainty < self.uncertainty_threshold
            or interest < self.interest_threshold
            or unexpectedness < 0.30
        ):
            return denied
        candidates = tuple(
            token for token in words(content) if token not in _STOPWORDS and len(token) >= 6
        )
        if not candidates:
            return denied
        query = max(candidates, key=lambda token: (len(token), token))
        return KnowledgeDecision(True, query, "uncertainty", uncertainty)


def _context_number(intent: ActionIntent | None, key: str) -> float:
    if intent is None:
        return 0.0
    value = intent.decision_context.get(key, 0.0)
    return (
        max(0.0, min(1.0, float(value)))
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )
