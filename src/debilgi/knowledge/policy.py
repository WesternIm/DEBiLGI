"""Event-driven policy for deciding when an external concept lookup is useful."""

from __future__ import annotations

import re

from debilgi.cognition.models import ActionIntent
from debilgi.events.models import Event, EventType
from debilgi.knowledge.models import KnowledgeDecision, WikipediaResult
from debilgi.knowledge.text import normalize_key
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.content import content_tokens
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.recovery import is_social_routine

_EXPLICIT_RE = re.compile(
    r"(?:что\s+такое|кто\s+(?:такой|такая|такие)|что\s+за|"
    r"расскажи\s+(?:про|о)|объясни(?:\s+что\s+такое)?)\s+(.+)",
    re.IGNORECASE,
)
_LOOKUP_RE = re.compile(r"\b(?:узнай|найди|проверь|поищи)\b(?:\s+(.+))?", re.IGNORECASE)
_LOOKUP_FILLERS = frozenset(
    {
        "а",
        "давай",
        "же",
        "ка",
        "ну",
        "пожалуйста",
        "ты",
    }
)
MEDIA_ARTICLE_MARKERS = (
    "телесериал",
    "сериал",
    "фильм",
    "мультфильм",
    "видеоигр",
    "компьютерн",
    "альбом",
    "песня",
    "телешоу",
    "аниме",
)


def media_article_reasons(result: WikipediaResult) -> tuple[str, ...]:
    """Return metadata/intro markers that identify a media work."""

    evidence = normalize_key(" ".join((result.title, result.extract, *result.categories)))
    return tuple(dict.fromkeys(marker for marker in MEDIA_ARTICLE_MARKERS if marker in evidence))


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
        semantic_content = DEFAULT_ADDRESSING.analyze(content).content_text
        explicit = _EXPLICIT_RE.search(semantic_content)
        uncertainty = _context_number(intent, "state_uncertainty")
        if explicit:
            query = normalize_key(explicit.group(1))
            query = " ".join(query.split()[:6])
            query = canonical_lexical_query(query, semantic_content)
            return KnowledgeDecision(bool(query), query or None, "explicit", uncertainty)
        followup, lookup_query = self.followup_query(semantic_content)
        if followup:
            lookup_query = (
                canonical_lexical_query(lookup_query, semantic_content)
                if lookup_query
                else None
            )
            return KnowledgeDecision(
                lookup_query is not None,
                lookup_query,
                "explicit" if lookup_query else "followup",
                uncertainty,
            )
        if is_social_routine(semantic_content):
            return denied
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
            token
            for token in content_tokens(semantic_content)
            if len(token) >= 6
        )
        if not candidates:
            return denied
        query = max(candidates, key=lambda token: (len(token), token))
        query = canonical_lexical_query(query, semantic_content)
        return KnowledgeDecision(True, query, "uncertainty", uncertainty)

    @staticmethod
    def followup_query(content: str) -> tuple[bool, str | None]:
        """Recognize lookup commands and distinguish an object from a vocative."""

        semantic_content = DEFAULT_ADDRESSING.analyze(content).content_text
        match = _LOOKUP_RE.search(semantic_content)
        if match is None:
            return False, None
        tail = normalize_key(match.group(1) or "")
        tokens = [token for token in tail.split() if token not in _LOOKUP_FILLERS]
        for prefix in (
            ("значение", "слова"),
            ("что", "значит"),
            ("что", "такое"),
            ("кто", "такой"),
        ):
            if tuple(tokens[: len(prefix)]) == prefix:
                tokens = tokens[len(prefix) :]
                break
        if tokens[:1] == ["слово"]:
            tokens = tokens[1:]
        query = " ".join(tokens[:6])
        return True, query or None


def _context_number(intent: ActionIntent | None, key: str) -> float:
    if intent is None:
        return 0.0
    value = intent.decision_context.get(key, 0.0)
    return (
        max(0.0, min(1.0, float(value)))
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )


def canonical_lexical_query(query: str, context: str) -> str:
    if len(query.split()) != 1:
        return query
    reading = DEFAULT_MORPHOLOGY.reading_in_context(query, context)
    return normalize_key(reading.lemma) if reading is not None and reading.is_known else query
