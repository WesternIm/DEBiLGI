"""Optional last-resort semantic consultation without surrendering response control."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

import aiohttp

from debilgi.language.construction_nlg import ConstructionSpec
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.semantic_consultant")

_MAX_PROMPT_CHARS = 2_000
_MAX_CONTEXT_MESSAGES = 8
_MAX_CONTEXT_CHARS = 4_000
_MAX_FACT_CHARS = 500


@dataclass(frozen=True, slots=True)
class ConsultantEntity:
    surface: str
    lemma: str
    entity_type: str


@dataclass(frozen=True, slots=True)
class ConsultantRelation:
    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class ConsultantAmbiguity:
    surface: str
    candidates: tuple[str, ...]
    selected: str
    reason: str


@dataclass(frozen=True, slots=True)
class ConsultantToken:
    surface: str
    lemma: str
    part_of_speech: str
    role: str
    features: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ConsultantSocialFrame:
    """Word-free social interpretation supplied by an optional teacher."""

    prompt_act: str
    speaker_stance: str
    emotion: str
    response_obligation: str
    recommended_moves: tuple[str, ...]
    avoid_moves: tuple[str, ...]
    directness: float
    warmth: float
    playfulness: float


@dataclass(frozen=True, slots=True)
class ConsultationRequest:
    prompt: str
    context_messages: tuple[str, ...]
    known_terms: tuple[str, ...]
    unknown_terms: tuple[str, ...]
    local_relations: tuple[tuple[str, str, str], ...]
    local_confidence: float
    reason_codes: tuple[str, ...]
    mode: str = "utterance"

    @property
    def input_hash(self) -> str:
        material = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "prompt": self.prompt[:_MAX_PROMPT_CHARS],
            "context_messages": list(_bounded_context(self.context_messages)),
            "known_terms": list(self.known_terms[:24]),
            "unknown_terms": list(self.unknown_terms[:16]),
            "local_relations": [list(item) for item in self.local_relations[:24]],
            "local_confidence": _clamp(self.local_confidence),
            "reason_codes": list(self.reason_codes[:12]),
        }


@dataclass(frozen=True, slots=True)
class ConsultationResult:
    interpretation: str
    intent: str
    entities: tuple[ConsultantEntity, ...]
    relations: tuple[ConsultantRelation, ...]
    ambiguities: tuple[ConsultantAmbiguity, ...]
    tokens: tuple[ConsultantToken, ...]
    sentence_pattern: tuple[str, ...]
    grammar_rules: tuple[str, ...]
    constructions: tuple[ConstructionSpec, ...]
    expected_response_acts: tuple[str, ...]
    social_frame: ConsultantSocialFrame | None
    register: str
    facts: tuple[str, ...]
    response_text: str | None
    confidence: float
    provider: str
    model: str
    cached: bool = False

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        provider: str,
        model: str,
    ) -> ConsultationResult:
        interpretation = _clean_text(payload.get("interpretation"), 700)
        intent = _clean_token(payload.get("intent"), "unknown")
        entities = tuple(
            ConsultantEntity(
                surface=_clean_text(item.get("surface"), 120),
                lemma=_clean_token(item.get("lemma"), ""),
                entity_type=_clean_token(item.get("type"), "unknown"),
            )
            for item in _mapping_items(payload.get("entities"), 16)
            if _clean_text(item.get("surface"), 120)
        )
        relations = tuple(
            ConsultantRelation(
                source=_clean_text(item.get("source"), 120),
                relation=_clean_token(item.get("relation"), "associated_with"),
                target=_clean_text(item.get("target"), 120),
            )
            for item in _mapping_items(payload.get("relations"), 20)
            if _clean_text(item.get("source"), 120)
            and _clean_text(item.get("target"), 120)
        )
        ambiguities = tuple(
            ConsultantAmbiguity(
                surface=_clean_text(item.get("surface"), 120),
                candidates=_text_items(item.get("candidates"), 8, 120),
                selected=_clean_text(item.get("selected"), 120),
                reason=_clean_text(item.get("reason"), 400),
            )
            for item in _mapping_items(payload.get("ambiguities"), 12)
            if _clean_text(item.get("surface"), 120)
        )
        tokens = tuple(
            ConsultantToken(
                surface=_clean_text(item.get("surface"), 120),
                lemma=_clean_token(item.get("lemma"), ""),
                part_of_speech=_clean_token(item.get("part_of_speech"), "unknown"),
                role=_clean_token(item.get("role"), "content"),
                features=_clean_features(item.get("features")),
            )
            for item in _mapping_items(payload.get("tokens"), 32)
            if _clean_text(item.get("surface"), 120)
        )
        sentence_pattern = _text_items(payload.get("sentence_pattern"), 24, 80)
        grammar_rules = tuple(
            item
            for item in _text_items(payload.get("grammar_rules"), 8, 300)
            if internal_surface_reason(item) is None
        )
        try:
            confidence = _clamp(float(payload.get("confidence", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        constructions = tuple(
            spec
            for item in _mapping_items(payload.get("constructions"), 8)
            if (
                spec := ConstructionSpec.from_mapping(
                    item,
                    source="semantic_consultant",
                    confidence=min(confidence, 0.78),
                )
            )
            is not None
        )
        expected_response_acts = tuple(
            _clean_token(item, "")
            for item in _text_items(payload.get("expected_response_acts"), 8, 80)
            if _clean_token(item, "")
        )
        social_frame = _social_frame(payload.get("social_frame"))
        if not sentence_pattern:
            sentence_pattern = tuple(
                token.role
                for token in tokens
                if token.role
                not in {"function", "function_word", "address", "punctuation"}
            )
        if not expected_response_acts and social_frame is not None:
            expected_response_acts = tuple(
                dict.fromkeys(
                    (
                        social_frame.response_obligation,
                        *social_frame.recommended_moves,
                    )
                )
            )
        register = _clean_token(payload.get("register"), "neutral")
        facts = tuple(
            item
            for item in _text_items(payload.get("facts"), 8, _MAX_FACT_CHARS)
            if internal_surface_reason(item) is None
        )
        if not interpretation and not facts:
            raise ValueError("consultation contains no usable interpretation")
        return cls(
            interpretation=interpretation,
            intent=intent,
            entities=entities,
            relations=relations,
            ambiguities=ambiguities,
            tokens=tokens,
            sentence_pattern=sentence_pattern,
            grammar_rules=grammar_rules,
            constructions=constructions,
            expected_response_acts=expected_response_acts,
            social_frame=social_frame,
            register=register,
            facts=facts,
            # Compatibility field for stored consultations.  External models
            # are semantic teachers only; their ready-made wording is never
            # accepted into the response pipeline.
            response_text=None,
            confidence=min(confidence, 0.86),
            provider=provider,
            model=model,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "interpretation": self.interpretation,
            "intent": self.intent,
            "entities": [
                {"surface": item.surface, "lemma": item.lemma, "type": item.entity_type}
                for item in self.entities
            ],
            "relations": [
                {"source": item.source, "relation": item.relation, "target": item.target}
                for item in self.relations
            ],
            "ambiguities": [
                {
                    "surface": item.surface,
                    "candidates": list(item.candidates),
                    "selected": item.selected,
                    "reason": item.reason,
                }
                for item in self.ambiguities
            ],
            "tokens": [
                {
                    "surface": item.surface,
                    "lemma": item.lemma,
                    "part_of_speech": item.part_of_speech,
                    "role": item.role,
                    "features": dict(item.features),
                }
                for item in self.tokens
            ],
            "sentence_pattern": list(self.sentence_pattern),
            "grammar_rules": list(self.grammar_rules),
            "constructions": [item.as_dict() for item in self.constructions],
            "expected_response_acts": list(self.expected_response_acts),
            "social_frame": (
                {
                    "prompt_act": self.social_frame.prompt_act,
                    "speaker_stance": self.social_frame.speaker_stance,
                    "emotion": self.social_frame.emotion,
                    "response_obligation": self.social_frame.response_obligation,
                    "recommended_moves": list(self.social_frame.recommended_moves),
                    "avoid_moves": list(self.social_frame.avoid_moves),
                    "directness": self.social_frame.directness,
                    "warmth": self.social_frame.warmth,
                    "playfulness": self.social_frame.playfulness,
                }
                if self.social_frame is not None
                else None
            ),
            "register": self.register,
            "facts": list(self.facts),
            "response_text": self.response_text,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
        }


class SemanticConsultant(Protocol):
    async def consult(self, request: ConsultationRequest) -> ConsultationResult | None: ...

    async def close(self) -> None: ...


class SemanticAssimilator(Protocol):
    async def assimilate(
        self,
        *,
        guild_id: int,
        request: ConsultationRequest,
        result: ConsultationResult,
        observed_at: datetime | None = None,
    ) -> int: ...


class PuterSidecarConsultant:
    """Call the local Node/Puter bridge and validate its structured response."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        timeout_seconds: int = 12,
        shared_secret: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.shared_secret = shared_secret
        self._session: aiohttp.ClientSession | None = None
        self.last_failure_retryable = False

    async def consult(self, request: ConsultationRequest) -> ConsultationResult | None:
        self.last_failure_retryable = False
        session = await self._get_session()
        headers = {"Content-Type": "application/json"}
        if self.shared_secret:
            headers["Authorization"] = f"Bearer {self.shared_secret}"
        try:
            async with session.post(
                self.endpoint,
                json={"model": self.model, "request": request.as_dict()},
                headers=headers,
            ) as response:
                if response.status != 200:
                    detail = "unknown_error"
                    try:
                        error_payload = await response.json(content_type=None)
                        if isinstance(error_payload, Mapping):
                            detail = _clean_text(error_payload.get("error"), 240)
                    except (aiohttp.ClientError, json.JSONDecodeError, TypeError):
                        pass
                    logger.warning(
                        "Puter sidecar returned HTTP %d: %s",
                        response.status,
                        detail or "unknown_error",
                    )
                    normalized_detail = (detail or "").casefold()
                    self.last_failure_retryable = response.status in {
                        408,
                        429,
                        502,
                        503,
                        504,
                    } or any(
                        marker in normalized_detail
                        for marker in (
                            "usage left",
                            "rate limit",
                            "timeout",
                            "temporarily",
                            "unavailable",
                            "bridge",
                        )
                    )
                    return None
                envelope = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self.last_failure_retryable = True
            logger.warning("Puter semantic consultation failed: %s", type(exc).__name__)
            return None
        content = envelope.get("content") if isinstance(envelope, Mapping) else None
        parsed = _parse_json_object(content)
        if parsed is None:
            logger.warning("Puter semantic consultation returned invalid JSON")
            return None
        provider = _clean_token(envelope.get("provider"), "puter")
        model = _clean_token(envelope.get("model"), self.model)
        try:
            return ConsultationResult.from_payload(parsed, provider=provider, model=model)
        except ValueError as exc:
            logger.warning("Rejected Puter consultation: %s", exc)
            return None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            )
        return self._session


class CloudflareWorkersAIConsultant:
    """Use Workers AI as a structured teacher, never as a reply generator."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str,
        instruction: str,
        timeout_seconds: int = 45,
    ) -> None:
        self.account_id = account_id.strip()
        self.api_token = api_token.strip()
        self.model = model.strip()
        self.instruction = instruction.strip()
        self.timeout_seconds = timeout_seconds
        self.endpoint = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )
        self._session: aiohttp.ClientSession | None = None
        self.last_failure_retryable = False

    async def consult(self, request: ConsultationRequest) -> ConsultationResult | None:
        self.last_failure_retryable = False
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": json.dumps(request.as_dict(), ensure_ascii=False),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": _consultation_response_schema(),
            },
            "temperature": 0.1,
            "max_tokens": 2_048,
        }
        try:
            async with session.post(
                self.endpoint,
                json=payload,
                headers=headers,
            ) as response:
                envelope = await response.json(content_type=None)
                if response.status != 200:
                    detail = _cloudflare_error_detail(envelope)
                    logger.warning(
                        "Cloudflare Workers AI returned HTTP %d: %s",
                        response.status,
                        detail,
                    )
                    self.last_failure_retryable = response.status in {
                        408,
                        409,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    return None
        except (aiohttp.ClientError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self.last_failure_retryable = True
            logger.warning(
                "Cloudflare semantic consultation failed: %s",
                type(exc).__name__,
            )
            return None

        result = envelope.get("result") if isinstance(envelope, Mapping) else None
        content = result.get("response") if isinstance(result, Mapping) else None
        parsed = content if isinstance(content, Mapping) else _parse_json_object(content)
        if parsed is None:
            logger.warning("Cloudflare semantic consultation returned invalid JSON")
            return None
        try:
            return ConsultationResult.from_payload(
                parsed,
                provider="cloudflare-workers-ai",
                model=self.model,
            )
        except ValueError as exc:
            logger.warning("Rejected Cloudflare consultation: %s", exc)
            return None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            )
        return self._session


class SemanticConsultationRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def cached(self, input_hash: str, now: datetime) -> ConsultationResult | None:
        row = await self._database.fetchone(
            """SELECT result_json FROM language_semantic_consultations
               WHERE input_hash = ? AND status IN ('provisional', 'confirmed')
                 AND expires_at > ?
               ORDER BY updated_at DESC LIMIT 1""",
            (input_hash, now.isoformat()),
        )
        if row is None:
            return None
        return _stored_result(row["result_json"])

    async def cached_lexical(
        self,
        *,
        guild_id: int,
        terms: Sequence[str],
        now: datetime,
    ) -> ConsultationResult | None:
        wanted = {item.casefold() for item in terms if item.strip()}
        if not wanted:
            return None
        rows = await self._database.fetchall(
            """SELECT subject_terms_json, result_json
               FROM language_semantic_consultations
               WHERE guild_id = ? AND mode = 'lexical'
                 AND status IN ('provisional', 'confirmed') AND expires_at > ?
               ORDER BY result_confidence DESC, updated_at DESC LIMIT 20""",
            (guild_id, now.isoformat()),
        )
        for row in rows:
            try:
                stored_terms = {
                    str(item).casefold()
                    for item in json.loads(str(row["subject_terms_json"]))
                }
            except (json.JSONDecodeError, TypeError):
                continue
            if wanted & stored_terms:
                return _stored_result(row["result_json"])
        return None

    async def save(
        self,
        *,
        event_id: str | None,
        guild_id: int,
        channel_id: int | None,
        request: ConsultationRequest,
        result: ConsultationResult,
        now: datetime,
        ttl: timedelta,
    ) -> None:
        consultation_id = f"consultation:{uuid4()}"
        await self._database.execute(
            """INSERT INTO language_semantic_consultations (
                   consultation_id, event_id, guild_id, channel_id, input_hash,
                   mode, subject_terms_json, reason_codes_json, local_confidence,
                   provider, model,
                   result_json, result_confidence, status, created_at, updated_at,
                   expires_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'provisional', ?, ?, ?)
               ON CONFLICT (guild_id, input_hash) DO UPDATE SET
                   event_id = excluded.event_id,
                   channel_id = excluded.channel_id,
                   subject_terms_json = excluded.subject_terms_json,
                   reason_codes_json = excluded.reason_codes_json,
                   local_confidence = excluded.local_confidence,
                   provider = excluded.provider,
                   model = excluded.model,
                   result_json = excluded.result_json,
                   result_confidence = excluded.result_confidence,
                   updated_at = excluded.updated_at,
                   expires_at = excluded.expires_at,
                   occurrence_count = language_semantic_consultations.occurrence_count + 1""",
            (
                consultation_id,
                event_id,
                guild_id,
                channel_id,
                request.input_hash,
                request.mode,
                json.dumps(
                    tuple(dict.fromkeys((*request.unknown_terms, *request.known_terms))),
                    ensure_ascii=False,
                ),
                json.dumps(request.reason_codes, ensure_ascii=False),
                request.local_confidence,
                result.provider,
                result.model,
                json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True),
                result.confidence,
                now.isoformat(),
                now.isoformat(),
                (now + ttl).isoformat(),
            ),
        )


class SemanticConsultationService:
    """Gate, cache and audit external help while keeping it provisional."""

    def __init__(
        self,
        database: Database,
        consultant: SemanticConsultant,
        *,
        enabled: bool = False,
        trigger_threshold: float = 0.72,
        minimum_result_confidence: float = 0.58,
        cache_ttl: timedelta = timedelta(days=14),
        assimilator: SemanticAssimilator | None = None,
    ) -> None:
        self._repository = SemanticConsultationRepository(database)
        self._consultant = consultant
        self.enabled = enabled
        self.trigger_threshold = _clamp(trigger_threshold)
        self.minimum_result_confidence = _clamp(minimum_result_confidence)
        self.cache_ttl = cache_ttl
        self._assimilator = assimilator

    @property
    def last_failure_retryable(self) -> bool:
        return bool(getattr(self._consultant, "last_failure_retryable", False))

    def should_consult(
        self,
        *,
        local_confidence: float,
        unknown_terms: Sequence[str],
        has_bound_answer: bool,
    ) -> bool:
        # A high lexical score alone is not proof that the system knows what to
        # say: the old pipeline frequently scored every word as known while
        # failing to construct a reply-bound proposition.
        return self.enabled and not has_bound_answer

    async def consult(
        self,
        *,
        event_id: str | None,
        guild_id: int,
        channel_id: int | None,
        request: ConsultationRequest,
    ) -> ConsultationResult | None:
        if not self.enabled:
            return None
        now = datetime.now(UTC)
        cached = await self._repository.cached(request.input_hash, now)
        if cached is not None:
            logger.info(
                "event=%s mode=%s result=exact_cache confidence=%.3f",
                event_id,
                request.mode,
                cached.confidence,
            )
            return cached
        if request.mode == "utterance" and request.unknown_terms:
            lexical = await self._repository.cached_lexical(
                guild_id=guild_id,
                terms=request.unknown_terms,
                now=now,
            )
            if (
                lexical is not None
                and (lexical.interpretation or lexical.facts)
                and _utterance_requests_lexical_meaning(request.prompt)
            ):
                if self._assimilator is not None:
                    await self._assimilator.assimilate(
                        guild_id=guild_id,
                        request=request,
                        result=lexical,
                        observed_at=now,
                    )
                logger.info(
                    "event=%s mode=utterance result=lexical_cache "
                    "terms=%s confidence=%.3f",
                    event_id,
                    request.unknown_terms,
                    lexical.confidence,
                )
                return lexical
        result = await self._consultant.consult(request)
        if result is None:
            logger.info(
                "event=%s mode=%s result=provider_empty retryable=%s reasons=%s",
                event_id,
                request.mode,
                self.last_failure_retryable,
                request.reason_codes,
            )
            return None
        if result.confidence < self.minimum_result_confidence:
            logger.info(
                "event=%s mode=%s result=below_confidence confidence=%.3f "
                "minimum=%.3f intent=%s",
                event_id,
                request.mode,
                result.confidence,
                self.minimum_result_confidence,
                result.intent,
            )
            return None
        await self._repository.save(
            event_id=event_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request=request,
            result=result,
            now=now,
            ttl=self.cache_ttl,
        )
        assimilated = 0
        if self._assimilator is not None:
            assimilated = await self._assimilator.assimilate(
                guild_id=guild_id,
                request=request,
                result=result,
                observed_at=now,
            )
        logger.info(
            "event=%s mode=%s provider=%s model=%s confidence=%.3f status=%s lessons=%d",
            event_id,
            request.mode,
            result.provider,
            result.model,
            result.confidence,
            "confirmed" if assimilated else "provisional",
            assimilated,
        )
        return result

    async def close(self) -> None:
        await self._consultant.close()


def _utterance_requests_lexical_meaning(prompt: str) -> bool:
    """Only reuse a lexical lesson when the turn actually asks for meaning.

    A cached dictionary entry for ``ключ`` cannot answer ``ты терял ключи?``.
    The previous shortcut returned that entry as if it were a full utterance
    analysis and prevented the teacher from resolving intent and reference.
    """

    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    lexical_markers = (
        "что такое",
        "что значит",
        "что означает",
        "значение слова",
        "значение выражения",
        "как употребляется",
        "как использовать слово",
    )
    return any(marker in normalized for marker in lexical_markers)


def _parse_json_object(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, Mapping) else None


def _consultation_response_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "interpretation": {"type": "string"},
            "intent": {"type": "string"},
            "entities": {"type": "array", "items": {"type": "object"}},
            "relations": {"type": "array", "items": {"type": "object"}},
            "ambiguities": {"type": "array", "items": {"type": "object"}},
            "tokens": {"type": "array", "items": {"type": "object"}},
            "sentence_pattern": string_array,
            "grammar_rules": string_array,
            "constructions": {"type": "array", "items": {"type": "object"}},
            "expected_response_acts": string_array,
            "social_frame": {"type": "object"},
            "register": {"type": "string"},
            "facts": string_array,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["interpretation", "intent", "confidence"],
    }


def _cloudflare_error_detail(envelope: object) -> str:
    if not isinstance(envelope, Mapping):
        return "unknown_error"
    errors = envelope.get("errors")
    if isinstance(errors, list):
        messages = tuple(
            _clean_text(item.get("message"), 200)
            for item in errors[:3]
            if isinstance(item, Mapping) and _clean_text(item.get("message"), 200)
        )
        if messages:
            return "; ".join(messages)[:240]
    return _clean_text(envelope.get("message"), 240) or "unknown_error"


def _mapping_items(value: object, limit: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value[:limit] if isinstance(item, Mapping))


def _social_frame(value: object) -> ConsultantSocialFrame | None:
    if not isinstance(value, Mapping):
        return None
    prompt_act = _clean_token(value.get("prompt_act"), "")
    obligation = _clean_token(value.get("response_obligation"), "")
    recommended = tuple(
        _clean_token(item, "")
        for item in _text_items(value.get("recommended_moves"), 8, 80)
        if _clean_token(item, "")
    )
    if not prompt_act or not obligation or not recommended:
        return None
    return ConsultantSocialFrame(
        prompt_act=prompt_act,
        speaker_stance=_clean_token(value.get("speaker_stance"), "neutral"),
        emotion=_clean_token(value.get("emotion"), "neutral"),
        response_obligation=obligation,
        recommended_moves=recommended,
        avoid_moves=tuple(
            _clean_token(item, "")
            for item in _text_items(value.get("avoid_moves"), 8, 80)
            if _clean_token(item, "")
        ),
        directness=_safe_number(value.get("directness"), 0.5),
        warmth=_safe_number(value.get("warmth"), 0.5),
        playfulness=_safe_number(value.get("playfulness"), 0.0),
    )


def _stored_result(value: object) -> ConsultationResult | None:
    try:
        payload = json.loads(str(value))
        result = ConsultationResult.from_payload(
            payload,
            provider=_clean_token(payload.get("provider"), "puter"),
            model=_clean_token(payload.get("model"), "unknown"),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return ConsultationResult(
        **{
            name: getattr(result, name)
            for name in result.__dataclass_fields__
            if name != "cached"
        },
        cached=True,
    )


def _bounded_context(messages: Sequence[str]) -> tuple[str, ...]:
    selected: list[str] = []
    remaining = _MAX_CONTEXT_CHARS
    for message in reversed(messages[-_MAX_CONTEXT_MESSAGES:]):
        cleaned = _clean_text(message, min(_MAX_PROMPT_CHARS, remaining))
        if not cleaned:
            continue
        selected.append(cleaned)
        remaining -= len(cleaned)
        if remaining <= 0:
            break
    return tuple(reversed(selected))


def _text_items(value: object, limit: int, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(
            cleaned
            for item in value[:limit]
            if (cleaned := _clean_text(item, maximum))
        )
    )


def _clean_text(value: object, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split())[:maximum].strip()


def _clean_features(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        return ()
    cleaned: list[tuple[str, str]] = []
    for raw_key, raw_value in list(value.items())[:12]:
        key = _clean_token(raw_key, "")
        feature = _clean_token(raw_value, "")
        if key and feature:
            cleaned.append((key, feature))
    return tuple(dict.fromkeys(cleaned))


def _clean_token(value: object, default: str) -> str:
    cleaned = _clean_text(value, 120).casefold()
    return cleaned or default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_number(value: object, default: float) -> float:
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return default
