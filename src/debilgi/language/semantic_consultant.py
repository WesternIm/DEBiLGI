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
from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.semantic_consultant")

_MAX_PROMPT_CHARS = 2_000
_MAX_CONTEXT_MESSAGES = 8
_MAX_CONTEXT_CHARS = 4_000
_MAX_FACT_CHARS = 500
_PLANNING_INSTRUCTION = """
Ты — семантический планировщик русскоязычного агента DEBiLGI. Ты не пишешь ответ
пользователю. Верни только JSON по переданной схеме.

Сформируй 1–3 различных response_plans. Не дублируй одинаковые пропозиции. В них
разрешены только:
- абстрактный move и stance;
- лемма русского глагола в инфинитиве;
- аргументы как отдельные словарные леммы с семантическими ролями;
- грамматические признаки, отрицание и evidence_ids.

Запрещены готовые фразы, surface-текст, порядок слов, цитаты и поля answer/reply/response.
Каждое утверждение обязано ссылаться на существующий id из grounding_facts.
Не выдумывай факты и id.

Если grounding fact начинается с
predicate=...;subject=...;role=...;truth=...,
перенеси эту пропозицию точно:
- agent становится subject с lemma=я;
- каждый указанный role остаётся тем же role;
- truth=false означает negated=true, truth=true означает negated=false;
- субъект и объект нельзя переставлять или опускать;
- predicate должен оставаться глагольной леммой, а не названием темы.

personality_drives влияют только на выбор разных move и stance. Окончательный выбор и
русскую формулировку делает локальная система. В плане я означает DEBiLGI, ты —
пользователя. response_polarity используй только для прямого да/нет, иначе null.
""".strip()


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
class ConsultantResponseArgument:
    role: str
    lemma: str
    part_of_speech: str
    grammatical_case: str | None
    preposition: str | None
    modifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsultantResponseProposition:
    predicate: str
    predicate_features: tuple[tuple[str, str], ...]
    arguments: tuple[ConsultantResponseArgument, ...]
    clause_kind: str
    negated: bool
    response_polarity: bool | None
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsultantResponsePlan:
    move: str
    stance: str
    propositions: tuple[ConsultantResponseProposition, ...]
    confidence: float


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
    grounding_facts: tuple[tuple[str, str], ...] = ()
    personality_drives: tuple[tuple[str, float], ...] = ()
    response_obligation: str = ""

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
            "grounding_facts": [
                {"id": fact_id, "value": value} for fact_id, value in self.grounding_facts[:24]
            ],
            "personality_drives": {
                name: _clamp(value) for name, value in self.personality_drives[:24]
            },
            "response_obligation": self.response_obligation,
            "contract_version": 2,
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
    response_plans: tuple[ConsultantResponsePlan, ...]
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
            if _clean_text(item.get("source"), 120) and _clean_text(item.get("target"), 120)
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
        response_plans = _response_plans(payload.get("response_plans"))
        if not sentence_pattern:
            sentence_pattern = tuple(
                token.role
                for token in tokens
                if token.role not in {"function", "function_word", "address", "punctuation"}
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
            response_plans=response_plans,
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
            "response_plans": [
                {
                    "move": plan.move,
                    "stance": plan.stance,
                    "confidence": plan.confidence,
                    "propositions": [
                        {
                            "predicate": proposition.predicate,
                            "predicate_features": dict(proposition.predicate_features),
                            "arguments": [
                                {
                                    "role": argument.role,
                                    "lemma": argument.lemma,
                                    "part_of_speech": argument.part_of_speech,
                                    "grammatical_case": (argument.grammatical_case),
                                    "preposition": argument.preposition,
                                    "modifiers": list(argument.modifiers),
                                }
                                for argument in proposition.arguments
                            ],
                            "clause_kind": proposition.clause_kind,
                            "negated": proposition.negated,
                            "response_polarity": (proposition.response_polarity),
                            "evidence_ids": list(proposition.evidence_ids),
                        }
                        for proposition in plan.propositions
                    ],
                }
                for plan in self.response_plans
            ],
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
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        )
        self._session: aiohttp.ClientSession | None = None
        self.last_failure_retryable = False
        self._quota_blocked_until: datetime | None = None

    async def consult(self, request: ConsultationRequest) -> ConsultationResult | None:
        now = datetime.now(UTC)
        if self._quota_blocked_until is not None and now < self._quota_blocked_until:
            self.last_failure_retryable = True
            return None
        self.last_failure_retryable = False
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        _PLANNING_INSTRUCTION if request.mode == "planning" else self.instruction
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request.as_dict(), ensure_ascii=False),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": _consultation_response_schema(planning=request.mode == "planning"),
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
                    lowered_detail = detail.casefold()
                    if (
                        response.status == 429
                        and "daily" in lowered_detail
                        and ("allocation" in lowered_detail or "quota" in lowered_detail)
                    ):
                        tomorrow = (now + timedelta(days=1)).date()
                        self._quota_blocked_until = datetime.combine(
                            tomorrow,
                            datetime.min.time(),
                            tzinfo=UTC,
                        ) + timedelta(minutes=5)
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
                    str(item).casefold() for item in json.loads(str(row["subject_terms_json"]))
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
    def autonomous_planning(self) -> bool:
        """Enable planner-only replies only while the provider is configured."""

        return self.enabled

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
        # Response planning is invoked separately for every turn.  This gate
        # remains specific to semantic repair of an unresolved input.
        del local_confidence, unknown_terms
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
                    "event=%s mode=utterance result=lexical_cache terms=%s confidence=%.3f",
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
                "event=%s mode=%s result=below_confidence confidence=%.3f minimum=%.3f intent=%s",
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

    async def research_lexeme(
        self,
        *,
        event_id: str | None,
        guild_id: int,
        channel_id: int | None,
        term: str,
        context_messages: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Ask the teacher for reusable facts after reference sources fail.

        This deliberately uses lexical mode.  The provider still cannot return
        a ready-made reply: only topical facts or a lexical interpretation are
        admitted, and the local response pipeline remains responsible for the
        final surface.
        """

        normalized = " ".join(term.casefold().replace("ё", "е").split())
        if not self.enabled or not normalized:
            return ()
        result = await self.consult(
            event_id=event_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request=ConsultationRequest(
                prompt=normalized,
                context_messages=context_messages,
                known_terms=(),
                unknown_terms=(normalized,),
                local_relations=(),
                local_confidence=0.0,
                reason_codes=(
                    "reference_sources_exhausted",
                    "lexical_research",
                ),
                mode="lexical",
            ),
        )
        if result is None:
            return ()
        return _topical_research_facts(
            normalized,
            (*result.facts, result.interpretation),
        )

    async def research_question(
        self,
        *,
        event_id: str | None,
        guild_id: int,
        channel_id: int | None,
        prompt: str,
        context_messages: tuple[str, ...] = (),
        subject_terms: Sequence[str] = (),
        local_relations: Sequence[tuple[str, str, str]] = (),
    ) -> tuple[str, ...]:
        """Research facts required by an unresolved request before arbitration."""

        normalized_prompt = " ".join(prompt.split())
        normalized_terms = tuple(
            dict.fromkeys(
                term.casefold().replace("ё", "е").strip() for term in subject_terms if term.strip()
            )
        )
        if not self.enabled or not normalized_prompt:
            return ()
        result = await self.consult(
            event_id=event_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request=ConsultationRequest(
                prompt=normalized_prompt,
                context_messages=context_messages,
                known_terms=normalized_terms,
                unknown_terms=(),
                local_relations=tuple(local_relations),
                local_confidence=0.0,
                reason_codes=(
                    "no_reply_bound_local_candidate",
                    "foreground_answer_research",
                    "reference_sources_consulted",
                ),
                mode="research",
            ),
        )
        if result is None:
            return ()
        facts = _topical_research_facts(normalized_prompt, result.facts)
        return _facts_fitting_research_intent(
            result,
            normalized_terms,
            facts,
        )

    async def plan_response(
        self,
        *,
        event_id: str | None,
        guild_id: int,
        channel_id: int | None,
        prompt: str,
        context_messages: tuple[str, ...],
        known_terms: Sequence[str],
        local_relations: Sequence[tuple[str, str, str]],
        grounding_facts: Sequence[tuple[str, str]],
        personality_drives: Sequence[tuple[str, float]],
        response_obligation: str,
        local_confidence: float,
    ) -> ConsultationResult | None:
        """Request semantic alternatives without requesting reply wording."""

        if not self.enabled or not prompt.strip():
            return None
        return await self.consult(
            event_id=event_id,
            guild_id=guild_id,
            channel_id=channel_id,
            request=ConsultationRequest(
                prompt=" ".join(prompt.split()),
                context_messages=context_messages,
                known_terms=tuple(known_terms),
                unknown_terms=(),
                local_relations=tuple(local_relations),
                local_confidence=local_confidence,
                reason_codes=(
                    "autonomous_semantic_response_planning",
                    "no_external_surface_allowed",
                ),
                mode="planning",
                grounding_facts=tuple(grounding_facts),
                personality_drives=tuple(personality_drives),
                response_obligation=response_obligation,
            ),
        )

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


def _topical_research_facts(
    term: str,
    candidates: Sequence[str],
) -> tuple[str, ...]:
    """Keep definitions, not descriptions of the user's request."""

    target_terms = set(content_lemmas(term)) or {term}
    meta_markers = (
        "пользователь просит",
        "пользователь хочет",
        "определить значение",
        "объяснить значение",
        "запрос о",
        "вопрос о",
        "the user asks",
        "define the meaning",
    )
    accepted: list[str] = []
    for candidate in candidates:
        cleaned = _clean_text(candidate, _MAX_FACT_CHARS)
        normalized = cleaned.casefold().replace("ё", "е")
        if (
            not cleaned
            or internal_surface_reason(cleaned) is not None
            or any(marker in normalized for marker in meta_markers)
        ):
            continue
        fact_terms = set(content_lemmas(cleaned))
        topical = bool(target_terms & fact_terms) or any(
            same_lexical_family(target, fact) for target in target_terms for fact in fact_terms
        )
        if not topical or len(fact_terms) < 2:
            continue
        if cleaned not in accepted:
            accepted.append(cleaned)
    return tuple(accepted[:4])


def _facts_fitting_research_intent(
    result: ConsultationResult,
    subject_terms: Sequence[str],
    facts: Sequence[str],
) -> tuple[str, ...]:
    """Keep whole-proposition evidence for evaluations, not word definitions."""

    intent = result.intent.casefold().replace("-", "_")
    if not any(marker in intent for marker in ("opinion", "evaluation", "appraisal")):
        return tuple(facts)
    normalized_terms = tuple(
        dict.fromkeys(
            term.casefold().replace("ё", "е").strip() for term in subject_terms if term.strip()
        )
    )
    quality_terms = {
        token.lemma.casefold().replace("ё", "е")
        for token in result.tokens
        if token.lemma
        and token.part_of_speech == "adjective"
        and token.role in {"predicate", "predicate_complement", "modifier"}
    }
    target_terms = tuple(term for term in normalized_terms if term not in quality_terms)
    accepted: list[str] = []
    for fact in facts:
        fact_terms = tuple(content_lemmas(fact))
        matched_quality = any(
            same_lexical_family(quality, term) for quality in quality_terms for term in fact_terms
        )
        matched_target = any(
            same_lexical_family(target, term) for target in target_terms for term in fact_terms
        )
        if quality_terms:
            fits = matched_quality and matched_target
        else:
            matched_subjects = {
                subject
                for subject in normalized_terms
                if any(same_lexical_family(subject, term) for term in fact_terms)
            }
            fits = len(matched_subjects) >= 2
        if fits:
            accepted.append(fact)
    return tuple(accepted)


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


def _consultation_response_schema(
    *,
    planning: bool = False,
) -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    response_argument = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": [
                    "subject",
                    "direct_object",
                    "indirect_object",
                    "predicate_complement",
                    "action_complement",
                    "manner",
                    "cause",
                    "location",
                    "topic",
                ],
            },
            "lemma": {
                "type": "string",
                "description": (
                    "One Russian dictionary lemma for this argument; "
                    "never a phrase or inflected surface."
                ),
            },
            "part_of_speech": {
                "type": "string",
                "enum": [
                    "noun",
                    "verb",
                    "adjective",
                    "adverb",
                    "pronoun",
                    "numeral",
                ],
            },
            "grammatical_case": {
                "type": ["string", "null"],
                "enum": [
                    "nominative",
                    "genitive",
                    "dative",
                    "accusative",
                    "instrumental",
                    "prepositional",
                    None,
                ],
            },
            "preposition": {
                "type": ["string", "null"],
            },
            "modifiers": string_array,
        },
        "required": [
            "role",
            "lemma",
            "part_of_speech",
            "grammatical_case",
            "preposition",
            "modifiers",
        ],
    }
    response_proposition = {
        "type": "object",
        "properties": {
            "predicate": {
                "type": "string",
                "description": (
                    "Exactly one Russian verb lemma in infinitive form, "
                    "for example видеть, знать, считать, уточнить. "
                    "Never use a noun or repeat an argument lemma here."
                ),
            },
            "predicate_features": {
                "type": "object",
                "properties": {
                    "tense": {
                        "type": "string",
                        "enum": ["present", "past", "future"],
                    },
                    "person": {
                        "type": "string",
                        "enum": ["first", "second", "third"],
                    },
                    "number": {
                        "type": "string",
                        "enum": ["singular", "plural"],
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["masculine", "feminine", "neuter"],
                    },
                },
            },
            "arguments": {
                "type": "array",
                "items": response_argument,
                "minItems": 1,
            },
            "clause_kind": {
                "type": "string",
                "enum": ["statement", "question", "command"],
            },
            "negated": {"type": "boolean"},
            "response_polarity": {
                "type": ["boolean", "null"],
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1 if planning else 0,
            },
        },
        "required": [
            "predicate",
            "predicate_features",
            "arguments",
            "clause_kind",
            "negated",
            "response_polarity",
            "evidence_ids",
        ],
    }
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
            "response_plans": {
                "type": "array",
                "minItems": 1 if planning else 0,
                "maxItems": 3 if planning else 0,
                "items": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "string",
                            "enum": [
                                "answer",
                                "acknowledge",
                                "repair",
                                "decide",
                                "engage",
                                "clarify",
                            ],
                        },
                        "stance": {
                            "type": "string",
                            "enum": [
                                "direct",
                                "warm",
                                "playful",
                                "skeptical",
                                "curious",
                                "neutral",
                            ],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "propositions": {
                            "type": "array",
                            "items": response_proposition,
                            "minItems": 1,
                            "maxItems": 4,
                        },
                    },
                    "required": [
                        "move",
                        "stance",
                        "propositions",
                        "confidence",
                    ],
                },
            },
            "register": {"type": "string"},
            "facts": string_array,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["interpretation", "intent", "response_plans", "confidence"],
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


def _response_plans(value: object) -> tuple[ConsultantResponsePlan, ...]:
    allowed_roles = {
        "subject",
        "direct_object",
        "indirect_object",
        "predicate_complement",
        "action_complement",
        "manner",
        "cause",
        "location",
        "topic",
    }
    allowed_clause_kinds = {"statement", "question", "command"}
    plans: list[ConsultantResponsePlan] = []
    for raw_plan in _mapping_items(value, 6):
        move = _clean_token(raw_plan.get("move"), "")
        if not move:
            continue
        propositions: list[ConsultantResponseProposition] = []
        for raw_proposition in _mapping_items(
            raw_plan.get("propositions"),
            4,
        ):
            predicate = _single_lemma(raw_proposition.get("predicate"))
            if not predicate:
                continue
            arguments: list[ConsultantResponseArgument] = []
            for raw_argument in _mapping_items(
                raw_proposition.get("arguments"),
                10,
            ):
                role = _clean_token(raw_argument.get("role"), "")
                lemma = _single_lemma(raw_argument.get("lemma"))
                if role not in allowed_roles or not lemma:
                    continue
                arguments.append(
                    ConsultantResponseArgument(
                        role=role,
                        lemma=lemma,
                        part_of_speech=_clean_token(
                            raw_argument.get("part_of_speech"),
                            "noun",
                        ),
                        grammatical_case=_optional_token(raw_argument.get("grammatical_case")),
                        preposition=_optional_token(raw_argument.get("preposition")),
                        modifiers=tuple(
                            lemma
                            for item in _text_items(
                                raw_argument.get("modifiers"),
                                6,
                                80,
                            )
                            if (lemma := _single_lemma(item))
                        ),
                    )
                )
            clause_kind = _clean_token(
                raw_proposition.get("clause_kind"),
                "statement",
            )
            if clause_kind not in allowed_clause_kinds:
                clause_kind = "statement"
            raw_polarity = raw_proposition.get("response_polarity")
            response_polarity = raw_polarity if isinstance(raw_polarity, bool) else None
            propositions.append(
                ConsultantResponseProposition(
                    predicate=predicate,
                    predicate_features=_clean_features(raw_proposition.get("predicate_features")),
                    arguments=tuple(arguments),
                    clause_kind=clause_kind,
                    negated=raw_proposition.get("negated") is True,
                    response_polarity=response_polarity,
                    evidence_ids=tuple(
                        _clean_token(item, "")
                        for item in _text_items(
                            raw_proposition.get("evidence_ids"),
                            12,
                            120,
                        )
                        if _clean_token(item, "")
                    ),
                )
            )
        if not propositions:
            continue
        plans.append(
            ConsultantResponsePlan(
                move=move,
                stance=_clean_token(raw_plan.get("stance"), "engage"),
                propositions=tuple(propositions),
                confidence=_safe_number(raw_plan.get("confidence"), 0.5),
            )
        )
    return tuple(plans)


def _single_lemma(value: object) -> str:
    cleaned = _clean_token(value, "")
    if (
        not cleaned
        or len(cleaned) > 64
        or any(character.isspace() for character in cleaned)
        or internal_surface_reason(cleaned) is not None
    ):
        return ""
    return cleaned


def _optional_token(value: object) -> str | None:
    cleaned = _clean_token(value, "")
    return cleaned or None


def _stored_result(value: object) -> ConsultationResult | None:
    try:
        payload = json.loads(str(value))
        result = ConsultationResult.from_payload(
            payload,
            provider=_clean_token(payload.get("provider"), "unknown"),
            model=_clean_token(payload.get("model"), "unknown"),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return ConsultationResult(
        **{name: getattr(result, name) for name in result.__dataclass_fields__ if name != "cached"},
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
        dict.fromkeys(cleaned for item in value[:limit] if (cleaned := _clean_text(item, maximum)))
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
