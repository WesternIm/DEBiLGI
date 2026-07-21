"""Consolidate external semantic analysis into durable, provenance-aware lessons."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.language.construction_nlg import ConstructionSpec
from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, MorphologicalReading
from debilgi.language.productive_nlg import (
    DependencyPattern,
    dependency_pattern_id,
    observe_dependency_patterns,
    productive_pattern_curriculum,
)
from debilgi.language.semantic_consultant import ConsultationRequest, ConsultationResult
from debilgi.language.social_reasoning import (
    SOCIAL_POLICY_CURRICULUM,
    SocialPolicy,
)
from debilgi.language.structured_utterance import StructuredUtterance
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.semantic_assimilation")

_ENTITY_POS = {
    "person": "noun",
    "nickname": "noun",
    "object": "noun",
    "place": "noun",
    "work": "noun",
    "abstract": "noun",
    "action": "verb",
}


@dataclass(frozen=True, slots=True)
class SemanticLessonGuidance:
    facts: tuple[str, ...] = ()
    syntax_patterns: tuple[tuple[str, ...], ...] = ()
    constructions: tuple[ConstructionSpec, ...] = ()
    dependency_patterns: tuple[DependencyPattern, ...] = ()
    expected_response_acts: tuple[str, ...] = ()
    registers: tuple[str, ...] = ()
    social_policies: tuple[SocialPolicy, ...] = ()


@dataclass(frozen=True, slots=True)
class _Lesson:
    kind: str
    key: str
    payload: dict[str, Any]
    confidence: float
    local_support: bool = False


@dataclass(frozen=True, slots=True)
class _LessonUpdate:
    lesson_id: str
    status: str
    evidence_count: int
    evidence_added: bool
    activated_now: bool


class SemanticLessonRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def observe(
        self,
        *,
        guild_id: int,
        input_hash: str,
        lesson: _Lesson,
        observed_at: datetime,
    ) -> _LessonUpdate:
        lesson_id = f"semantic-lesson:{uuid4()}"
        now = observed_at.isoformat()
        payload = json.dumps(lesson.payload, ensure_ascii=False, sort_keys=True)
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_semantic_lessons (
                       lesson_id, guild_id, lesson_kind, lesson_key, payload_json,
                       confidence, first_seen_at, last_seen_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (guild_id, lesson_kind, lesson_key) DO UPDATE SET
                       payload_json = excluded.payload_json,
                       confidence = MAX(confidence, excluded.confidence),
                       last_seen_at = excluded.last_seen_at""",
                (
                    lesson_id,
                    guild_id,
                    lesson.kind,
                    lesson.key,
                    payload,
                    _clamp(lesson.confidence),
                    now,
                    now,
                ),
            )
            row = await (
                await connection.execute(
                    """SELECT lesson_id, status, evidence_count, confidence
                       FROM language_semantic_lessons
                       WHERE guild_id = ? AND lesson_kind = ? AND lesson_key = ?""",
                    (guild_id, lesson.kind, lesson.key),
                )
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to store semantic lesson")
            stored_id = str(row["lesson_id"])
            cursor = await connection.execute(
                """INSERT OR IGNORE INTO language_semantic_lesson_evidence (
                       lesson_id, input_hash, observed_at
                   ) VALUES (?, ?, ?)""",
                (stored_id, input_hash, now),
            )
            evidence_added = cursor.rowcount > 0
            evidence_count = int(row["evidence_count"])
            if evidence_added:
                evidence_count += 1
                await connection.execute(
                    """UPDATE language_semantic_lessons
                       SET evidence_count = ?, last_seen_at = ? WHERE lesson_id = ?""",
                    (evidence_count, now, stored_id),
                )
            previous_status = str(row["status"])
            confidence = max(float(row["confidence"]), lesson.confidence)
            should_activate = (
                previous_status != "rejected"
                and confidence >= 0.64
                and (lesson.local_support or evidence_count >= 2)
            )
            status = "active" if should_activate else previous_status
            activated_now = status == "active" and previous_status != "active"
            if activated_now:
                await connection.execute(
                    """UPDATE language_semantic_lessons
                       SET status = 'active', activated_at = ?, last_seen_at = ?
                       WHERE lesson_id = ?""",
                    (now, now, stored_id),
                )
        return _LessonUpdate(
            lesson_id=stored_id,
            status=status,
            evidence_count=evidence_count,
            evidence_added=evidence_added,
            activated_now=activated_now,
        )

    async def guidance_for_prompt(
        self,
        guild_id: int,
        prompt: str,
        *,
        limit: int = 40,
    ) -> SemanticLessonGuidance:
        rows = await self._database.fetchall(
            """SELECT lesson_kind, payload_json, confidence
               FROM language_semantic_lessons
               WHERE guild_id = ? AND status = 'active'
               ORDER BY confidence DESC, evidence_count DESC, last_seen_at DESC
               LIMIT ?""",
            (guild_id, max(1, min(limit, 100))),
        )
        prompt_terms = set(content_lemmas(prompt))
        facts: list[str] = []
        patterns: list[tuple[str, ...]] = []
        constructions: list[ConstructionSpec] = []
        dependency_patterns: list[DependencyPattern] = []
        response_acts: list[str] = []
        registers: list[str] = []
        social_policies: list[SocialPolicy] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            kind = str(row["lesson_kind"])
            lesson_terms = set(_payload_terms(payload))
            relevant = not prompt_terms or bool(prompt_terms & lesson_terms)
            if kind == "lexical" and relevant:
                for fact in payload.get("facts", ()):
                    if (
                        isinstance(fact, str)
                        and fact.strip()
                        and internal_surface_reason(fact) is None
                        and fact not in facts
                    ):
                        facts.append(fact.strip())
            elif kind == "syntax":
                pattern = tuple(
                    str(item).casefold()
                    for item in payload.get("pos_pattern", ())
                    if isinstance(item, str) and item
                )
                if pattern and pattern not in patterns:
                    patterns.append(pattern)
                dependency_roles = _dependency_roles(
                    payload.get("dependency_roles", payload.get("roles", ()))
                )
                dependency_pattern = DependencyPattern(
                    roles=dependency_roles,
                    clause_kind=str(payload.get("clause_kind", "any")),
                    confidence=float(row["confidence"]),
                    source="semantic_lesson",
                )
                if (
                    dependency_pattern.valid
                    and dependency_pattern not in dependency_patterns
                ):
                    dependency_patterns.append(dependency_pattern)
            elif kind == "grammar" and isinstance(payload.get("construction"), dict):
                spec = ConstructionSpec.from_mapping(
                    payload["construction"],
                    source="learned",
                    confidence=float(row["confidence"]),
                )
                if spec is not None and spec not in constructions:
                    constructions.append(spec)
            elif kind == "dialogue":
                for act in payload.get("expected_response_acts", ()):
                    if isinstance(act, str) and act and act not in response_acts:
                        response_acts.append(act)
                register = payload.get("register")
                if isinstance(register, str) and register and register not in registers:
                    registers.append(register)
                policy = SocialPolicy.from_payload(payload.get("social_policy"))
                if policy is not None and policy not in social_policies:
                    social_policies.append(policy)
        productive_rows = await self._database.fetchall(
            """SELECT * FROM language_productive_constructions
               WHERE guild_id = ? AND confidence >= 0.40 AND utility > -0.75
               ORDER BY utility DESC, confidence DESC, observation_count DESC
               LIMIT 32""",
            (guild_id,),
        )
        for row in productive_rows:
            try:
                learned_pattern = DependencyPattern(
                    roles=tuple(json.loads(str(row["roles_json"]))),
                    clause_kind=str(row["clause_kind"]),
                    confidence=float(row["confidence"]),
                    evidence_count=int(row["observation_count"]),
                    source=str(row["source"]),
                    construction_id=str(row["construction_id"]),
                    predicate_class=str(row["predicate_class"]),
                    required_roles=tuple(
                        json.loads(str(row["required_roles_json"]))
                    ),
                    optional_roles=tuple(
                        json.loads(str(row["optional_roles_json"]))
                    ),
                    focus_role=(
                        str(row["focus_role"])
                        if row["focus_role"] is not None
                        else None
                    ),
                    negation_mode=str(row["negation_mode"]),
                    utility=float(row["utility"]),
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if learned_pattern.valid and learned_pattern not in dependency_patterns:
                dependency_patterns.append(learned_pattern)
        return SemanticLessonGuidance(
            facts=tuple(facts[:6]),
            syntax_patterns=tuple(patterns[:12]),
            constructions=tuple(constructions[:12]),
            dependency_patterns=tuple(dependency_patterns[:16]),
            expected_response_acts=tuple(response_acts[:8]),
            registers=tuple(registers[:4]),
            social_policies=tuple(social_policies[:24]),
        )


class SemanticAssimilationService:
    """Persist structured lessons and apply corroborated ones to language memory."""

    def __init__(self, database: Database, dataset: SelfDatasetRepository) -> None:
        self._database = database
        self._dataset = dataset
        self._lessons = SemanticLessonRepository(database)

    async def assimilate(
        self,
        *,
        guild_id: int,
        request: ConsultationRequest,
        result: ConsultationResult,
        observed_at: datetime | None = None,
        source_input_hash: str | None = None,
    ) -> int:
        now = observed_at or datetime.now(UTC)
        evidence_hash = source_input_hash or request.input_hash
        lessons = _derive_lessons(request, result)
        active = 0
        for lesson in lessons:
            update = await self._lessons.observe(
                guild_id=guild_id,
                input_hash=evidence_hash,
                lesson=lesson,
                observed_at=now,
            )
            if update.status != "active" or not update.evidence_added:
                continue
            active += 1
            await self._apply_active(guild_id, lesson, now)
        if active:
            await self._database.execute(
                """UPDATE language_semantic_consultations
                   SET status = 'confirmed', updated_at = ?
                   WHERE guild_id = ? AND input_hash = ?""",
                (now.isoformat(), guild_id, evidence_hash),
            )
        logger.info(
            "guild=%d input=%s lessons=%d active=%d",
            guild_id,
            evidence_hash[:12],
            len(lessons),
            active,
        )
        return active

    async def observe_local_structure(
        self,
        *,
        guild_id: int,
        structure: StructuredUtterance,
        observed_at: datetime | None = None,
    ) -> int:
        """Learn grammar from local parsing without calling an external model."""

        now = observed_at or datetime.now(UTC)
        observed = 0
        for pattern in observe_dependency_patterns(structure):
            lesson = _Lesson(
                "syntax",
                _lesson_key(
                    "dependency",
                    pattern.clause_kind,
                    pattern.predicate_class,
                    pattern.negation_mode,
                    *pattern.roles,
                ),
                {
                    "roles": pattern.roles,
                    "dependency_roles": pattern.roles,
                    "clause_kind": pattern.clause_kind,
                    "predicate_class": pattern.predicate_class,
                    "required_roles": pattern.required_roles,
                    "optional_roles": pattern.optional_roles,
                    "focus_role": pattern.focus_role,
                    "negation_mode": pattern.negation_mode,
                    "construction_id": pattern.construction_id,
                    "pos_pattern": (),
                    "cases": (),
                    "terms": tuple(content_lemmas(structure.text)),
                },
                min(0.92, pattern.confidence),
                True,
            )
            input_hash = hashlib.sha256(
                f"local|{structure.text}|{'|'.join(pattern.roles)}".encode()
            ).hexdigest()
            update = await self._lessons.observe(
                guild_id=guild_id,
                input_hash=input_hash,
                lesson=lesson,
                observed_at=now,
            )
            if update.status == "active" and update.evidence_added:
                await self._apply_active(guild_id, lesson, now)
                observed += 1
        return observed

    async def backfill(self, *, limit: int = 100) -> int:
        """Extract durable lessons from older lexical consultations once."""

        rows = await self._database.fetchall(
            """SELECT input_hash, guild_id, subject_terms_json, reason_codes_json,
                      local_confidence, result_json, updated_at
               FROM language_semantic_consultations AS consultation
               WHERE mode = 'lexical' AND status = 'provisional'
                 AND NOT EXISTS (
                     SELECT 1 FROM language_semantic_lesson_evidence AS evidence
                     WHERE evidence.input_hash = consultation.input_hash
                 )
               ORDER BY updated_at ASC LIMIT ?""",
            (max(1, min(limit, 500)),),
        )
        processed = 0
        for row in rows:
            try:
                payload = json.loads(str(row["result_json"]))
                terms = tuple(
                    str(item)
                    for item in json.loads(str(row["subject_terms_json"]))
                    if isinstance(item, str) and item.strip()
                )
                reasons = tuple(
                    str(item)
                    for item in json.loads(str(row["reason_codes_json"]))
                    if isinstance(item, str) and item.strip()
                )
                result = ConsultationResult.from_payload(
                    payload,
                    provider=str(payload.get("provider", "puter")),
                    model=str(payload.get("model", "unknown")),
                )
                updated_at = datetime.fromisoformat(str(row["updated_at"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not terms:
                continue
            word = terms[0]
            request = ConsultationRequest(
                prompt=f"Что означает слово «{word}»?",
                context_messages=(),
                known_terms=(),
                unknown_terms=(word,),
                local_relations=(),
                local_confidence=float(row["local_confidence"]),
                reason_codes=(*reasons, "semantic_lesson_backfill"),
                mode="lexical",
            )
            await self.assimilate(
                guild_id=int(row["guild_id"]),
                request=request,
                result=result,
                observed_at=updated_at,
                source_input_hash=str(row["input_hash"]),
            )
            processed += 1
        return processed

    async def backfill_productive_constructions(self, *, limit: int = 500) -> int:
        """Seed the new construction memory from already confirmed syntax lessons."""

        rows = await self._database.fetchall(
            """SELECT guild_id, payload_json, confidence, evidence_count, activated_at,
                      last_seen_at
               FROM language_semantic_lessons
               WHERE lesson_kind = 'syntax' AND status = 'active'
               ORDER BY evidence_count DESC, confidence DESC
               LIMIT ?""",
            (max(1, min(limit, 2000)),),
        )
        inserted = 0
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            roles = _dependency_roles(
                payload.get("dependency_roles", payload.get("roles", ()))
            )
            pattern = DependencyPattern(
                roles=roles,
                clause_kind=str(payload.get("clause_kind", "any")),
                confidence=float(row["confidence"]),
                evidence_count=max(1, int(row["evidence_count"])),
                source="backfilled_semantic_lesson",
                predicate_class=str(payload.get("predicate_class", "any")),
                required_roles=_dependency_roles(payload.get("required_roles", ())),
                optional_roles=_dependency_roles(payload.get("optional_roles", ())),
                focus_role=(
                    str(payload["focus_role"])
                    if payload.get("focus_role")
                    else None
                ),
                negation_mode=str(payload.get("negation_mode", "any")),
            )
            if not pattern.valid:
                continue
            construction_id = dependency_pattern_id(pattern)
            observed_at = str(row["activated_at"] or row["last_seen_at"])
            cursor = await self._database.execute(
                """INSERT OR IGNORE INTO language_productive_constructions (
                       guild_id, construction_id, clause_kind, predicate_class,
                       roles_json, required_roles_json, optional_roles_json,
                       focus_role, negation_mode, observation_count, confidence,
                       source, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(row["guild_id"]),
                    construction_id,
                    pattern.clause_kind,
                    pattern.predicate_class,
                    json.dumps(pattern.roles, ensure_ascii=False),
                    json.dumps(pattern.required_roles, ensure_ascii=False),
                    json.dumps(pattern.optional_roles, ensure_ascii=False),
                    pattern.focus_role,
                    pattern.negation_mode,
                    pattern.evidence_count,
                    _clamp(pattern.confidence),
                    pattern.source,
                    observed_at,
                    observed_at,
                ),
            )
            inserted += max(0, cursor.rowcount)
        return inserted

    async def guidance_for_prompt(
        self, guild_id: int, prompt: str
    ) -> SemanticLessonGuidance:
        return await self._lessons.guidance_for_prompt(guild_id, prompt)

    async def install_social_curriculum(
        self,
        *,
        guild_id: int,
        observed_at: datetime | None = None,
    ) -> int:
        """Install word-free contrastive social policies as active lessons."""

        now = observed_at or datetime.now(UTC)
        installed = 0
        for policy in SOCIAL_POLICY_CURRICULUM:
            lesson = _Lesson(
                "dialogue",
                _lesson_key("social_policy", policy.policy_id),
                {
                    "intent": policy.prompt_acts[0] if policy.prompt_acts else "unknown",
                    "expected_response_acts": (
                        policy.primary_move,
                        *policy.alternative_moves,
                    ),
                    "register": policy.registers[0] if policy.registers else "any",
                    "terms": (),
                    "social_policy": policy.as_payload(),
                },
                policy.confidence,
                True,
            )
            input_hash = hashlib.sha256(
                f"social-policy-curriculum-v1|{policy.policy_id}".encode()
            ).hexdigest()
            update = await self._lessons.observe(
                guild_id=guild_id,
                input_hash=input_hash,
                lesson=lesson,
                observed_at=now,
            )
            if update.status == "active":
                installed += 1
                if update.evidence_added:
                    await self._apply_active(guild_id, lesson, now)
        return installed

    async def install_productive_curriculum(
        self,
        *,
        guild_id: int,
        observed_at: datetime | None = None,
    ) -> int:
        """Install safe grammatical priors without storing authored wording."""

        now = observed_at or datetime.now(UTC)
        installed = 0
        for pattern in productive_pattern_curriculum():
            before = await self._database.fetchone(
                """SELECT 1 FROM language_productive_constructions
                   WHERE guild_id = ? AND construction_id = ?""",
                (guild_id, pattern.construction_id),
            )
            if before is not None:
                continue
            await self._store_productive_pattern(
                guild_id=guild_id,
                pattern=pattern,
                observed_at=now,
            )
            installed += 1
        return installed

    async def _apply_active(
        self, guild_id: int, lesson: _Lesson, observed_at: datetime
    ) -> None:
        if lesson.kind == "lexical":
            word = str(lesson.payload.get("word", ""))
            facts = tuple(
                item
                for item in lesson.payload.get("facts", ())
                if isinstance(item, str) and internal_surface_reason(item) is None
            )
            concepts = tuple(
                dict.fromkeys(
                    term
                    for fact in facts
                    for term in content_lemmas(fact)
                    if term != word
                )
            )[:16]
            if word and concepts:
                await self._dataset.link_reference_concepts(
                    guild_id=guild_id,
                    word=word,
                    concepts=concepts,
                    source="semantic_consultant",
                    observed_at=observed_at,
                    confidence=min(0.72, lesson.confidence),
                )
        elif lesson.kind == "relation":
            source = str(lesson.payload.get("source", ""))
            relation = str(lesson.payload.get("relation", "associated_with"))
            target = str(lesson.payload.get("target", ""))
            if source and target:
                await self._dataset.link_reference_concepts(
                    guild_id=guild_id,
                    word=source,
                    concepts=(target,),
                    relations=((source, relation, target, min(0.70, lesson.confidence)),),
                    source="semantic_consultant",
                    observed_at=observed_at,
                    confidence=min(0.70, lesson.confidence),
                )
        elif lesson.kind == "morphology":
            features = dict(lesson.payload.get("features", {}))
            features["provenance"] = "semantic_consultant"
            await self._database.execute(
                """INSERT INTO language_morphology_observations (
                       guild_id, surface_form, lemma, part_of_speech,
                       grammatical_features_json, confidence, first_seen_at, last_seen_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (
                       guild_id, surface_form, lemma, grammatical_features_json
                   ) DO UPDATE SET
                       occurrence_count = occurrence_count + 1,
                       confidence = MAX(confidence, excluded.confidence),
                       last_seen_at = excluded.last_seen_at""",
                (
                    guild_id,
                    lesson.payload.get("surface", ""),
                    lesson.payload.get("lemma", ""),
                    lesson.payload.get("part_of_speech"),
                    json.dumps(features, ensure_ascii=False, sort_keys=True),
                    min(0.74, lesson.confidence),
                    observed_at.isoformat(),
                    observed_at.isoformat(),
                ),
            )
        elif lesson.kind == "syntax":
            roles = tuple(lesson.payload.get("roles", ()))
            cases = tuple(lesson.payload.get("cases", ()))
            if roles:
                pattern_key = hashlib.sha256("|".join(roles).encode("utf-8")).hexdigest()[:24]
                await self._database.execute(
                    """INSERT INTO language_grammar_patterns (
                           guild_id, pattern_key, roles_json, cases_json,
                           confidence, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT (guild_id, pattern_key) DO UPDATE SET
                           occurrence_count = occurrence_count + 1,
                           confidence = MAX(confidence, excluded.confidence),
                           updated_at = excluded.updated_at""",
                    (
                        guild_id,
                        pattern_key,
                        json.dumps(roles, ensure_ascii=False),
                        json.dumps(cases, ensure_ascii=False),
                        min(0.76, lesson.confidence),
                        observed_at.isoformat(),
                    ),
                )
            dependency_roles = _dependency_roles(
                lesson.payload.get("dependency_roles", roles)
            )
            pattern = DependencyPattern(
                roles=dependency_roles,
                clause_kind=str(lesson.payload.get("clause_kind", "any")),
                confidence=min(0.92, lesson.confidence),
                source=(
                    "local_observation"
                    if lesson.local_support
                    else "semantic_teacher"
                ),
                construction_id=(
                    str(lesson.payload["construction_id"])
                    if lesson.payload.get("construction_id")
                    else None
                ),
                predicate_class=str(
                    lesson.payload.get("predicate_class", "any")
                ),
                required_roles=_dependency_roles(
                    lesson.payload.get("required_roles", ())
                ),
                optional_roles=_dependency_roles(
                    lesson.payload.get("optional_roles", ())
                ),
                focus_role=(
                    str(lesson.payload["focus_role"])
                    if lesson.payload.get("focus_role")
                    else None
                ),
                negation_mode=str(lesson.payload.get("negation_mode", "any")),
            )
            if pattern.valid:
                await self._store_productive_pattern(
                    guild_id=guild_id,
                    pattern=pattern,
                    observed_at=observed_at,
                )
        elif lesson.kind == "dialogue":
            policy = SocialPolicy.from_payload(lesson.payload.get("social_policy"))
            prompt_acts = (
                policy.prompt_acts
                if policy is not None
                else (str(lesson.payload.get("intent", "unknown")),)
            )
            register = str(lesson.payload.get("register", "neutral"))
            source = (
                "social_curriculum"
                if policy and policy.source == "curriculum"
                else "semantic_consultant"
            )
            for prompt_act in prompt_acts:
                for response_act in lesson.payload.get("expected_response_acts", ()):
                    if not isinstance(response_act, str) or not response_act:
                        continue
                    await self._database.execute(
                        """INSERT INTO language_dialogue_act_patterns (
                               guild_id, prompt_act, response_act, register_label,
                               confidence, source, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT (
                               guild_id, prompt_act, response_act, register_label, source
                           ) DO UPDATE SET
                               evidence_count = evidence_count + 1,
                               confidence = MAX(confidence, excluded.confidence),
                               updated_at = excluded.updated_at""",
                        (
                            guild_id,
                            prompt_act,
                            response_act,
                            register,
                            min(0.86 if policy else 0.74, lesson.confidence),
                            source,
                            observed_at.isoformat(),
                        ),
                    )

    async def _store_productive_pattern(
        self,
        *,
        guild_id: int,
        pattern: DependencyPattern,
        observed_at: datetime,
    ) -> None:
        """Store only abstract slots and constraints, never an authored sentence."""

        construction_id = pattern.construction_id or dependency_pattern_id(pattern)
        await self._database.execute(
            """INSERT INTO language_productive_constructions (
                   guild_id, construction_id, clause_kind, predicate_class,
                   roles_json, required_roles_json, optional_roles_json,
                   focus_role, negation_mode, confidence, source,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (guild_id, construction_id) DO UPDATE SET
                   observation_count = observation_count + 1,
                   confidence = MIN(
                       0.98,
                       MAX(confidence, excluded.confidence) + 0.01
                   ),
                   updated_at = excluded.updated_at""",
            (
                guild_id,
                construction_id,
                pattern.clause_kind,
                pattern.predicate_class,
                json.dumps(pattern.roles, ensure_ascii=False),
                json.dumps(pattern.required_roles, ensure_ascii=False),
                json.dumps(pattern.optional_roles, ensure_ascii=False),
                pattern.focus_role,
                pattern.negation_mode,
                _clamp(pattern.confidence),
                pattern.source,
                observed_at.isoformat(),
                observed_at.isoformat(),
            ),
        )


def _derive_lessons(
    request: ConsultationRequest, result: ConsultationResult
) -> tuple[_Lesson, ...]:
    analysis = DEFAULT_MORPHOLOGY.analyze_sentence(request.prompt)
    local_readings = tuple(item for item in analysis.readings if item is not None)
    local_relations = {
        (_norm(source), _norm(relation), _norm(target))
        for source, relation, target in request.local_relations
    }
    lessons: list[_Lesson] = []
    base_confidence = min(0.86, result.confidence)

    for word in request.unknown_terms:
        normalized = _norm(word)
        facts = tuple(dict.fromkeys((*result.facts, result.interpretation)))
        if normalized and facts:
            lessons.append(
                _Lesson(
                    "lexical",
                    normalized,
                    {"word": normalized, "facts": facts},
                    base_confidence * 0.82,
                )
            )

    for entity in result.entities:
        lemma = _norm(entity.lemma or entity.surface)
        expected_pos = _ENTITY_POS.get(entity.entity_type)
        local_support = any(
            reading.lemma == lemma
            and (expected_pos is None or reading.part_of_speech == expected_pos)
            for reading in local_readings
        )
        if lemma:
            lessons.append(
                _Lesson(
                    "entity",
                    f"{lemma}:{entity.entity_type}",
                    {
                        "surface": entity.surface,
                        "lemma": lemma,
                        "entity_type": entity.entity_type,
                        "part_of_speech": expected_pos,
                    },
                    base_confidence * 0.88,
                    local_support,
                )
            )

    for relation in result.relations:
        source, rel, target = (
            _norm(relation.source),
            _norm(relation.relation),
            _norm(relation.target),
        )
        if source and target:
            lessons.append(
                _Lesson(
                    "relation",
                    f"{source}:{rel}:{target}",
                    {"source": source, "relation": rel, "target": target},
                    base_confidence * 0.78,
                    (source, rel, target) in local_relations,
                )
            )

    for ambiguity in result.ambiguities:
        surface, selected = _norm(ambiguity.surface), _norm(ambiguity.selected)
        if surface and selected:
            lessons.append(
                _Lesson(
                    "ambiguity",
                    f"{surface}:{selected}",
                    {
                        "surface": surface,
                        "selected": selected,
                        "candidates": ambiguity.candidates,
                        "reason": ambiguity.reason,
                    },
                    base_confidence * 0.72,
                )
            )

    aligned = 0
    pos_pattern: list[str] = []
    roles: list[str] = []
    cases: list[str] = []
    for token in result.tokens:
        reading = _matching_reading(token.surface, local_readings)
        local_support = bool(
            reading
            and reading.lemma == token.lemma
            and (
                token.part_of_speech == "unknown"
                or reading.part_of_speech == token.part_of_speech
            )
        )
        if local_support:
            aligned += 1
        if token.part_of_speech != "unknown":
            pos_pattern.append(token.part_of_speech)
        roles.append(token.role)
        features = dict(token.features)
        case = features.get("case")
        if case:
            cases.append(case)
        if token.lemma:
            lessons.append(
                _Lesson(
                    "morphology",
                    _lesson_key(
                        token.surface,
                        token.lemma,
                        token.part_of_speech,
                        json.dumps(features, sort_keys=True),
                    ),
                    {
                        "surface": token.surface,
                        "lemma": token.lemma,
                        "part_of_speech": token.part_of_speech,
                        "role": token.role,
                        "features": features,
                    },
                    base_confidence * 0.86,
                    local_support,
                )
            )
    token_alignment = aligned / max(1, len(result.tokens))
    sentence_roles = tuple(result.sentence_pattern or tuple(roles))
    dependency_roles = _dependency_roles(sentence_roles)
    if sentence_roles and pos_pattern:
        lessons.append(
            _Lesson(
                "syntax",
                _lesson_key(*sentence_roles, "pos", *pos_pattern),
                {
                    "roles": sentence_roles,
                    "dependency_roles": dependency_roles,
                    # The observation contains role order, not punctuation or
                    # a ready sentence, so it can compete in both statements
                    # and questions. Clause-specific evidence can be added
                    # later without lexicalizing the construction.
                    "clause_kind": "any",
                    "pos_pattern": tuple(pos_pattern),
                    "cases": tuple(cases),
                    "terms": tuple(content_lemmas(request.prompt)),
                },
                base_confidence * 0.82,
                token_alignment >= 0.70,
            )
        )

    for rule in result.grammar_rules:
        lessons.append(
            _Lesson(
                "grammar",
                _lesson_key(rule),
                {"rule": rule, "terms": tuple(content_lemmas(request.prompt))},
                base_confidence * 0.70,
            )
        )

    for construction in result.constructions:
        lessons.append(
            _Lesson(
                "grammar",
                _lesson_key(
                    "construction",
                    construction.response_act,
                    construction.relation,
                    construction.truth_state,
                    *construction.sequence,
                ),
                {
                    "construction": construction.as_dict(),
                    "terms": tuple(content_lemmas(request.prompt)),
                },
                base_confidence * 0.82,
                # External schemas must recur in independent consultations.
                # Structural validation is necessary, but is not by itself
                # evidence that a construction is idiomatic.
                False,
            )
        )

    if result.social_frame is not None:
        social = result.social_frame
        social_policy = SocialPolicy(
            policy_id=_lesson_key(
                "teacher_social",
                social.prompt_act,
                social.response_obligation,
                *social.recommended_moves,
            ),
            prompt_acts=(social.prompt_act,),
            response_obligations=(social.response_obligation,),
            registers=(result.register,),
            primary_move=social.recommended_moves[0],
            alternative_moves=social.recommended_moves[1:],
            forbidden_moves=social.avoid_moves,
            stance_mode=(
                "follow_evidence"
                if social.speaker_stance == "uncertain"
                else "own_position"
            ),
            answer_first=social.response_obligation in {"answer", "decide"},
            reciprocity=0.35 if social.response_obligation == "engage" else 0.0,
            warmth=social.warmth,
            playfulness=social.playfulness,
            truth_priority=1.0,
            confidence=base_confidence * 0.90,
            source="semantic_teacher",
        )
        lessons.append(
            _Lesson(
                "dialogue",
                _lesson_key("social_policy", social_policy.policy_id),
                {
                    "intent": social.prompt_act,
                    "expected_response_acts": social.recommended_moves,
                    "register": result.register,
                    "speaker_stance": social.speaker_stance,
                    "emotion": social.emotion,
                    "directness": social.directness,
                    "terms": (),
                    "social_policy": social_policy.as_payload(),
                },
                base_confidence * 0.90,
                _intent_matches_prompt(social.prompt_act, request.prompt),
            )
        )

    if result.intent != "unknown":
        dialogue_support = _intent_matches_prompt(result.intent, request.prompt)
        lessons.append(
            _Lesson(
                "dialogue",
                _lesson_key(
                    result.intent,
                    result.register,
                    *result.expected_response_acts,
                ),
                {
                    "intent": result.intent,
                    "expected_response_acts": result.expected_response_acts,
                    "register": result.register,
                    "terms": tuple(content_lemmas(request.prompt)),
                },
                base_confidence * 0.84,
                dialogue_support and bool(result.expected_response_acts),
            )
        )
    if result.register != "neutral":
        lessons.append(
            _Lesson(
                "style",
                _lesson_key(result.register, result.intent),
                {"register": result.register, "intent": result.intent},
                base_confidence * 0.68,
            )
        )
    return tuple(lessons)


def _matching_reading(
    surface: str, readings: tuple[MorphologicalReading, ...]
) -> MorphologicalReading | None:
    normalized = _norm(surface)
    return next(
        (
            reading
            for reading in readings
            if _norm(reading.surface_form) == normalized
            or _norm(reading.corrected_form or "") == normalized
        ),
        None,
    )


def _intent_matches_prompt(intent: str, prompt: str) -> bool:
    lowered = intent.casefold()
    if prompt.rstrip().endswith("?"):
        return any(marker in lowered for marker in ("question", "ask", "request", "query"))
    return not any(marker in lowered for marker in ("question", "ask", "query"))


def _payload_terms(payload: dict[str, Any]) -> tuple[str, ...]:
    parts: list[str] = []
    for key in ("word", "surface", "lemma", "source", "target"):
        value = payload.get(key)
        if isinstance(value, str):
            parts.append(value)
    terms = payload.get("terms")
    if isinstance(terms, list):
        parts.extend(str(item) for item in terms if isinstance(item, str))
    facts = payload.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, str):
                parts.extend(content_lemmas(fact))
    return tuple(dict.fromkeys(_norm(item) for item in parts if _norm(item)))


def _dependency_roles(value: object) -> tuple[str, ...]:
    """Convert teacher labels to the local, word-free dependency vocabulary."""

    if not isinstance(value, (list, tuple)):
        return ()
    aliases = {
        "subject": "subject",
        "predicate": "predicate",
        "verb": "predicate",
        "object": "direct_object",
        "direct_object": "direct_object",
        "indirect_object": "indirect_object",
        "predicate_complement": "predicate_complement",
        "action_complement": "action_complement",
        "location": "location",
        "topic": "topic",
        "negation": "negation",
    }
    roles: list[str] = []
    for item in value:
        role = aliases.get(_norm(str(item)))
        if role is not None and role not in roles:
            roles.append(role)
    return tuple(roles)


def _lesson_key(*parts: str) -> str:
    material = "|".join(_norm(part) for part in parts if _norm(part))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _norm(value: str) -> str:
    return " ".join(str(value).casefold().replace("ё", "е").split()).strip()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
