"""Persistent dialogue deliberation over effects, evidence and unfinished work.

The task layer never maps an input phrase to an authored reply.  It records what
has to change in the conversation, keeps failed attempts open, and evaluates a
candidate by whether its meaning can advance that task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from debilgi.events.models import Event
from debilgi.language.content import content_lemmas
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

if TYPE_CHECKING:
    from debilgi.language.dialogue_state import DialogueStateAnalysis


_COMMUNICATION_ACTIONS = frozenset(
    {"сказать", "говорить", "повторить", "произнести", "написать", "назвать", "ответить"}
)
_QUESTION_WORDS = frozenset(
    {
        "что",
        "кто",
        "где",
        "куда",
        "откуда",
        "когда",
        "как",
        "какой",
        "почему",
        "зачем",
        "сколько",
        "насколько",
    }
)
_DISCOURSE_WORDS = frozenset(
    {"а", "и", "но", "ну", "так", "тогда", "же", "ли", "не", "то", "бы", "пожалуйста"}
)
_GENERIC_FOCUS = frozenset(
    {
        "быть",
        "делать",
        "говорить",
        "знать",
        "ответ",
        "писать",
        "сказать",
        "слово",
        "смысл",
        "этот",
    }
)
_AFFIRMATION = frozenset({"да", "ага", "верно", "точно", "ок", "ладно"})
_REJECTION = frozenset({"нет", "неверно", "неправильно", "ошибка", "ошибся", "ошиблась"})
_CONTACT_RESPONSE_ACTIONS = frozenset(
    {
        "отвечать",
        "ответить",
        "слушать",
        "слышать",
        "говорить",
        "сказать",
        "понимать",
    }
)
_UNRESOLVED_COGNITIVE_ACTIONS = frozenset(
    {"понимать", "знать", "разбирать", "распознавать", "схватывать"}
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalized_words(text: str) -> tuple[str, ...]:
    return tuple(
        token.casefold().replace("ё", "е")
        for token in tokenize(text)
        if token[:1].isalnum()
    )


def _contains_sequence(words: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not sequence or len(sequence) > len(words):
        return False
    return any(
        words[index : index + len(sequence)] == sequence
        for index in range(len(words) - len(sequence) + 1)
    )


def contact_response_binding(text: str) -> float:
    """Measure whether wording itself takes the conversational turn.

    A completed sentence from an unrelated article is not an answer to an
    attention call.  The candidate must express an agent-side communicative
    action; the check is semantic and does not prescribe a ready phrase.
    """

    structure = analyze_structured_utterance(text)
    clause = _main_clause(structure)
    if (
        clause is None
        or clause.clause_kind in {"question", "command"}
        or len(structure.clauses) != 1
        or structure.clause_links
    ):
        return 0.0
    predicate = (
        structure.tokens[clause.predicate_index]
        if clause.predicate_index is not None
        else None
    )
    if predicate is None or predicate.lemma not in _CONTACT_RESPONSE_ACTIONS:
        return 0.0
    subjects = {structure.tokens[index].lemma for index in clause.subject_indices}
    features = dict(predicate.grammatical_features)
    agent_side = bool(subjects & {"я", "мы"}) or features.get("person") == "first"
    extra_content = set(content_lemmas(text)) - {
        predicate.lemma,
        "я",
        "мы",
        "ты",
        "вы",
    }
    return 1.0 if agent_side and not extra_content else 0.0


def unresolved_input_binding(text: str) -> float:
    """Recognize a first-person cognitive state caused by failed interpretation."""

    structure = analyze_structured_utterance(text)
    clause = _main_clause(structure)
    if (
        clause is None
        or clause.clause_kind in {"question", "command"}
        or len(structure.clauses) != 1
        or structure.clause_links
        or not clause.negated
    ):
        return 0.0
    predicate = (
        structure.tokens[clause.predicate_index]
        if clause.predicate_index is not None
        else None
    )
    if predicate is None or predicate.lemma not in _UNRESOLVED_COGNITIVE_ACTIONS:
        return 0.0
    subjects = {structure.tokens[index].lemma for index in clause.subject_indices}
    features = dict(predicate.grammatical_features)
    agent_side = bool(subjects & {"я", "мы"}) or features.get("person") == "first"
    extra_content = set(content_lemmas(text)) - {
        predicate.lemma,
        "я",
        "мы",
    }
    return 1.0 if agent_side and not extra_content else 0.0


@dataclass(frozen=True, slots=True)
class TaskObjective:
    speech_act: str
    predicate: str | None
    question_word: str | None
    focus_terms: tuple[str, ...]
    required_content: str | None = None
    required_roles: tuple[tuple[str, str], ...] = ()
    revisions: tuple[tuple[str, str], ...] = ()
    missing_slots: tuple[str, ...] = ()
    target_event_id: str | None = None
    required_response_act: str | None = None


@dataclass(frozen=True, slots=True)
class DialogueTask:
    task_id: str
    guild_id: int
    channel_id: int
    user_id: int
    source_event_id: str
    parent_task_id: str | None
    signature: str
    kind: str
    goal: str
    objective: TaskObjective
    priority: float
    confidence: float
    status: str
    recurrence_count: int
    attempt_count: int
    created_at: datetime
    updated_at: datetime

    @property
    def locally_verifiable(self) -> bool:
        return self.goal == "perform_utterance" and bool(self.objective.required_content)


@dataclass(frozen=True, slots=True)
class DialogueSituation:
    guild_id: int
    channel_id: int
    user_id: int
    active_task: DialogueTask | None
    open_tasks: tuple[DialogueTask, ...]
    revision: int
    updated_at: datetime

    @property
    def features(self) -> tuple[tuple[str, float], ...]:
        task = self.active_task
        if task is None:
            return ()
        return (
            ("task_priority", task.priority),
            ("task_confidence", task.confidence),
            ("task_recurrence", min(1.0, task.recurrence_count / 4.0)),
            ("task_attempts", min(1.0, task.attempt_count / 4.0)),
            ("task_is_repair", float(task.goal == "repair_shared_model")),
            ("task_is_directive", float(task.objective.speech_act == "command")),
        )


@dataclass(frozen=True, slots=True)
class TaskProgress:
    score: float
    reasons: tuple[str, ...]

    @property
    def advances(self) -> bool:
        return self.score >= 0.45


@dataclass(frozen=True, slots=True)
class _TaskDraft:
    kind: str
    goal: str
    objective: TaskObjective
    priority: float
    confidence: float

    @property
    def signature(self) -> str:
        semantic_identity = (
            self.goal,
            self.objective.predicate,
            self.objective.question_word,
            tuple(sorted(self.objective.focus_terms)),
            " ".join(_normalized_words(self.objective.required_content or "")),
            tuple(sorted(self.objective.required_roles)),
            tuple(sorted(self.objective.revisions)),
            self.objective.required_response_act,
        )
        return hashlib.sha256(_json(semantic_identity).encode()).hexdigest()[:24]


def infer_dialogue_task(
    text: str,
    *,
    source_event_id: str,
    dialogue_state: DialogueStateAnalysis | None = None,
    latest_agent_turn: str | None = None,
    latest_user_turn: str | None = None,
) -> _TaskDraft:
    """Infer an unresolved conversational effect from structure and local state."""

    structure = analyze_structured_utterance(text)
    clause = _request_clause(structure)
    forced_question = bool(text.rstrip().endswith("?"))
    predicate = (
        structure.tokens[clause.predicate_index].lemma
        if clause is not None and clause.predicate_index is not None
        else clause.implicit_predicate if clause is not None else None
    )
    if forced_question and (clause is None or clause.clause_kind != "question"):
        predicate = next(
            (
                token.lemma
                for token in reversed(structure.tokens)
                if token.part_of_speech == "verb"
            ),
            predicate,
        )
    question_word = _question_word(structure, clause)
    speech_act = (
        "question"
        if forced_question
        else clause.clause_kind
        if clause is not None
        else "fragment"
    )
    required_roles = _known_roles(structure, clause)
    turn_focus = set(content_lemmas(text)) - _QUESTION_WORDS - _GENERIC_FOCUS
    bare_follow_up = bool(
        speech_act == "question"
        and predicate is None
        and not required_roles
        and not turn_focus
    )
    turn_words = set(_normalized_words(text))
    self_identity_question = bool(
        question_word == "кто"
        and turn_words.intersection({"я", "ты"})
    )
    # Dialogue-state topics contain both sides of the exchange. For an
    # elliptical follow-up, inheriting those topics can make the agent's own
    # bad reply the subject of its next answer. Recover the preceding human
    # turn explicitly; an empty contact call intentionally contributes no
    # fake topic.
    inherited_focus = (
        content_lemmas(latest_user_turn or "")
        if bare_follow_up
        else ()
        if self_identity_question
        else dialogue_state.topic.active_concepts
        if dialogue_state is not None
        else ()
    )
    focus = tuple(dict.fromkeys((*inherited_focus, *content_lemmas(text))))[:12]
    missing = tuple(
        dict.fromkeys(
            reference.semantic_role
            for reference in (dialogue_state.unresolved_references if dialogue_state else ())
        )
    )
    payload = _requested_utterance(structure, clause, predicate)
    repair_kind = _repair_kind(structure, latest_agent_turn)

    if repair_kind is not None:
        return _TaskDraft(
            repair_kind,
            "repair_shared_model",
            TaskObjective(
                speech_act,
                predicate,
                question_word,
                focus,
                required_roles=required_roles,
                revisions=_repair_revisions(structure, latest_agent_turn),
                missing_slots=missing,
                target_event_id=source_event_id,
                required_response_act="repair",
            ),
            1.0,
            0.88,
        )
    if predicate in _COMMUNICATION_ACTIONS and payload:
        return _TaskDraft(
            "requested_communication",
            "perform_utterance",
            TaskObjective(
                speech_act,
                predicate,
                question_word,
                focus,
                required_content=payload,
                required_roles=required_roles,
                target_event_id=source_event_id,
                required_response_act="perform",
            ),
            0.97 if speech_act == "command" else 0.92,
            0.96,
        )
    if (
        speech_act == "command"
        and predicate in {"ответить", "отвечать"}
        and payload is None
    ):
        # "Ответь нормально/короче/иначе" changes how the pending response
        # should be produced.  The manner adverb is not literal text to echo.
        return _TaskDraft(
            "response_control",
            "respond_relevantly",
            TaskObjective(
                speech_act,
                predicate,
                None,
                focus,
                required_roles=required_roles,
                missing_slots=missing,
                target_event_id=source_event_id,
                required_response_act="respond",
            ),
            0.90,
            0.86,
        )
    if speech_act == "question":
        if bare_follow_up:
            # A bare "почему?/как?/что?" is a dialogue continuation.  It has
            # no independent proposition to answer and must inherit meaning
            # through the context resolver rather than a canned no-answer.
            return _TaskDraft(
                "elliptical_follow_up",
                "respond_relevantly",
                TaskObjective(
                    speech_act,
                    predicate,
                    question_word,
                    focus,
                    target_event_id=source_event_id,
                    required_response_act="respond",
                ),
                0.88,
                0.78,
            )
        return _TaskDraft(
            "information_need",
            "answer_question",
            TaskObjective(
                speech_act,
                predicate,
                question_word,
                focus,
                required_roles=required_roles,
                missing_slots=missing,
                target_event_id=source_event_id,
                required_response_act="answer",
            ),
            0.91,
            max(0.55, clause.confidence if clause is not None else 0.55),
        )
    if speech_act == "command":
        slots = missing or (("direct_object",) if not required_roles else ())
        return _TaskDraft(
            "requested_action",
            "satisfy_directive",
            TaskObjective(
                speech_act,
                predicate,
                None,
                focus,
                required_roles=required_roles,
                missing_slots=slots,
                target_event_id=source_event_id,
                required_response_act="perform",
            ),
            0.88,
            max(0.50, clause.confidence if clause is not None else 0.50),
        )
    return _TaskDraft(
        "dialogue_update",
        "respond_relevantly",
        TaskObjective(
            speech_act,
            predicate,
            question_word,
            focus,
            required_roles=required_roles,
            missing_slots=missing,
            target_event_id=source_event_id,
            required_response_act="respond",
        ),
        0.66,
        max(0.45, clause.confidence if clause is not None else 0.45),
    )


def evaluate_task_progress(
    task: DialogueTask,
    text: str,
    metadata: Mapping[str, object] | None = None,
) -> TaskProgress:
    """Measure the candidate against the task effect, never against phrase frequency."""

    metadata = metadata or {}
    words = _normalized_words(text)
    lemmas = set(content_lemmas(text))
    objective = task.objective
    reasons: list[str] = []

    if task.goal == "perform_utterance":
        requested = _normalized_words(objective.required_content or "")
        if requested and (words == requested or _contains_sequence(words, requested)):
            return TaskProgress(1.0, ("requested_content_realized",))
        return TaskProgress(0.0, ("requested_content_absent",))

    candidate_structure = analyze_structured_utterance(text)
    candidate_clause = _main_clause(candidate_structure)
    candidate_is_question = bool(
        text.rstrip().endswith("?")
        or (candidate_clause is not None and candidate_clause.clause_kind == "question")
    )
    candidate_is_command = bool(
        candidate_clause is not None and candidate_clause.clause_kind == "command"
    )
    genuine_clarification = bool(
        metadata.get("genuine_clarification")
        or metadata.get("unresolved_reference")
        or metadata.get("response_move") in {"clarify_reference", "request_required_argument"}
    )
    if (
        objective.required_response_act in {"answer", "repair"}
        and candidate_is_question
        and not genuine_clarification
    ):
        return TaskProgress(0.0, ("question_does_not_fulfill_response_obligation",))
    if candidate_is_command and objective.required_response_act != "perform":
        return TaskProgress(0.0, ("command_reverses_response_obligation",))

    focused_request = metadata.get("focused_request")
    focus = (
        set(content_lemmas(focused_request))
        if isinstance(focused_request, str)
        else set(objective.focus_terms)
    ) - _GENERIC_FOCUS - _QUESTION_WORDS
    overlap = len(lemmas & focus) / len(focus) if focus else 0.0
    predicate_fit = float(bool(objective.predicate and objective.predicate in lemmas))
    polar_question = bool(objective.question_word is None and objective.predicate)
    polar = float(polar_question and bool(set(words[:2]) & {"да", "нет"}))
    grounded = float(bool(metadata.get("premises") or metadata.get("reasoning_evidence")))
    answered_slot = float(bool(metadata.get("answer_slot")))
    response_move = str(metadata.get("response_move", ""))
    obligation_fit = (
        float(metadata["obligation_fit"])
        if isinstance(metadata.get("obligation_fit"), int | float)
        and not isinstance(metadata.get("obligation_fit"), bool)
        else 0.0
    )

    if task.kind == "dialogue_update" and task.goal == "respond_relevantly":
        learned_support = bool(
            metadata.get("corpus_source_event_ids")
            or metadata.get("uses_persistent_word_statistics")
        )
        semantic_transform = bool(
            metadata.get("meaning_origin") == "affect_weighted_stance"
            and metadata.get("semantic_origin") == "current_turn_roles"
            and metadata.get("semantic_operation")
            and metadata.get("reasoning_evidence")
        )
        relation_support = (
            float(metadata["semantic_relation_support"])
            if isinstance(metadata.get("semantic_relation_support"), int | float)
            and not isinstance(metadata.get("semantic_relation_support"), bool)
            else 0.0
        )
        relation_conflicts = int(metadata.get("semantic_relation_conflicts", 0) or 0)
        if metadata.get("contact_request") is True:
            contact_binding = (
                float(metadata["contact_response_binding"])
                if isinstance(metadata.get("contact_response_binding"), int | float)
                and not isinstance(metadata.get("contact_response_binding"), bool)
                else 0.0
            )
            if learned_support and contact_binding >= 0.80 and not candidate_is_question:
                return TaskProgress(
                    max(0.52, contact_binding * 0.72),
                    ("agent_communicative_action_answers_contact",),
                )
            return TaskProgress(0.0, ("contact_not_answered_by_candidate_meaning",))
        unresolved_binding = (
            float(metadata["unresolved_input_binding"])
            if isinstance(metadata.get("unresolved_input_binding"), int | float)
            and not isinstance(metadata.get("unresolved_input_binding"), bool)
            else 0.0
        )
        if (
            learned_support
            and metadata.get("unresolved_input_reaction") is True
            and bool(metadata.get("unresolved_prompt_terms"))
            and unresolved_binding >= 0.80
            and not candidate_is_question
        ):
            return TaskProgress(
                max(0.58, unresolved_binding * 0.68),
                ("learned_cognitive_state_marks_unresolved_input",),
            )
        if (
            (learned_support or semantic_transform)
            and lemmas
            and not candidate_is_question
            and not relation_conflicts
        ):
            score = overlap * 0.62 + predicate_fit * 0.25 + relation_support * 0.13
            if overlap:
                reasons.append("dialogue_topic_preserved")
            if predicate_fit:
                reasons.append("dialogue_predicate_preserved")
            if relation_support:
                reasons.append("learned_word_relations_preserved")
            if semantic_transform:
                reasons.append("current_proposition_transformed_by_stance")
            return TaskProgress(
                _clamp(score),
                tuple(reasons or ("dialogue_update_not_semantically_connected",)),
            )

    if task.goal == "resolve_missing_evidence":
        corpus_sources = tuple(metadata.get("corpus_source_event_ids", ()))
        learned_support = bool(
            corpus_sources
            or metadata.get("uses_persistent_word_statistics")
            or metadata.get("premises")
            or metadata.get("reasoning_evidence")
        )
        substantive = bool(lemmas)
        score = overlap * 0.35 + predicate_fit * 0.15
        relevant = bool(overlap or predicate_fit or polar or answered_slot)
        if learned_support and substantive and relevant:
            score = max(score + 0.25, 0.52)
            reasons.append("learned_language_evidence_used")
        if overlap:
            reasons.append("missing_evidence_focus_preserved")
        if predicate_fit:
            reasons.append("missing_evidence_predicate_preserved")
        return TaskProgress(
            _clamp(score),
            tuple(reasons or ("missing_evidence_not_advanced",)),
        )

    if task.goal == "answer_question":
        score = overlap * 0.42 + predicate_fit * 0.20 + polar * 0.45
        score += grounded * 0.22 + answered_slot * 0.18
        if (
            objective.question_word == "кто"
            and metadata.get("identity_grounded") is True
            and grounded
            and answered_slot
        ):
            score = max(score, 0.96)
            reasons.append("identity_slot_grounded_by_runtime_fact")
        if (
            metadata.get("semantic_recovery") is True
            and grounded
            and response_move.startswith(("state_", "withhold_"))
        ):
            # An explicit, current-question-bound knowledge limit closes the
            # response obligation without inventing an answer or returning a
            # question.  It is weaker than grounded content, but meaningful.
            score = max(score, 0.58)
            reasons.append("question_bound_epistemic_limit")
        if overlap:
            reasons.append("focus_preserved")
        if predicate_fit:
            reasons.append("question_predicate_preserved")
        if grounded:
            reasons.append("answer_has_evidence")
        if polar:
            reasons.append("polar_answer")
        return TaskProgress(_clamp(score), tuple(reasons or ("question_not_advanced",)))

    if task.goal == "repair_shared_model":
        self_reference = bool(set(words) & {"я", "меня", "мне", "мой", "моя"})
        revision = bool(set(words) & {"ошибка", "ошибиться", "перепутать", "исправить", "не"})
        contact_binding = (
            float(metadata["contact_response_binding"])
            if isinstance(metadata.get("contact_response_binding"), int | float)
            and not isinstance(metadata.get("contact_response_binding"), bool)
            else 0.0
        )
        meaning_connected = bool(overlap or predicate_fit) or bool(
            metadata.get("contact_request") is True and contact_binding >= 0.80
        )
        if task.kind == "response_quality_repair" and not meaning_connected:
            return TaskProgress(0.0, ("repair_does_not_replace_failed_meaning",))
        score = overlap * 0.45 + float(self_reference) * 0.20 + float(revision) * 0.35
        if (
            response_move in {"targeted_repair", "clarify_reference"}
            and obligation_fit >= 0.70
            and meaning_connected
        ):
            score = max(score, obligation_fit * 0.72)
            reasons.append("repair_obligation_fulfilled")
        return TaskProgress(
            _clamp(score),
            tuple(
                (
                    *reasons,
                    *(
                        reason
                        for present, reason in (
                            (bool(overlap), "correction_focus_preserved"),
                            (self_reference, "agent_role_acknowledged"),
                            (revision, "revision_expressed"),
                        )
                        if present
                    ),
                )
            )
            or ("repair_not_addressed",),
        )

    score = overlap * 0.65 + predicate_fit * 0.25 + grounded * 0.20
    if response_move and obligation_fit >= 0.70:
        score = max(score, obligation_fit * 0.72)
        reasons.append("dialogue_obligation_fulfilled")
    if objective.missing_slots and text.rstrip().endswith("?"):
        score = max(score, 0.58)
        reasons.append("missing_slot_queried")
    if overlap:
        reasons.append("active_focus_preserved")
    if predicate_fit:
        reasons.append("task_predicate_preserved")
    return TaskProgress(_clamp(score), tuple(reasons or ("task_not_advanced",)))


def task_candidate_metadata(
    task: DialogueTask | None,
    text: str,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result = dict(metadata or {})
    if task is None:
        return result
    result["dialogue_task_id"] = task.task_id
    result["dialogue_task_goal"] = task.goal
    result["required_response_act"] = task.objective.required_response_act
    progress = evaluate_task_progress(task, text, result)
    result.update(
        {
            # An addressed turn always creates a conversational obligation.
            # Priority orders open tasks; it must not license an unrelated
            # surface when the current task happens to be low confidence.
            "dialogue_task_required": True,
            "dialogue_task_progress": progress.score,
            "dialogue_task_progress_reasons": progress.reasons,
            "context_relevance": max(
                progress.score,
                float(result.get("context_relevance", 0.0))
                if isinstance(result.get("context_relevance"), int | float)
                else 0.0,
            ),
        }
    )
    return result


class DialogueTaskRepository:
    def __init__(self, database: Database, *, attempt_window: timedelta = timedelta(minutes=15)):
        self._database = database
        self._attempt_window = attempt_window

    async def deliberate(
        self,
        event: Event,
        *,
        prompt: str,
        consultation_messages: tuple[str, ...],
        dialogue_state: DialogueStateAnalysis | None,
    ) -> DialogueSituation | None:
        if event.channel_id is None or event.actor_id is None or not prompt.strip():
            return None
        existing = await self._database.fetchone(
            """SELECT task_id FROM dialogue_task_events
               WHERE event_id = ? AND relation = 'source' ORDER BY observed_at DESC LIMIT 1""",
            (event.event_id,),
        )
        if existing is not None:
            return await self.situation(event.guild_id, event.channel_id, event.actor_id)

        latest_agent = _latest_role_turn(consultation_messages, "agent")
        latest_user = _latest_role_turn(consultation_messages, "user")
        draft = infer_dialogue_task(
            prompt,
            source_event_id=event.event_id,
            dialogue_state=dialogue_state,
            latest_agent_turn=latest_agent,
            latest_user_turn=latest_user,
        )
        now = event.utc_timestamp
        async with self._database.transaction() as connection:
            prior = await (
                await connection.execute(
                    """SELECT a.*, t.signature, t.goal, t.objective_json
                       FROM dialogue_task_attempts a
                       JOIN dialogue_tasks t USING (task_id)
                       WHERE t.guild_id = ? AND t.channel_id = ? AND t.user_id = ?
                         AND a.assessment = 'pending' AND a.created_at >= ?
                       ORDER BY a.created_at DESC LIMIT 1""",
                    (
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        (now - self._attempt_window).isoformat(),
                    ),
                )
            ).fetchone()
            parent_task_id = None
            if prior is not None:
                assessment = _assess_attempt(prompt, draft, prior)
                await connection.execute(
                    """UPDATE dialogue_task_attempts
                       SET assessment = ?, outcome_event_id = ?, assessed_at = ?
                       WHERE attempt_id = ?""",
                    (assessment, event.event_id, now.isoformat(), str(prior["attempt_id"])),
                )
                if assessment == "completed":
                    await connection.execute(
                        """UPDATE dialogue_tasks SET status = 'completed', resolved_at = ?,
                               updated_at = ? WHERE task_id = ?""",
                        (now.isoformat(), now.isoformat(), str(prior["task_id"])),
                    )
                elif assessment in {"failed", "continued"}:
                    await connection.execute(
                        """UPDATE dialogue_tasks SET status = 'open',
                               priority = MIN(1.0, priority + 0.05), updated_at = ?
                           WHERE task_id = ?""",
                        (now.isoformat(), str(prior["task_id"])),
                    )
                if draft.goal == "repair_shared_model":
                    parent_task_id = str(prior["task_id"])

            row = await (
                await connection.execute(
                    """SELECT * FROM dialogue_tasks
                       WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                         AND signature = ? AND status IN ('open', 'attempted')
                       ORDER BY updated_at DESC LIMIT 1""",
                    (event.guild_id, event.channel_id, event.actor_id, draft.signature),
                )
            ).fetchone()
            objective_json = _json(asdict(draft.objective))
            if row is None:
                task_id = "dialogue-task:" + hashlib.sha256(
                    f"{event.event_id}|{draft.signature}".encode()
                ).hexdigest()[:32]
                await connection.execute(
                    """INSERT INTO dialogue_tasks (
                           task_id, guild_id, channel_id, user_id, source_event_id,
                           parent_task_id, signature, kind, goal, objective_json,
                           priority, confidence, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        task_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        event.event_id,
                        parent_task_id,
                        draft.signature,
                        draft.kind,
                        draft.goal,
                        objective_json,
                        draft.priority,
                        draft.confidence,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            else:
                task_id = str(row["task_id"])
                await connection.execute(
                    """UPDATE dialogue_tasks
                       SET objective_json = ?, priority = MAX(priority, ?),
                           confidence = MAX(confidence, ?), status = 'open',
                           recurrence_count = recurrence_count + 1, updated_at = ?
                       WHERE task_id = ?""",
                    (
                        objective_json,
                        draft.priority,
                        draft.confidence,
                        now.isoformat(),
                        task_id,
                    ),
                )
            await connection.execute(
                """INSERT OR IGNORE INTO dialogue_task_events (
                       task_id, event_id, relation, observation_json, observed_at
                   ) VALUES (?, ?, 'source', ?, ?)""",
                (
                    task_id,
                    event.event_id,
                    _json({"prompt_terms": content_lemmas(prompt), "goal": draft.goal}),
                    now.isoformat(),
                ),
            )
            situation_row = await (
                await connection.execute(
                    """SELECT revision FROM dialogue_situations
                       WHERE guild_id = ? AND channel_id = ? AND user_id = ?""",
                    (event.guild_id, event.channel_id, event.actor_id),
                )
            ).fetchone()
            revision = (int(situation_row["revision"]) + 1) if situation_row else 1
            await connection.execute(
                """INSERT INTO dialogue_situations (
                       guild_id, channel_id, user_id, active_task_id, revision,
                       situation_json, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (guild_id, channel_id, user_id) DO UPDATE SET
                       active_task_id = excluded.active_task_id,
                       revision = excluded.revision,
                       situation_json = excluded.situation_json,
                       updated_at = excluded.updated_at""",
                (
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    task_id,
                    revision,
                    _json(
                        {
                            "active_goal": draft.goal,
                            "focus_terms": draft.objective.focus_terms,
                            "parent_task_id": parent_task_id,
                            "reason": "current_turn_deliberation",
                        }
                    ),
                    now.isoformat(),
                ),
            )
        return await self.situation(event.guild_id, event.channel_id, event.actor_id)

    async def spawn_subtask(
        self,
        event: Event,
        parent: DialogueTask,
        *,
        missing_slots: tuple[str, ...],
    ) -> DialogueSituation:
        """Create an agent-originated reasoning step when the parent cannot be solved yet."""

        now = event.utc_timestamp
        identity = (parent.task_id, "resolve_missing_evidence", tuple(sorted(missing_slots)))
        signature = hashlib.sha256(_json(identity).encode()).hexdigest()[:24]
        task_id = "dialogue-task:" + hashlib.sha256(
            f"{event.event_id}|{signature}".encode()
        ).hexdigest()[:32]
        objective = TaskObjective(
            "internal",
            parent.objective.predicate,
            parent.objective.question_word,
            parent.objective.focus_terms,
            required_roles=parent.objective.required_roles,
            missing_slots=missing_slots or ("evidence",),
            target_event_id=parent.source_event_id,
            required_response_act=parent.objective.required_response_act,
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO dialogue_tasks (
                       task_id, guild_id, channel_id, user_id, source_event_id,
                       parent_task_id, signature, kind, goal, objective_json,
                       priority, confidence, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reasoning_subtask',
                             'resolve_missing_evidence', ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    event.event_id,
                    parent.task_id,
                    signature,
                    _json(asdict(objective)),
                    max(0.72, parent.priority - 0.08),
                    parent.confidence,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await connection.execute(
                """INSERT OR IGNORE INTO dialogue_task_events (
                       task_id, event_id, relation, observation_json, observed_at
                   ) VALUES (?, ?, 'self_generated', ?, ?)""",
                (
                    task_id,
                    event.event_id,
                    _json(
                        {
                            "parent_task_id": parent.task_id,
                            "missing_slots": objective.missing_slots,
                        }
                    ),
                    now.isoformat(),
                ),
            )
            await connection.execute(
                """UPDATE dialogue_situations SET active_task_id = ?, revision = revision + 1,
                       situation_json = ?, updated_at = ?
                   WHERE guild_id = ? AND channel_id = ? AND user_id = ?""",
                (
                    task_id,
                    _json(
                        {
                            "active_goal": "resolve_missing_evidence",
                            "parent_task_id": parent.task_id,
                            "missing_slots": objective.missing_slots,
                            "reason": "planner_could_not_resolve_parent",
                        }
                    ),
                    now.isoformat(),
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                ),
            )
        situation = await self.situation(event.guild_id, event.channel_id, event.actor_id)
        if situation is None:
            raise RuntimeError("failed to persist dialogue subtask")
        return situation

    async def record_attempt(
        self,
        prompt: Event,
        *,
        task_id: str,
        response_message_id: int,
        text: str,
        created_at: datetime,
        predicted_progress: float | None = None,
    ) -> bool:
        task = await self.task(task_id)
        linked = await self._database.fetchone(
            """SELECT 1 FROM dialogue_task_events
               WHERE task_id = ? AND event_id = ?""",
            (task_id, prompt.event_id),
        )
        if task is None or linked is None:
            return False
        progress = evaluate_task_progress(task, text)
        progress_score = (
            _clamp(predicted_progress) if predicted_progress is not None else progress.score
        )
        assessment = (
            "completed" if task.locally_verifiable and progress_score >= 0.95 else "pending"
        )
        attempt_id = "dialogue-attempt:" + hashlib.sha256(
            f"{task_id}|{response_message_id}".encode()
        ).hexdigest()[:32]
        timestamp = created_at.astimezone(UTC).isoformat()
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """INSERT OR IGNORE INTO dialogue_task_attempts (
                       attempt_id, task_id, prompt_event_id, response_message_id,
                       response_text, predicted_progress, assessment, created_at, assessed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    task_id,
                    prompt.event_id,
                    response_message_id,
                    text,
                    progress_score,
                    assessment,
                    timestamp,
                    timestamp if assessment == "completed" else None,
                ),
            )
            if cursor.rowcount != 1:
                return False
            await connection.execute(
                """UPDATE dialogue_tasks SET attempt_count = attempt_count + 1,
                       status = ?, updated_at = ?, resolved_at = ? WHERE task_id = ?""",
                (
                    "completed" if assessment == "completed" else "attempted",
                    timestamp,
                    timestamp if assessment == "completed" else None,
                    task_id,
                ),
            )
        return True

    async def task(self, task_id: str) -> DialogueTask | None:
        row = await self._database.fetchone(
            "SELECT * FROM dialogue_tasks WHERE task_id = ?", (task_id,)
        )
        return _task_from_row(row) if row is not None else None

    async def situation(
        self, guild_id: int, channel_id: int, user_id: int
    ) -> DialogueSituation | None:
        state = await self._database.fetchone(
            """SELECT * FROM dialogue_situations
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?""",
            (guild_id, channel_id, user_id),
        )
        if state is None:
            return None
        rows = await self._database.fetchall(
            """SELECT * FROM dialogue_tasks
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND status IN ('open', 'attempted')
               ORDER BY priority DESC, updated_at DESC LIMIT 12""",
            (guild_id, channel_id, user_id),
        )
        tasks = tuple(_task_from_row(row) for row in rows)
        active_id = str(state["active_task_id"]) if state["active_task_id"] else None
        active = next((task for task in tasks if task.task_id == active_id), None)
        if active is None and active_id:
            active = await self.task(active_id)
        return DialogueSituation(
            guild_id,
            channel_id,
            user_id,
            active,
            tasks,
            int(state["revision"]),
            datetime.fromisoformat(str(state["updated_at"])),
        )


def _main_clause(structure: StructuredUtterance):
    top_level = {link.child_clause_index for link in structure.clause_links}
    return next(
        (clause for clause in structure.clauses if clause.clause_index not in top_level),
        next(iter(structure.clauses), None),
    )


def _request_clause(structure: StructuredUtterance):
    top_level = {link.child_clause_index for link in structure.clause_links}
    clauses = tuple(
        clause for clause in structure.clauses if clause.clause_index not in top_level
    )
    return next(
        (clause for clause in reversed(clauses) if clause.clause_kind == "question"),
        next(
            (clause for clause in reversed(clauses) if clause.clause_kind == "command"),
            clauses[0] if clauses else next(iter(structure.clauses), None),
        ),
    )


def _question_word(structure: StructuredUtterance, clause) -> str | None:
    if clause is None:
        return None
    if clause.question_target_index is not None:
        return structure.tokens[clause.question_target_index].lemma
    if clause.clause_kind != "question" and not structure.text.rstrip().endswith("?"):
        return None
    return next(
        (
            token.lemma
            for token in structure.tokens[clause.token_start : clause.token_end]
            if token.lemma in _QUESTION_WORDS
        ),
        None,
    )


def _known_roles(structure: StructuredUtterance, clause) -> tuple[tuple[str, str], ...]:
    if clause is None:
        return ()
    return tuple(
        (role, structure.tokens[index].lemma)
        for role, indices in (
            ("subject", clause.subject_indices),
            ("direct_object", clause.direct_object_indices),
            ("indirect_object", clause.indirect_object_indices),
            ("action_complement", clause.action_complement_indices),
            ("predicate_complement", clause.predicate_complement_indices),
        )
        for index in indices
    )


def _requested_utterance(
    structure: StructuredUtterance, clause, predicate: str | None
) -> str | None:
    if clause is None or predicate not in _COMMUNICATION_ACTIONS or clause.predicate_index is None:
        return None
    if any(
        structure.tokens[index].lemma.endswith(("-нибудь", "-то", "-либо"))
        for index in clause.direct_object_indices
    ):
        # An indefinite object requests original content ("напиши что-нибудь
        # странное"), not the literal adjective "Странное".
        return None
    excluded = {
        clause.predicate_index,
        *clause.subject_indices,
        *clause.indirect_object_indices,
        *clause.vocative_indices,
        *(
            index
            for complement in clause.complements
            for index in (complement.preposition_index, *complement.object_indices)
        ),
    }
    payload = []
    for token in structure.tokens[clause.token_start : clause.token_end]:
        if token.index in excluded or not token.surface[:1].isalnum():
            continue
        if token.part_of_speech == "adverb":
            # Manner belongs to the requested response policy, not to the
            # utterance payload ("ответь нормально" must not emit "Нормально").
            continue
        if token.lemma in _DISCOURSE_WORDS:
            continue
        if token.semantic_role == "question_target":
            continue
        payload.append(token.corrected_form or token.surface)
    if not payload:
        return None
    rendered = " ".join(payload).strip()
    return rendered if rendered else None


def _repair_kind(structure: StructuredUtterance, latest_agent_turn: str | None) -> str | None:
    if not latest_agent_turn:
        return None
    if "prior_answer_challenge" in analyze_pragmatics(structure.text).functions:
        return "response_quality_repair"
    words = _normalized_words(structure.text)
    if words in {("что",), ("чего",)}:
        return "response_quality_repair"
    lemmas = {
        token.lemma.removesuffix("-то").removesuffix("-ка") for token in structure.tokens
    }
    role_contrast = {"я", "ты"} <= lemmas and (
        "не" in lemmas
        or any(
            coordination.relation == "contrast"
            for clause in structure.clauses
            for coordination in clause.coordinations
        )
    )
    if role_contrast:
        return "attribution_repair"
    main = _main_clause(structure)
    question_word = (
        structure.tokens[main.question_target_index].lemma
        if main is not None and main.question_target_index is not None
        else None
    )
    subordinate = any(link.relation.startswith("ccomp") for link in structure.clause_links)
    if main is not None and main.clause_kind == "question" and subordinate and question_word in {
        "почему",
        "зачем",
    }:
        return "failure_explanation"
    return None


def _repair_revisions(
    structure: StructuredUtterance, latest_agent_turn: str | None
) -> tuple[tuple[str, str], ...]:
    """Recover an actor correction as role changes for any predicate."""

    if latest_agent_turn is None:
        return ()
    lemmas = {
        token.lemma.removesuffix("-то").removesuffix("-ка") for token in structure.tokens
    }
    if not ({"я", "ты"} <= lemmas and "не" in lemmas):
        return ()
    previous = analyze_structured_utterance(latest_agent_turn)
    clause = _main_clause(previous)
    if clause is None or clause.predicate_index is None:
        return ()
    predicate = previous.tokens[clause.predicate_index].lemma
    object_indices = (
        *clause.direct_object_indices,
        *clause.action_complement_indices,
        *clause.predicate_complement_indices,
    )
    object_token = previous.tokens[object_indices[-1]] if object_indices else None
    revisions = [
        ("predicate", predicate),
        ("asserted_actor", "agent"),
        ("retracted_actor", "user"),
    ]
    if object_token is not None:
        revisions.append(("object", object_token.lemma))
    return tuple(revisions)


def _latest_role_turn(messages: tuple[str, ...], role: str) -> str | None:
    for message in reversed(messages):
        speaker, separator, text = message.partition(": ")
        if separator and speaker.casefold() == role and text.strip():
            return text.strip()
    return None


def _assess_attempt(prompt: str, draft: _TaskDraft, row) -> str:
    words = set(_normalized_words(prompt))
    if draft.signature == str(row["signature"]):
        return "failed"
    if draft.goal == "repair_shared_model" or words & _REJECTION:
        return "failed"
    if words and words <= _AFFIRMATION and "?" not in prompt:
        return "completed"
    objective = TaskObjective(**json.loads(str(row["objective_json"])))
    previous_focus = set(objective.focus_terms) - _GENERIC_FOCUS
    current_focus = set(draft.objective.focus_terms) - _GENERIC_FOCUS
    if previous_focus and current_focus and previous_focus & current_focus:
        return "continued"
    return "unassessed"


def _task_from_row(row) -> DialogueTask:
    raw = json.loads(str(row["objective_json"]))
    objective = TaskObjective(
        speech_act=str(raw["speech_act"]),
        predicate=raw.get("predicate"),
        question_word=raw.get("question_word"),
        focus_terms=tuple(raw.get("focus_terms", ())),
        required_content=raw.get("required_content"),
        required_roles=tuple(tuple(item) for item in raw.get("required_roles", ())),
        revisions=tuple(tuple(item) for item in raw.get("revisions", ())),
        missing_slots=tuple(raw.get("missing_slots", ())),
        target_event_id=raw.get("target_event_id"),
        required_response_act=(
            raw.get("required_response_act")
            or (
                "answer"
                if str(row["goal"]) in {"answer_question", "resolve_missing_evidence"}
                else "repair"
                if str(row["goal"]) == "repair_shared_model"
                else "perform"
                if str(row["goal"]) in {"perform_utterance", "satisfy_directive"}
                else "respond"
            )
        ),
    )
    return DialogueTask(
        task_id=str(row["task_id"]),
        guild_id=int(row["guild_id"]),
        channel_id=int(row["channel_id"]),
        user_id=int(row["user_id"]),
        source_event_id=str(row["source_event_id"]),
        parent_task_id=str(row["parent_task_id"]) if row["parent_task_id"] else None,
        signature=str(row["signature"]),
        kind=str(row["kind"]),
        goal=str(row["goal"]),
        objective=objective,
        priority=float(row["priority"]),
        confidence=float(row["confidence"]),
        status=str(row["status"]),
        recurrence_count=int(row["recurrence_count"]),
        attempt_count=int(row["attempt_count"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )
