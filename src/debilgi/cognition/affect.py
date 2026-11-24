"""Persistent continuous affect shaped by observed interaction consequences."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache

from debilgi.events.models import Event, EventType
from debilgi.social.models import PrimitiveVector
from debilgi.social.repository import SocialReactionRepository
from debilgi.storage.database import Database

_MESSAGES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


@dataclass(frozen=True, slots=True)
class AffectVector:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    novelty: float = 0.0
    affiliation: float = 0.0

    def features(self, prefix: str) -> tuple[tuple[str, float], ...]:
        return tuple((f"{prefix}_{key}", value) for key, value in asdict(self).items())


@dataclass(frozen=True, slots=True)
class AffectState:
    guild_id: int
    actor_id: int
    mood: AffectVector
    emotion: AffectVector
    mood_observations: int
    emotion_observations: int
    mood_updated_at: datetime | None
    emotion_updated_at: datetime | None

    @property
    def features(self) -> tuple[tuple[str, float], ...]:
        return (*self.mood.features("mood"), *self.emotion.features("emotion"))

    @property
    def labels(self) -> tuple[str, str]:
        return nearest_affect(self.mood), nearest_affect(self.emotion)


# Labels make the wheel inspectable. They never select an action or phrase.
_POINTS = {
    "радость": (0.85, 0.45),
    "спокойствие": (0.65, -0.65),
    "скука": (-0.35, -0.85),
    "печаль": (-0.80, -0.40),
    "злость": (-0.72, 0.76),
    "тревога": (-0.45, 0.88),
    "удивление": (0.15, 0.95),
    "интерес": (0.48, 0.68),
    "нейтральность": (0.0, 0.0),
}


@lru_cache(maxsize=16384)
def lexical_valence(text: str) -> float | None:
    """Valence of the heard event, not the agent's accumulated mood.

    Reuse existing social evidence and emotion coordinates. Unknown vocabulary
    stays unknown; no event names or response phrases are bootstrapped here.
    """
    from debilgi.language.lexical_matching import expressive_family, same_lexical_family
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY
    from debilgi.language.structured_utterance import analyze_structured_utterance
    from debilgi.social.signals import APPROVAL, AVERSION, DISAPPROVAL, evidence_counts

    structure = analyze_structured_utterance(text)
    values = []
    for clause in structure.clauses:
        tokens = structure.tokens[clause.token_start:clause.token_end]
        counts = evidence_counts(tuple(t.surface for t in tokens))
        score = counts["approval"] - counts["disapproval"] - counts["aversion"]
        for token in tokens:
            for words, direction in ((APPROVAL, 1), (DISAPPROVAL | AVERSION, -1)):
                if token.surface not in words and any(
                    same_lexical_family(token.lemma, word) for word in words
                ):
                    score += direction
            short = DEFAULT_MORPHOLOGY.inflect(
                token.lemma, form="short_adjective", gender="neuter", number="singular",
            ) if token.part_of_speech == "adjective" else None
            if short:
                evidence = evidence_counts((short,))
                score += evidence["approval"] - evidence["disapproval"] - evidence["aversion"]
            for label, (valence, _) in _POINTS.items():
                if abs(valence) >= 0.3 and expressive_family(token.lemma, label):
                    score += valence
        if score:
            values.append(max(-1.0, min(1.0, -score if clause.negated else score)))
    return sum(values) / len(values) if values else None


def nearest_affect(vector: AffectVector) -> str:
    return min(
        _POINTS,
        key=lambda name: (
            (vector.valence - _POINTS[name][0]) ** 2 + (vector.arousal - _POINTS[name][1]) ** 2
        ),
    )


def stimulus(vector: PrimitiveVector, hostile: bool = False) -> AffectVector:
    positive = (vector.approval + vector.agreement + vector.trust) / 3.0
    # (4.42, flag only) An attack on the agent is aversion by the evidence of its cues: the
    # reaction vector, built from approval and agreement words and from familiarity, made an
    # insult from a familiar person a pleasant stimulus.
    negative = 1.0 if hostile else (vector.resistance + vector.aversion) / 2.0
    return AffectVector(
        valence=_signed(positive - negative),
        arousal=_signed(vector.arousal * 2.0 - 1.0),
        dominance=_signed(vector.agreement + vector.trust - vector.resistance - 0.5),
        novelty=_signed(vector.curiosity * 2.0 - 1.0),
        affiliation=_signed(vector.trust + vector.approval - vector.aversion - 0.5),
    )


class AffectRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def observe(self, event: Event, social: PrimitiveVector, hostile: bool = False) -> bool:
        if (
            event.event_type not in _MESSAGES
            or event.payload.get("author_is_bot") is True
            or event.actor_id is None
        ):
            return False
        if await self._database.fetchone(
            "SELECT 1 FROM affect_observations WHERE event_id = ?", (event.event_id,)
        ):
            return False
        current = await self.state(event.guild_id, event.actor_id, at=event.utc_timestamp)
        impulse = stimulus(social, hostile)
        mood = _advance(
            current.mood,
            impulse,
            event.utc_timestamp,
            current.mood_updated_at,
            half_life_seconds=21600.0,
            learning_rate=0.08,
        )
        emotion = _advance(
            current.emotion,
            impulse,
            event.utc_timestamp,
            current.emotion_updated_at,
            half_life_seconds=720.0,
            learning_rate=0.38,
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO affect_moods VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id) DO UPDATE SET
                    valence=excluded.valence, arousal=excluded.arousal,
                    dominance=excluded.dominance, novelty=excluded.novelty,
                    affiliation=excluded.affiliation,
                    observations=affect_moods.observations+1, updated_at=excluded.updated_at""",
                (
                    event.guild_id,
                    *asdict(mood).values(),
                    current.mood_observations + 1,
                    event.utc_timestamp.isoformat(),
                ),
            )
            await connection.execute(
                """INSERT OR IGNORE INTO affect_emotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id,actor_id) DO UPDATE SET
                    valence=excluded.valence, arousal=excluded.arousal,
                    dominance=excluded.dominance, novelty=excluded.novelty,
                    affiliation=excluded.affiliation,
                    observations=affect_emotions.observations+1, updated_at=excluded.updated_at""",
                (
                    event.guild_id,
                    event.actor_id,
                    *asdict(emotion).values(),
                    current.emotion_observations + 1,
                    event.utc_timestamp.isoformat(),
                ),
            )
            await connection.execute(
                """INSERT INTO affect_observations VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.guild_id,
                    event.actor_id,
                    _json(impulse),
                    _json(mood),
                    _json(emotion),
                    event.utc_timestamp.isoformat(),
                ),
            )
        return True

    async def state(
        self, guild_id: int, actor_id: int, *, at: datetime | None = None
    ) -> AffectState:
        mood = await self._database.fetchone(
            "SELECT * FROM affect_moods WHERE guild_id=?", (guild_id,)
        )
        emotion = await self._database.fetchone(
            "SELECT * FROM affect_emotions WHERE guild_id=? AND actor_id=?", (guild_id, actor_id)
        )
        mood_updated = datetime.fromisoformat(mood["updated_at"]) if mood else None
        emotion_updated = datetime.fromisoformat(emotion["updated_at"]) if emotion else None
        mood_vector = _decay(_vector(mood), at, mood_updated, 21600.0)
        emotion_vector = _decay(_vector(emotion), at, emotion_updated, 720.0)
        return AffectState(
            guild_id,
            actor_id,
            mood_vector,
            emotion_vector,
            int(mood["observations"]) if mood else 0,
            int(emotion["observations"]) if emotion else 0,
            mood_updated,
            emotion_updated,
        )


class AffectEngine:
    def __init__(self, repository: AffectRepository, reactions: SocialReactionRepository) -> None:
        self.repository = repository
        self._reactions = reactions

    async def observe(self, event: Event) -> None:
        reaction = await self._reactions.reaction_for_event(event.event_id)
        if reaction is not None:
            from debilgi.config import thought_content_enabled

            hostile = thought_content_enabled() and bool(hostility_cues(
                str(event.payload.get("content", ""))))
            await self.repository.observe(event, reaction.vector, hostile)


def _advance(previous, impulse, now, updated_at, *, half_life_seconds, learning_rate):
    elapsed = max(0.0, (now - updated_at).total_seconds()) if updated_at else half_life_seconds * 8
    retention = 0.5 ** (elapsed / half_life_seconds)
    return AffectVector(
        **{
            key: _signed(
                getattr(previous, key) * retention * (1.0 - learning_rate)
                + getattr(impulse, key) * learning_rate
            )
            for key in asdict(previous)
        }
    )


def _decay(vector, at, updated_at, half_life_seconds):
    if at is None or updated_at is None:
        return vector
    elapsed = max(0.0, (at - updated_at).total_seconds())
    retention = 0.5 ** (elapsed / half_life_seconds)
    return AffectVector(**{key: getattr(vector, key) * retention for key in asdict(vector)})


def _vector(row) -> AffectVector:
    return (
        AffectVector(**{key: float(row[key]) for key in asdict(AffectVector())})
        if row
        else AffectVector()
    )


def _json(value) -> str:
    return json.dumps(asdict(value), ensure_ascii=False, sort_keys=True)


def _signed(value: float) -> float:
    return max(-1.0, min(1.0, value if math.isfinite(value) else 0.0))


# --- Irritation R (4.34, flag DEBILGI_THOUGHT_CONTENT only) -------------------------------------
# R is a function of the journal and the clock, like boredom B, but it is a separate scale: it
# reads what a person did to the agent (hostile address, pressure by repeating, a repeated
# question after an answer), never what the agent said. Nothing is stored.
#   R = 1 - exp(-g * sum_i w_i * s_i),  w_i = 2^(-age_i / T_R),  g = 1 - patience (genome),
#   T_R = the per-person emotion half-life already used by AffectRepository.state (720 s).
IRRITATION_HALF_LIFE = 720.0
_REJECTIONS = frozenset({"explicit_rejection", "contextual_correction", "refusal",
                         "follow_up_question"})  # (rejected, or pressed again: not satisfied)
_ABUSE_PARTS = frozenset({"noun", "adverb", "interjection", "particle"})
_SECOND_PERSON = frozenset({"ты", "твой", "вы", "ваш"})
_OTHER_SUBJECTS = frozenset({"я", "мы", "он", "она", "оно", "они"})
_MENTION = None


def strip_mentions(text: str) -> str:
    import re

    global _MENTION
    if _MENTION is None:
        _MENTION = re.compile(r"<@[!&]?\d+>")
    return " ".join(_MENTION.sub(" ", text or "").split())


_OTHER_PERSON = frozenset({"я", "мы", "он", "она", "оно", "они"})
_NAMELIKE = frozenset({"Name", "Surn", "Patr", "Geox", "Orgn", "Abbr"})


@lru_cache(maxsize=16384)
def hostility_cues(text: str, aggravated: bool = False,
                   signs: tuple[tuple[str, int], ...] | None = None) -> tuple[str, ...]:
    """Evidence that an utterance addressed to the agent is an attack on it.

    No word list. Evidence kinds (the prefix is the kind, the rest the lemma):
    * ``register:`` — the morphology dictionary marks the word as slang (``Slng``, also when the
      suffix predictor inherits it: ``похуй`` from ``хуй``), and the structure puts it on the
      addressee: ``ты``/``твой`` present, a bare predicate («пошёл …») in the same clause, or an
      animate known noun standing alone as an address («але пидарас»). A guessed register word with
      no addressee («да и похуй»), and a
      sentence about «я/он/она/мы/они» without ``ты``, are not an attack.
    * ``unplaced:`` — «ты + noun» where the noun is one the dictionary does not know and that is
      not written with a capital (a guessed ``Name``/``Orgn`` grammeme is a guess) and not a
      keystroke away from a known word (a typo of ``занимаешься`` is a verb, not a word): the
      speaker characterises the addressee with a word the agent cannot place.
    * ``escalation:`` — the person has been hostile a moment ago (``aggravated``) and now
      predicates a noun (4.42: or an adjective) of unknown sign of the addressee («ты дурак»,
      «ты тупой», «быть петушком?»).
      Its strength is the irritation already in force (see ``irritation_reading``).
    * ``negative_valence_at_addressee`` — «ты + оценка < 0» in a nominal statement.
    A positively valenced word never counts as an attack.

    ``signs`` (4.43, flag only; ``None`` = the earlier reading): what this chat's people showed by
    their reactions (``chat_norms``) — (lemma, +1) a word they take as play, (lemma, −1) one they
    answer with hostility. A friendly word addressed to someone is no attack; a hostile one is an
    attack on its own (``chat:``); a word of unknown sign is not an attack by itself — after a
    complaint it counts only as a statement, a question about the addressee is only a question.
    """
    from debilgi.language.dialogue_learning import exchange_analysis
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    text = strip_mentions(text)
    if not text:
        return ()
    structure = exchange_analysis(text)[0]  # the parse every other reader already holds
    alpha = [t for t in structure.tokens if t.surface[:1].isalpha()]
    second = any(t.lemma in _SECOND_PERSON for t in alpha)
    subjects = {structure.tokens[i].lemma for c in structure.clauses for i in c.subject_indices}
    if (subjects & _OTHER_SUBJECTS or any(t.lemma in _OTHER_PERSON for t in alpha)) and not second:
        return ()
    bare_clauses = {c.clause_index for c in structure.clauses if (
        c.predicate_index is not None and not c.subject_indices) or c.clause_kind == "command"}
    questioned = any(c.clause_kind == "question" for c in structure.clauses)
    negated = any(c.negated for c in structure.clauses)
    # «ты балбес что ли?» (any spelling of the particle, see the tokenizer) asserts what it
    # appears to ask: an accusation, not a question about the addressee (4.46)
    rhetorical = any(left.lemma == "что" and right.lemma == "ли"
                     for left, right in zip(structure.tokens, structure.tokens[1:], strict=False))
    capitalised = {w.casefold() for w in re.findall(r"\w+", text)[1:] if w[:1].isupper()}
    cues = []
    chat_sign = dict(signs) if signs is not None else {}
    for token in alpha:
        readings = DEFAULT_MORPHOLOGY.parse_word(token.surface)
        register = any("Slng" in r.grammemes for r in readings)
        if not register and token.part_of_speech != "noun" and not (
                (aggravated or chat_sign.get(token.lemma, 0) < 0)
                and token.part_of_speech == "adjective"):
            continue
        if (lexical_valence(token.surface) or 0.0) > 0:
            continue
        bare_predicate = token.clause_index in bare_clauses
        if register and token.part_of_speech in _ABUSE_PARTS:
            if token.surface.casefold() in capitalised:
                continue  # a name written with a capital letter is not a word of abuse
            if token.semantic_role == "pragmatic_marker":
                directed = second or bare_predicate
            elif token.part_of_speech == "noun" and token.semantic_role in {
                "predicate_complement", "vocative", "direct_object", "subject",
            }:
                alone = all(t is token or t.semantic_role in {
                    "subject", "vocative", "predicate_complement"} and t.part_of_speech == "noun"
                    or t.part_of_speech != "verb" and t.part_of_speech != "noun" for t in alpha)
                directed = second or bare_predicate or (
                    alone and token.morphology_known and "anim" in token.grammemes)
            else:
                directed = False
            if directed and chat_sign.get(token.lemma, 0) <= 0:
                cues.append("register:" + token.lemma)
            continue
        if token.part_of_speech not in {"noun", "adjective"} or not second or negated:
            continue
        # «ты + noun»: the noun is a predicate complement of ты (or of an implicit addressee)
        clause = structure.clauses[token.clause_index] if token.clause_index < len(
            structure.clauses) else None
        if clause is None or token.index not in clause.predicate_complement_indices or not any(
                structure.tokens[i].lemma in _SECOND_PERSON for i in clause.subject_indices):
            continue
        learned = chat_sign.get(token.lemma, 0)
        if learned > 0:
            continue
        if learned < 0:
            cues.append("chat:" + token.lemma)
        elif (not token.morphology_known and token.surface.casefold() not in capitalised
                and not DEFAULT_MORPHOLOGY.has_known_neighbour(token.surface)):
            cues.append("unplaced:" + token.lemma)
        elif aggravated and lexical_valence(token.surface) is None and not (
                signs is not None and questioned and not rhetorical):
            cues.append("escalation:" + token.lemma)
    if aggravated and not cues and not second:
        # «нравится быть петушком?»: a question with no subject whose complement of «быть» is an
        # instrumental noun; the implicit subject of a bare question is the addressee
        for clause in structure.clauses:
            if clause.clause_kind != "question" and not questioned:
                continue
            for index in clause.action_complement_indices:
                if structure.tokens[index].lemma != "быть":
                    continue
                for token in structure.tokens[index + 1:clause.token_end]:
                    if lexical_valence(token.surface) is not None:
                        continue
                    if any(r.part_of_speech == "noun" and r.case == "instrumental"
                           and not (r.grammemes & _NAMELIKE)
                           for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)):
                        cues.append("escalation:" + token.lemma)
    if (not cues and (lexical_valence(text) or 0.0) < 0 and second and any(
            c.clause_kind == "nominal_statement" and any(
                structure.tokens[i].lemma in _SECOND_PERSON for i in c.subject_indices)
            for c in structure.clauses) and not questioned and not negated):
        cues.append("negative_valence_at_addressee")
    return tuple(cues)


@dataclass(frozen=True, slots=True)
class IrritationReading:
    value: float
    hostile: tuple[str, ...]
    impulses: tuple[dict, ...]
    sensitivity: float
    half_life: float

    def as_dict(self) -> dict:
        return {"value": round(self.value, 4), "hostile_now": list(self.hostile),
                "sensitivity": self.sensitivity, "half_life_seconds": self.half_life,
                "impulses": list(self.impulses)}


async def irritation_reading(
    database: Database, event: Event, text: str, half_life: float = None
) -> IrritationReading | None:
    from datetime import timedelta

    from debilgi.config import thought_content_enabled
    from debilgi.personality.genome import DEBILGI_GENOME

    if not thought_content_enabled() or event.actor_id is None:
        return None
    half_life = half_life or IRRITATION_HALF_LIFE  # (4.42: P reads the same R over T_S)
    patience = DEBILGI_GENOME.traits()["patience"]
    gain = max(1e-3, 1.0 - patience)
    now = event.utc_timestamp
    rows = await database.fetchall(
        """SELECT t.source_event_id, COALESCE(json_extract(e.payload_json, '$.content'), t.prompt)
                  AS prompt, t.status, t.started_at, m.operation,
                  (SELECT c.outcome_type FROM language_meaning_consequences c
                   WHERE c.event_id = t.source_event_id LIMIT 1) AS outcome
           FROM language_response_traces t
           LEFT JOIN events e ON e.event_id = t.source_event_id
           LEFT JOIN language_meaning_intentions m
             ON m.guild_id = t.guild_id AND m.response_message_id = t.response_message_id
           WHERE t.guild_id=? AND t.user_id=? AND t.started_at < ? AND t.started_at >= ?
           ORDER BY t.started_at DESC LIMIT 48""",
        (event.guild_id, event.actor_id, now.isoformat(),
         (now - timedelta(seconds=6 * half_life)).isoformat()))
    turns = [(r["source_event_id"], strip_mentions(r["prompt"]), r["status"], r["operation"],
              (now - datetime.fromisoformat(r["started_at"])).total_seconds(), r["outcome"])
             for r in reversed(rows)]
    # what the person wrote, not the prompt the agent resolved (an ellipsis resolves to the
    # previous question and would look like a repeat)
    turns.append((event.event_id, strip_mentions(event.payload.get("content") or text),
                  None, None, 0.0, None))
    chat_signs = None
    from debilgi.config import thought_content_enabled
    if thought_content_enabled():
        from debilgi.cognition.chat_norms import load as load_norms
        chat_signs = (await load_norms(database, event.guild_id)).signs()
    seen: dict[str, tuple[str | None, str | None]] = {}
    impulses, total, past, current = [], 0.0, [], ()
    dissatisfied = False
    for event_id, prompt, status, operation, age, outcome in turns:
        if not prompt:
            continue
        # (4.42) the person rejected, or pressed again, what the agent had said (within the
        # window of this reading): their next word about the addressee is read as one after
        # hostility («не то» … «ты тупой»)
        was_dissatisfied = dissatisfied
        dissatisfied = dissatisfied or outcome in _REJECTIONS
        key = " ".join("".join(ch for ch in prompt.casefold() if ch.isalnum() or ch == " ").split())
        kinds = {}
        # the irritation this person had already caused at that moment (the evidence for an
        # escalation: someone who was hostile a moment ago, calling the addressee by a word of
        # unknown sign)
        in_force = sum(value * 2.0 ** (-(earlier_age - age) / half_life)
                       for earlier_age, value in past)
        carried = 1.0 - math.exp(-gain * in_force)
        # "still in force": earlier attacks weigh at least as much as one attack a half-life ago
        cues = hostility_cues(prompt, in_force >= 0.5 or was_dissatisfied, chat_signs)
        if event_id == event.event_id:
            current = cues
        direct = [c for c in cues if not c.startswith("escalation:")]
        if direct:
            kinds["hostile_address"] = 1.0
        elif cues:
            kinds["hostile_escalation"] = carried  # the irritation already in force
        if key in seen:
            earlier_status, earlier_operation = seen[key]
            # (4.42) a question put back, or a request for a repair, is no answer to the repeated
            # request: repeating it after that is not repeating after an answer
            answered = earlier_status == "sent" and not (earlier_operation or "").startswith(
                "express_") and (earlier_operation or "") not in {
                "curiosity", "repair", "ask_unbound_variable"} and not (
                earlier_operation or "").startswith("clarify_")
            kinds["repeated_after_answer" if answered else "pressure_after_no_answer"] = (
                1.0 if answered else 0.25)
        seen[key] = (status, operation)
        if not kinds:
            continue
        strength = 1.0
        for value in kinds.values():
            strength *= 1.0 - value
        strength = 1.0 - strength
        weight = 2.0 ** (-age / half_life)
        total += weight * strength
        past.append((age, strength))
        impulses.append({"event_id": event_id, "kinds": sorted(kinds), "weight": round(weight, 4),
                         "strength": round(strength, 4)})
    value = 1.0 - math.exp(-gain * total)
    return IrritationReading(value, current, tuple(impulses[-8:]), gain, half_life)
