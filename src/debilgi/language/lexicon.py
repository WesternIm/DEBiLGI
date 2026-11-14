"""Local reference lexicon, personal meanings, server slang and explicit uncertainty."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from debilgi.events.models import Event, EventType
from debilgi.language.abbreviations import (
    abbreviation_meanings,
    common_abbreviations,
    extract_explicit_abbreviation_definition,
    is_abbreviation_candidate,
)
from debilgi.language.character_markov import (
    CharacterMarkovRepository,
    normalize_word,
    words_from_text,
)
from debilgi.language.content import content_lemmas
from debilgi.language.definition_relations import definition_relations
from debilgi.language.lexical_matching import fuzzy_match_allowed
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, POS_NAMES
from debilgi.language.ruwordnet import LexicalLink, RuWordNet
from debilgi.storage.database import Database

try:  # Optional at runtime; the service retains a deterministic local fallback.
    import pymorphy3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on deployment extras
    pymorphy3 = None

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_EXPLICIT_DEFINITION_RE = re.compile(
    r"(?:^|[\s,;:])[«\"']?([A-Za-zА-Яа-яЁё-]{2,48})[»\"']?\s*"
    r"(?:[—–-]\s*(?:это\s+)?|(?:это|означает|значит)\s+)(.+)$",
    re.IGNORECASE,
)
_INVALID_DEFINITION_SUBJECTS = frozenset({"это", "этот", "эта", "слово", "он", "она", "оно", "они"})
_META_DEFINITION_RE = re.compile(
    r"^(?:(?:да|нет)[,\s]+)?(?:у\s+(?:него|неё|нее)\s+)?"
    r"(?:это\s+)?(?:есть|имеет)?\s*(?:своё|свое|отдельное|"
    r"другое|особое|иное)\s+значение[.!?…]*$",
    re.IGNORECASE,
)
_SEMANTIC_STUDY_TERM_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,47}$")
_NON_DEFINITION_WORDS = frozenset(
    {
        "да",
        "нет",
        "это",
        "есть",
        "иметь",
        "значение",
        "слово",
        "отдельный",
        "другой",
        "особый",
        "иной",
        "свой",
        "просто",
        "такой",
        "он",
        "она",
        "оно",
    }
)
_BUILTIN: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "обманщик": (
        "NOUN",
        ("человек, который намеренно вводит других в заблуждение",),
        ("negative",),
    ),
    "лох": (
        "NOUN",
        ("грубое разговорное название человека, которого считают наивным или неудачливым",),
        ("colloquial", "derogatory"),
    ),
    "лжец": ("NOUN", ("человек, который говорит неправду",), ("negative",)),
    "умный": (
        "ADJF",
        ("способный хорошо понимать, рассуждать и находить решения",),
        (),
    ),
    "глупый": (
        "ADJF",
        ("плохо понимающий или совершающий неразумные действия",),
        ("negative",),
    ),
    "лук": ("NOUN", ("оружие для стрельбы стрелами", "огородное растение"), ()),
    "палка": ("NOUN", ("длинный прямой кусок дерева",), ()),
    "найти": ("VERB", ("обнаружить в результате поиска",), ()),
    "словарь": ("NOUN", ("собрание слов с пояснениями",), ()),
    "друг": ("NOUN", ("человек, связанный с кем-либо дружбой",), ()),
}
_WEAPON_CONTEXT = {
    "стрела",
    "стрелу",
    "стрелы",
    "стрелять",
    "оружие",
    "тетива",
    "тетиву",
    "выстрел",
    "колчан",
}
_FOOD_CONTEXT = {"еда", "есть", "салат", "овощ", "суп", "резать", "растение"}


class UnknownKind(StrEnum):
    UNKNOWN_FORM = "unknown_form"
    UNKNOWN_MEANING = "unknown_meaning"
    AMBIGUOUS_CONTEXT = "ambiguous_context"
    APPROXIMATE_MEANING = "approximate_meaning"
    LOW_MEMORY_CONFIDENCE = "low_memory_confidence"
    EXPRESSION_GAP = "expression_gap"


@dataclass(frozen=True, slots=True)
class ConfidenceProfile:
    form: float
    meaning: float
    context_fit: float
    memory: float
    response: float


@dataclass(frozen=True, slots=True)
class WordAnalysis:
    surface_form: str
    normalized_form: str
    lemma: str | None
    part_of_speech: str | None
    grammatical_features: dict[str, str]
    dictionary_meanings: tuple[str, ...]
    style_labels: tuple[str, ...]
    source: str
    confidence: ConfidenceProfile
    selected_meaning: str | None = None
    personal_interpretation: str | None = None
    local_meaning: str | None = None
    associations: tuple[str, ...] = ()
    unknown_kind: UnknownKind | None = None
    is_neologism: bool = False
    semantic_relations: tuple[tuple[str, str], ...] = ()
    sense_id: str | None = None
    sense_candidates: tuple[str, ...] = ()
    sense_options: tuple[tuple[str, str], ...] = ()
    relation_evidence: tuple[str, ...] = ()
    typed_relations: tuple[LexicalLink, ...] = ()

    @property
    def is_dictionary_word(self) -> bool:
        return bool(
            self.lemma
            and self.source
            in {
                "builtin",
                "opencorpora",
                "opencorpora_spelling",
                "dictionary_cache",
                "ru_wiktionary",
                "ru_wiktionary_derived",
                "ruwordnet",
                "builtin_abbreviation",
            }
        )

    @property
    def response_confidence(self) -> float:
        return self.confidence.response

    @property
    def has_usable_meaning(self) -> bool:
        """Whether generation has grounded semantics instead of only a known form."""

        return bool(
            self.selected_meaning
            or self.personal_interpretation
            or self.local_meaning
            or self.dictionary_meanings
        )

    @property
    def requires_lexical_study(self) -> bool:
        """Reserve the unknown-word queue for genuine lexical gaps.

        Ambiguity and low confidence are reasoning problems, not proof that the
        word is unknown.  In particular, a dictionary-backed ambiguous word
        must not be reported to the user as something the bot does not know.
        """

        return (
            self.unknown_kind
            in {UnknownKind.UNKNOWN_FORM, UnknownKind.UNKNOWN_MEANING}
            and not self.has_usable_meaning
        )

    def uncertainty_message(self) -> str | None:
        messages = {
            UnknownKind.UNKNOWN_FORM: "Я не знаю этого слова.",
            UnknownKind.UNKNOWN_MEANING: "Я знаю это слово, но не понимаю, что оно значит здесь.",
            UnknownKind.AMBIGUOUS_CONTEXT: (
                "У него несколько значений, я не уверен, какое используется."
            ),
            UnknownKind.APPROXIMATE_MEANING: "Кажется, я понимаю, но пока не уверен.",
            UnknownKind.LOW_MEMORY_CONFIDENCE: "Я это встречал, но помню значение ненадёжно.",
            UnknownKind.EXPRESSION_GAP: "Я понимаю идею, но пока не могу нормально её выразить.",
        }
        return messages.get(self.unknown_kind)


@dataclass(frozen=True, slots=True)
class UnknownConcept:
    concept_key: str
    unknown_kind: UnknownKind
    occurrence_count: int
    curiosity_score: float
    last_question_at: datetime | None
    question_count: int


@dataclass(frozen=True, slots=True)
class NeologismCandidate:
    candidate_id: str
    word: str
    intended_meaning: str
    status: str
    confidence: float
    usage_count: int
    human_repeat_count: int
    positive_reaction_count: int
    negative_reaction_count: int


@dataclass(frozen=True, slots=True)
class DictionarySense:
    sense_id: str
    lemma: str
    part_of_speech: str | None
    definition: str
    context_terms: tuple[str, ...]
    examples: tuple[str, ...]
    source: str
    confidence: float
    homonym_group: str | None = None


@dataclass(frozen=True, slots=True)
class MeaningComparison:
    word: str
    lemma: str | None
    reference_meaning: str | None
    personal_meaning: str | None
    local_meaning: str | None
    relation: str
    confidence: float


def is_substantive_definition(concept: str, explanation: str) -> bool:
    """Reject acknowledgements about a meaning that do not state the meaning itself."""

    cleaned = " ".join(explanation.split()).strip()
    if not cleaned or cleaned.rstrip().endswith("?") or _META_DEFINITION_RE.fullmatch(cleaned):
        return False
    concept_terms = set(content_lemmas(concept)) | {normalize_word(concept)}
    informative = {
        term
        for term in content_lemmas(cleaned)
        if term not in concept_terms and term not in _NON_DEFINITION_WORDS
    }
    return bool(informative)


def extract_explicit_definition(content: str) -> tuple[str, str] | None:
    """Extract a grounded ``word — definition`` lesson from ordinary chat text."""

    cleaned = " ".join(content.split()).strip()
    for match in _EXPLICIT_DEFINITION_RE.finditer(cleaned):
        concept = normalize_word(match.group(1))
        definition = match.group(2).strip(" ,;:.—–-")
        if concept in _INVALID_DEFINITION_SUBJECTS:
            continue
        reading = DEFAULT_MORPHOLOGY.best(concept)
        if (reading is not None and reading.is_known
                and reading.part_of_speech not in {"noun", "adjective", "verb"}):
            continue  # a connective or pronoun in a sentence is not a headword lesson
        if is_substantive_definition(concept, definition):
            return concept, definition[:1000]
    return None


class LexiconService:
    """Resolve meanings using conversation/local/personal/reference priority."""

    def __init__(
        self,
        database: Database,
        *,
        unknown_word_threshold: float = 0.45,
        curiosity_threshold: float = 0.68,
        curiosity_question_cooldown: timedelta = timedelta(minutes=30),
        curiosity_max_consecutive_questions: int = 1,
        ruwordnet: RuWordNet | None = None,
    ) -> None:
        self._database = database
        self._ruwordnet = ruwordnet
        self.unknown_word_threshold = unknown_word_threshold
        self.curiosity_threshold = curiosity_threshold
        self.curiosity_question_cooldown = curiosity_question_cooldown
        self.curiosity_max_consecutive_questions = curiosity_max_consecutive_questions
        self._morph = pymorphy3.MorphAnalyzer() if pymorphy3 is not None else None
        self._study_notifier: Callable[[], None] | None = None

    def set_study_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Wake the autonomous learner whenever durable study work is queued."""

        self._study_notifier = notifier

    def set_semantic_study_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Compatibility alias for callers configured before lexical wakeups."""

        self.set_study_notifier(notifier)

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        if event.payload.get("suppress_full_parse") is True:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        observed_at = event.utc_timestamp.isoformat()
        abbreviation_definition = extract_explicit_abbreviation_definition(content)
        explicit_definition = (
            (
                abbreviation_definition[0],
                f"сокращение от «{abbreviation_definition[1]}»",
            )
            if abbreviation_definition is not None
            else extract_explicit_definition(content)
        )
        if explicit_definition is not None:
            concept, definition = explicit_definition
            await self.teach_personal_interpretation(
                guild_id=event.guild_id,
                word=concept,
                interpretation=definition,
                associations=tuple(content_lemmas(definition))[:24],
                confidence=0.72,
                source_event_ids=(event.event_id,),
                now=event.utc_timestamp,
            )
            if event.channel_id is not None:
                await self.resolve_unknown(event.guild_id, event.channel_id, concept)
        for word in dict.fromkeys(words_from_text(content)):
            analysis = await self.analyze_word(
                word,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                context=content,
            )
            await self._remember_personal(event, analysis, observed_at)
            if analysis.requires_lexical_study:
                await self._record_unknown(event, analysis, observed_at)

    async def analyze_word(
        self,
        word: str,
        *,
        guild_id: int = 0,
        channel_id: int | None = None,
        context: str | None = None,
    ) -> WordAnalysis:
        surface = word
        original_normalized = (
            re.sub(r"[\s._-]+", "", normalize_word(word))
            if is_abbreviation_candidate(word)
            else normalize_word(word)
        )
        normalized = original_normalized
        morphological = DEFAULT_MORPHOLOGY.best(original_normalized)
        if morphological is not None and morphological.corrected_form:
            normalized = morphological.corrected_form
        slang = await self._slang(guild_id, channel_id, normalized) if guild_id else None
        personal = await self._personal(guild_id, normalized) if guild_id else None
        neologism = await self._neologism(guild_id, normalized) if guild_id else None
        reference = await self._cached_reference(normalized, context=context)
        if reference is None:
            reference = self._reference_analysis(original_normalized, context=context)
        if guild_id and (personal is None or not personal["explicitly_taught"]):
            lemma = reference[0] if reference is not None else normalized
            lesson = await self._database.fetchone(
                """SELECT * FROM language_personal_interpretations
                   WHERE guild_id = ? AND explicitly_taught = 1
                     AND (normalized_form = ? OR lemma = ?)
                   ORDER BY updated_at DESC LIMIT 1""", (guild_id, lemma, lemma),
            )
            if lesson is not None:
                personal = lesson
        if reference is None or not reference[3]:
            # A known morphological form is not yet a known meaning. Use only an
            # exact headword with a definition-backed graph, never partial aliases.
            concept = await self._database.fetchone(
                """SELECT c.definition, c.source, c.confidence FROM knowledge_concepts c
                   WHERE c.normalized_title = ?
                     AND c.source IN ('wikipedia', 'ru_wiktionary', 'ru_wiktionary_derived')
                     AND EXISTS (SELECT 1 FROM knowledge_definition_relations r
                                 WHERE r.concept_id = c.concept_id)
                   ORDER BY CASE WHEN c.source = 'wikipedia' THEN 1 ELSE 0 END,
                            c.confidence DESC LIMIT 1""",
                ((reference[0] if reference is not None else normalized).replace("ё", "е"),),
            )
            if concept is not None:
                if reference is None:
                    reference = (normalized, None, {}, (), (), str(concept["source"]),
                                 float(concept["confidence"]))
                reference = (
                    *reference[:3], (str(concept["definition"]),), reference[4],
                    str(concept["source"]), min(reference[6], float(concept["confidence"])),
                )

        if slang is not None:
            meaning = str(slang["meaning"])
            score = float(slang["confidence"])
            reference_lemma = reference[0] if reference is not None else normalized
            reference_part = reference[1] if reference is not None else None
            reference_features = reference[2] if reference is not None else {}
            reference_meanings = reference[3] if reference is not None else ()
            reference_labels = reference[4] if reference is not None else ()
            slang_labels = tuple(json.loads(str(slang["style_labels_json"])))
            return WordAnalysis(
                surface,
                normalized,
                str(slang["lemma"] or reference_lemma),
                reference_part,
                reference_features,
                reference_meanings,
                tuple(dict.fromkeys((*slang_labels, *reference_labels))),
                "server_slang",
                ConfidenceProfile(0.95, score, 0.95, score, min(0.95, score)),
                selected_meaning=meaning,
                local_meaning=meaning,
                personal_interpretation=(
                    str(personal["interpretation"])
                    if personal and personal["interpretation"]
                    else None
                ),
                associations=self._associations(personal),
                is_neologism=neologism is not None,
            )

        if (personal is not None and personal["explicitly_taught"] and personal["interpretation"]
                and not is_abbreviation_candidate(surface)):
            meaning = str(personal["interpretation"])
            lemma = reference[0] if reference else str(personal["lemma"] or normalized)
            part = reference[1] if reference else None
            features = reference[2] if reference else {}
            memory = float(personal["confidence"])
            return WordAnalysis(
                surface, normalized, lemma, part, features,
                reference[3] if reference else (), ("learned_in_conversation",),
                "personal_lexicon", ConfidenceProfile(0.9, memory, 0.85, memory, memory),
                selected_meaning=meaning, personal_interpretation=meaning,
                associations=self._associations(personal),
                semantic_relations=tuple((edge.predicate, edge.target)
                                         for edge in definition_relations(lemma, meaning)),
            )

        if personal is not None and is_abbreviation_candidate(surface):
            interpretation = (
                str(personal["interpretation"]) if personal["interpretation"] else None
            )
            memory = float(personal["confidence"])
            return WordAnalysis(
                surface,
                normalized,
                normalized,
                "NOUN",
                {"abbreviation": "yes"},
                (interpretation,) if interpretation else (),
                ("abbreviation", "learned_in_conversation"),
                "personal_abbreviation",
                ConfidenceProfile(0.96, memory, 0.9, memory, min(0.94, memory)),
                selected_meaning=interpretation,
                personal_interpretation=interpretation,
                associations=self._associations(personal),
                unknown_kind=(
                    UnknownKind.UNKNOWN_MEANING if interpretation is None else None
                ),
                is_neologism=neologism is not None,
            )

        dotted_expansions = abbreviation_meanings(surface) if "." in surface else ()
        if dotted_expansions:
            meanings = tuple(
                f"сокращение от «{expansion}»" for expansion in dotted_expansions
            )
            return WordAnalysis(
                surface,
                normalized,
                normalized,
                "NOUN",
                {"abbreviation": "yes"},
                meanings,
                ("abbreviation",),
                "builtin_abbreviation",
                ConfidenceProfile(0.98, 0.96, 0.9, 0.9, 0.9),
                selected_meaning=meanings[0],
            )

        if reference is not None:
            lemma, part, features, meanings, labels, source, form_confidence = reference
            senses = await self._senses_for_lemma(lemma)
            if (self._ruwordnet is not None and not senses and part
                    and (not meanings or source in {
                        "builtin", "opencorpora", "opencorpora_spelling",
                    })):
                selection = await asyncio.to_thread(
                    self._ruwordnet.lookup, lemma, part, context or "",
                )
                if selection.senses:
                    chosen = selection.selected
                    links = ()
                    if chosen is not None:
                        links = (*chosen.links, *(
                            LexicalLink("synonym", chosen.synset_id, word, (word,))
                            for word in chosen.synonyms if word != lemma
                        ))
                    semantic_links = tuple(dict.fromkeys((
                        *((link.relation, link.target) for link in links),
                        *((("synonym", word) for word in chosen.synonyms if word != lemma)
                          if chosen is not None else ()),
                    )))
                    return WordAnalysis(
                        surface, normalized, lemma, part, features,
                        (chosen.meaning,) if chosen else (), labels, "ruwordnet",
                        ConfidenceProfile(form_confidence, 0.86, 0.86 if chosen else 0.4,
                                          0.8, min(form_confidence, 0.86 if chosen else 0.4)),
                        selected_meaning=chosen.meaning if chosen else None,
                        personal_interpretation=(
                            str(personal["interpretation"])
                            if personal and personal["interpretation"] else None
                        ),
                        associations=self._associations(personal),
                        is_neologism=neologism is not None,
                        unknown_kind=None if chosen else UnknownKind.AMBIGUOUS_CONTEXT,
                        semantic_relations=semantic_links,
                        sense_id=chosen.synset_id if chosen else None,
                        sense_candidates=tuple(s.synset_id for s in selection.senses),
                        sense_options=tuple((s.synset_id, s.title) for s in selection.senses),
                        typed_relations=links,
                        relation_evidence=(*selection.evidence, *(
                            f"ruwordnet:{chosen.synset_id}:{link.relation}:{link.target_id}"
                            for link in links
                        )),
                    )
            if part:
                senses = tuple(
                    sense for sense in senses
                    if not sense.part_of_speech
                    or POS_NAMES.get(sense.part_of_speech, sense.part_of_speech)
                    == POS_NAMES.get(part, part)
                )
            if senses:
                meanings = tuple(item.definition for item in senses)
                resolved = self._resolve_sense_from_context(senses, context or "")
                selected = resolved.definition if resolved else None
                context_fit = resolved.confidence if resolved else 0.42
            else:
                selected, context_fit = _select_meaning(lemma, meanings, context or "")
            ambiguous = len(meanings) > 1 and selected is None
            meaning_confidence = 0.82 if meanings else 0.48
            response = min(form_confidence, meaning_confidence, context_fit)
            return WordAnalysis(
                surface,
                normalized,
                lemma,
                part,
                features,
                meanings,
                labels,
                source,
                ConfidenceProfile(form_confidence, meaning_confidence, context_fit, 0.75, response),
                selected_meaning=selected,
                personal_interpretation=(
                    str(personal["interpretation"])
                    if personal and personal["interpretation"]
                    else None
                ),
                associations=self._associations(personal),
                unknown_kind=(UnknownKind.AMBIGUOUS_CONTEXT if ambiguous else None),
                is_neologism=neologism is not None,
                semantic_relations=tuple(
                    (edge.predicate, edge.target)
                    for edge in definition_relations(lemma, selected)
                ) if selected is not None else (),
            )

        if personal is not None:
            memory = float(personal["confidence"])
            interpretation = str(personal["interpretation"]) if personal["interpretation"] else None
            kind = UnknownKind.UNKNOWN_MEANING if not interpretation else None
            return WordAnalysis(
                surface,
                normalized,
                str(personal["lemma"] or normalized),
                None,
                {},
                (),
                (),
                "personal_lexicon",
                ConfidenceProfile(0.72, memory, 0.55, memory, memory * 0.72),
                selected_meaning=interpretation,
                personal_interpretation=interpretation,
                associations=self._associations(personal),
                unknown_kind=kind,
                is_neologism=neologism is not None,
            )

        return WordAnalysis(
            surface,
            normalized,
            None,
            None,
            {},
            (),
            (),
            "unknown",
            ConfidenceProfile(0.12, 0.0, 0.0, 0.0, 0.0),
            unknown_kind=UnknownKind.UNKNOWN_FORM,
            is_neologism=neologism is not None,
        )

    async def lookup_lemma(self, word: str) -> str | None:
        return (await self.analyze_word(word)).lemma

    async def import_dictionary_entry(
        self,
        *,
        lemma: str,
        part_of_speech: str | None,
        meanings: tuple[str, ...],
        forms: dict[str, dict[str, str]] | None = None,
        style_labels: tuple[str, ...] = (),
        source: str = "local_import",
        confidence: float = 0.9,
    ) -> None:
        """Import a local record; analysis never makes a per-message network request."""

        normalized_lemma = normalize_word(lemma)
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_dictionary_entries (
                    lemma, part_of_speech, meanings_json, style_labels_json,
                    source, confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (lemma) DO UPDATE SET
                    part_of_speech = CASE
                        WHEN excluded.confidence >= confidence
                        THEN excluded.part_of_speech ELSE part_of_speech END,
                    meanings_json = CASE
                        WHEN excluded.confidence >= confidence
                        THEN excluded.meanings_json ELSE meanings_json END,
                    style_labels_json = CASE
                        WHEN excluded.confidence >= confidence
                        THEN excluded.style_labels_json ELSE style_labels_json END,
                    source = CASE
                        WHEN excluded.confidence >= confidence
                        THEN excluded.source ELSE source END,
                    confidence = MAX(confidence, excluded.confidence),
                    updated_at = CURRENT_TIMESTAMP""",
                (
                    normalized_lemma,
                    part_of_speech,
                    json.dumps(meanings, ensure_ascii=False),
                    json.dumps(style_labels, ensure_ascii=False),
                    source,
                    max(0.0, min(1.0, confidence)),
                ),
            )
            all_forms = forms or {normalized_lemma: {}}
            await connection.executemany(
                """INSERT INTO language_dictionary_forms (
                    surface_form, lemma, grammatical_features_json
                ) VALUES (?, ?, ?)
                ON CONFLICT (surface_form, lemma) DO UPDATE SET
                    grammatical_features_json = excluded.grammatical_features_json""",
                (
                    (
                        normalize_word(surface),
                        normalized_lemma,
                        json.dumps(features, ensure_ascii=False),
                    )
                    for surface, features in all_forms.items()
                ),
            )

    async def lookup_forms(self, lemma: str) -> tuple[str, ...]:
        normalized = normalize_word(lemma)
        rows = await self._database.fetchall(
            "SELECT surface_form FROM language_dictionary_forms WHERE lemma = ?",
            (normalized,),
        )
        if rows:
            return tuple(str(row["surface_form"]) for row in rows)
        if self._morph is None:
            return (normalized,) if normalized in _BUILTIN else ()
        parsed = self._morph.parse(normalized)
        selected = next(
            (item for item in parsed if str(item.normal_form) == normalized),
            parsed[0] if parsed else None,
        )
        return tuple(sorted({form.word for form in selected.lexeme})) if selected else ()

    async def import_sense(
        self,
        *,
        lemma: str,
        definition: str,
        context_terms: tuple[str, ...],
        examples: tuple[str, ...] = (),
        synonyms: tuple[str, ...] = (),
        homonym_group: str | None = None,
        part_of_speech: str | None = None,
        source: str = "local_curriculum",
        confidence: float = 0.92,
        guild_id: int = 0,
    ) -> DictionarySense:
        """Install one context-addressable meaning and its symmetric synonym links."""

        normalized_lemma = normalize_word(lemma)
        normalized_terms = tuple(
            dict.fromkeys(normalize_word(term) for term in context_terms if normalize_word(term))
        )
        normalized_synonyms = tuple(
            dict.fromkeys(
                normalize_word(item)
                for item in synonyms
                if normalize_word(item) and normalize_word(item) != normalized_lemma
            )
        )
        sense_id = f"{normalized_lemma}:{uuid4().hex[:16]}"
        existing = await self._database.fetchone(
            "SELECT sense_id FROM language_dictionary_senses WHERE lemma = ? AND definition = ?",
            (normalized_lemma, definition),
        )
        if existing is not None:
            sense_id = str(existing["sense_id"])
        score = max(0.0, min(1.0, confidence))
        now = datetime.now(UTC).isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_dictionary_entries (
                    lemma, part_of_speech, meanings_json, style_labels_json,
                    source, confidence, updated_at
                ) VALUES (?, ?, ?, '[]', ?, ?, ?)
                ON CONFLICT (lemma) DO UPDATE SET
                    part_of_speech = COALESCE(part_of_speech, excluded.part_of_speech),
                    confidence = MAX(confidence, excluded.confidence),
                    updated_at = excluded.updated_at""",
                (
                    normalized_lemma,
                    part_of_speech,
                    json.dumps((definition,), ensure_ascii=False),
                    source,
                    score,
                    now,
                ),
            )
            await connection.execute(
                """INSERT INTO language_dictionary_forms (
                    surface_form, lemma, grammatical_features_json
                ) VALUES (?, ?, '{}') ON CONFLICT (surface_form, lemma) DO NOTHING""",
                (normalized_lemma, normalized_lemma),
            )
            await connection.execute(
                """INSERT INTO language_dictionary_senses (
                    sense_id, lemma, part_of_speech, definition, context_terms_json,
                    examples_json, source, confidence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (sense_id) DO UPDATE SET
                    part_of_speech = excluded.part_of_speech,
                    context_terms_json = excluded.context_terms_json,
                    examples_json = excluded.examples_json,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at""",
                (
                    sense_id,
                    normalized_lemma,
                    part_of_speech,
                    definition,
                    json.dumps(normalized_terms, ensure_ascii=False),
                    json.dumps(examples, ensure_ascii=False),
                    source,
                    score,
                    now,
                ),
            )
            if homonym_group:
                await connection.execute(
                    """INSERT INTO language_homonym_groups (
                        group_key, lemma, sense_id, discriminator_terms_json
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT (group_key, sense_id) DO UPDATE SET
                        discriminator_terms_json = excluded.discriminator_terms_json""",
                    (
                        normalize_word(homonym_group),
                        normalized_lemma,
                        sense_id,
                        json.dumps(normalized_terms, ensure_ascii=False),
                    ),
                )
            for synonym in normalized_synonyms:
                await connection.executemany(
                    """INSERT INTO language_synonym_links (
                        guild_id, lemma, synonym_lemma, sense_id, style_label,
                        weight, source, updated_at
                    ) VALUES (?, ?, ?, ?, 'neutral', ?, ?, ?)
                    ON CONFLICT (guild_id, lemma, synonym_lemma, sense_id) DO UPDATE SET
                        weight = MAX(weight, excluded.weight),
                        source = excluded.source,
                        updated_at = excluded.updated_at""",
                    (
                        (guild_id, normalized_lemma, synonym, sense_id, score, source, now),
                        (guild_id, synonym, normalized_lemma, sense_id, score, source, now),
                    ),
                )
        return DictionarySense(
            sense_id,
            normalized_lemma,
            part_of_speech,
            definition,
            normalized_terms,
            examples,
            source,
            score,
            normalize_word(homonym_group) if homonym_group else None,
        )

    async def lookup_senses(self, word: str) -> tuple[DictionarySense, ...]:
        normalized = normalize_word(word)
        reference = await self._cached_reference(normalized)
        if reference is None:
            reference = self._reference_analysis(normalized)
        lemma = reference[0] if reference else normalized
        return await self._senses_for_lemma(lemma)

    async def resolve_sense(self, word: str, context: str) -> DictionarySense | None:
        return self._resolve_sense_from_context(await self.lookup_senses(word), context)

    async def lookup_synonyms(
        self, word: str, *, guild_id: int = 0, limit: int = 16
    ) -> tuple[str, ...]:
        normalized = normalize_word(word)
        reference = await self._cached_reference(normalized)
        if reference is None:
            reference = self._reference_analysis(normalized)
        lemma = reference[0] if reference else normalized
        rows = await self._database.fetchall(
            """SELECT synonym_lemma, MAX(weight) AS best_weight
               FROM language_synonym_links
               WHERE guild_id IN (0, ?) AND lemma = ?
               GROUP BY synonym_lemma
               ORDER BY best_weight DESC, synonym_lemma LIMIT ?""",
            (guild_id, lemma, max(1, min(limit, 100))),
        )
        return tuple(str(row["synonym_lemma"]) for row in rows)

    async def semantic_terms(self, text: str, *, guild_id: int = 0) -> tuple[str, ...]:
        """Expand surface words to lemmas and learned synonyms for semantic retrieval."""

        terms: list[str] = []
        for word in content_lemmas(text)[:32]:
            normalized = normalize_word(word)
            reference = await self._cached_reference(normalized)
            if reference is None:
                reference = self._reference_analysis(normalized)
            lemma = reference[0] if reference else normalized
            terms.extend((normalized, lemma))
            terms.extend(await self.lookup_synonyms(lemma, guild_id=guild_id, limit=6))
        return tuple(dict.fromkeys(item for item in terms if item))

    async def compare_with_reference(
        self,
        word: str,
        *,
        guild_id: int,
        channel_id: int | None = None,
        context: str = "",
    ) -> MeaningComparison:
        analysis = await self.analyze_word(
            word, guild_id=guild_id, channel_id=channel_id, context=context
        )
        reference_analysis = await self.analyze_word(word, context=context)
        reference = reference_analysis.selected_meaning
        if reference is None and len(reference_analysis.dictionary_meanings) == 1:
            reference = reference_analysis.dictionary_meanings[0]
        candidates = tuple(
            value for value in (analysis.personal_interpretation, analysis.local_meaning) if value
        )
        if not candidates:
            relation = "reference_only" if reference else "unknown"
        elif reference and any(_meaning_overlap(reference, value) >= 0.5 for value in candidates):
            relation = "compatible"
        elif reference:
            relation = "different"
        else:
            relation = "local_only"
        return MeaningComparison(
            word=analysis.normalized_form,
            lemma=analysis.lemma,
            reference_meaning=reference,
            personal_meaning=analysis.personal_interpretation,
            local_meaning=analysis.local_meaning,
            relation=relation,
            confidence=analysis.response_confidence,
        )

    async def lookup_meanings(self, word: str) -> tuple[str, ...]:
        return (await self.analyze_word(word)).dictionary_meanings

    async def lookup_style_labels(self, word: str) -> tuple[str, ...]:
        return (await self.analyze_word(word)).style_labels

    async def is_dictionary_word(self, word: str) -> bool:
        return (await self.analyze_word(word)).is_dictionary_word

    async def find_similar_words(self, word: str, limit: int = 5) -> tuple[str, ...]:
        from difflib import get_close_matches

        normalized = normalize_word(word)
        exact = await self._cached_reference(normalized)
        if exact is not None or (self._morph is not None and self._morph.word_is_known(normalized)):
            return ()
        rows = await self._database.fetchall(
            "SELECT lemma FROM language_dictionary_entries ORDER BY lemma LIMIT 20000"
        )
        choices = set(_BUILTIN) | set(common_abbreviations()) | {
            str(row["lemma"]) for row in rows
        }
        corrected = DEFAULT_MORPHOLOGY.best(normalized)
        standard_candidate = corrected.corrected_form if corrected is not None else None
        matches = get_close_matches(
            normalized,
            sorted(
                candidate for candidate in choices if fuzzy_match_allowed(normalized, candidate)
            ),
            n=max(1, limit),
            cutoff=0.72,
        )
        return tuple(
            dict.fromkeys(
                candidate
                for candidate in (standard_candidate, *matches)
                if candidate is not None and fuzzy_match_allowed(normalized, candidate)
            )
        )[: max(1, limit)]

    async def personal_words(self, guild_id: int, limit: int = 10_000) -> tuple[str, ...]:
        rows = await self._database.fetchall(
            """SELECT normalized_form FROM language_personal_interpretations
               WHERE guild_id = ? ORDER BY occurrence_count DESC LIMIT ?""",
            (guild_id, max(1, min(limit, 50_000))),
        )
        return tuple(str(row["normalized_form"]) for row in rows)

    async def is_rejected_word(self, guild_id: int, word: str) -> bool:
        row = await self._database.fetchone(
            """SELECT 1 FROM language_rejected_word_forms
               WHERE guild_id = ? AND normalized_form = ?""",
            (guild_id, normalize_word(word)),
        )
        return row is not None

    async def register_server_slang(
        self,
        *,
        guild_id: int,
        word: str,
        meaning: str,
        channel_id: int | None = None,
        confidence: float = 0.65,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await self._database.execute(
            """
            INSERT INTO language_server_slang (
                guild_id, channel_id, normalized_form, meaning, confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (guild_id, channel_id, normalized_form) DO UPDATE SET
                meaning = excluded.meaning,
                occurrence_count = occurrence_count + 1,
                confidence = MAX(confidence, excluded.confidence),
                updated_at = excluded.updated_at
            """,
            (guild_id, channel_id or 0, normalize_word(word), meaning, confidence, now, now),
        )

    async def teach_personal_interpretation(
        self,
        *,
        guild_id: int,
        word: str,
        interpretation: str,
        associations: tuple[str, ...] = (),
        confidence: float = 0.65,
        source_event_ids: tuple[str, ...] = (),
        now: datetime | None = None,
    ) -> None:
        """Install a grounded lesson without conflating it with the reference dictionary."""

        moment = (now or datetime.now(UTC)).isoformat()
        analysis = await self.analyze_word(word)
        await self._database.execute(
            """INSERT INTO language_personal_interpretations (
                guild_id, normalized_form, lemma, interpretation, associations_json,
                supporting_event_ids_json, confidence, first_seen_at, updated_at, explicitly_taught
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT (guild_id, normalized_form) DO UPDATE SET
                lemma = COALESCE(excluded.lemma, lemma),
                interpretation = excluded.interpretation,
                explicitly_taught = 1,
                associations_json = excluded.associations_json,
                supporting_event_ids_json = excluded.supporting_event_ids_json,
                occurrence_count = occurrence_count + 1,
                confidence = MAX(confidence, excluded.confidence),
                updated_at = excluded.updated_at""",
            (
                guild_id,
                normalize_word(word),
                analysis.lemma,
                interpretation[:1000],
                json.dumps(tuple(dict.fromkeys(associations))[:24], ensure_ascii=False),
                json.dumps(source_event_ids[:20], ensure_ascii=False),
                max(0.0, min(1.0, confidence)),
                moment,
                moment,
            ),
        )

    async def top_unknowns(
        self, guild_id: int, *, channel_id: int | None = None, limit: int = 10
    ) -> tuple[UnknownConcept, ...]:
        condition = "guild_id = ? AND resolved = 0"
        params: list[Any] = [guild_id]
        if channel_id is not None:
            condition += " AND channel_id IN (0, ?)"
            params.append(channel_id)
        params.append(max(1, min(limit, 50)))
        rows = await self._database.fetchall(
            f"""SELECT * FROM language_unknown_concepts WHERE {condition}
                ORDER BY curiosity_score DESC, occurrence_count DESC, last_seen_at DESC LIMIT ?""",
            params,
        )
        return tuple(
            UnknownConcept(
                concept_key=str(row["concept_key"]),
                unknown_kind=UnknownKind(str(row["unknown_kind"])),
                occurrence_count=int(row["occurrence_count"]),
                curiosity_score=float(row["curiosity_score"]),
                last_question_at=(
                    datetime.fromisoformat(str(row["last_question_at"]))
                    if row["last_question_at"]
                    else None
                ),
                question_count=int(row["question_count"]),
            )
            for row in rows
        )

    async def next_curiosity_question(
        self, guild_id: int, channel_id: int, *, now: datetime | None = None
    ) -> str | None:
        moment = now or datetime.now(UTC)
        if await self._consecutive_question_limit_reached(guild_id, channel_id):
            return None
        rows = await self._database.fetchall(
            """SELECT * FROM language_unknown_concepts
               WHERE guild_id = ? AND channel_id IN (0, ?) AND resolved = 0
                 AND study_status = 'exhausted' AND occurrence_count >= 2
                 AND curiosity_score >= ?
               ORDER BY curiosity_score DESC, occurrence_count DESC, last_seen_at DESC
               LIMIT 5""",
            (guild_id, channel_id, self.curiosity_threshold),
        )
        for row in rows:
            last_question = (
                datetime.fromisoformat(str(row["last_question_at"]))
                if row["last_question_at"]
                else None
            )
            if last_question and moment - last_question < self.curiosity_question_cooldown:
                continue
            concept = str(row["concept_key"])
            await self._database.execute(
                """UPDATE language_unknown_concepts
                   SET last_question_at = ?, question_count = question_count + 1
                   WHERE guild_id = ? AND channel_id IN (0, ?) AND concept_key = ?""",
                (moment.isoformat(), guild_id, channel_id, concept),
            )
            return (
                f"Я сам несколько раз искал значение «{concept}», но эталоны ничего "
                "надёжного не дали. Если это локальное выражение, покажи его в целой фразе."
            )
        return None

    async def _consecutive_question_limit_reached(self, guild_id: int, channel_id: int) -> bool:
        session = await self._database.fetchone(
            """SELECT conversation_id FROM language_conversation_sessions
               WHERE guild_id = ? AND channel_id = ? AND status = 'active'""",
            (guild_id, channel_id),
        )
        if session is None:
            return False
        rows = await self._database.fetchall(
            """SELECT message_intent FROM language_response_plans
               WHERE conversation_id = ? AND status = 'used'
               ORDER BY created_at DESC LIMIT ?""",
            (
                str(session["conversation_id"]),
                self.curiosity_max_consecutive_questions,
            ),
        )
        return len(rows) >= self.curiosity_max_consecutive_questions and all(
            str(row["message_intent"]) == "clarifying_question" for row in rows
        )

    async def cleanup(self, before: datetime) -> int:
        """Remove resolved gaps and expired rejection guards after their retention window."""

        cutoff = before.astimezone(UTC).isoformat()
        async with self._database.transaction() as connection:
            unknown_cursor = await connection.execute(
                """DELETE FROM language_unknown_concepts
                   WHERE resolved = 1 AND last_seen_at < ?""",
                (cutoff,),
            )
            rejected_cursor = await connection.execute(
                """DELETE FROM language_rejected_word_forms
                   WHERE expires_at IS NOT NULL AND expires_at < ?""",
                (cutoff,),
            )
        return unknown_cursor.rowcount + rejected_cursor.rowcount

    async def _remember_personal(
        self, event: Event, analysis: WordAnalysis, observed_at: str
    ) -> None:
        associations = tuple(dict.fromkeys(words_from_text(str(event.payload.get("content", "")))))[
            :12
        ]
        await self._database.execute(
            """
            INSERT INTO language_personal_interpretations (
                guild_id, normalized_form, lemma, interpretation, associations_json,
                supporting_event_ids_json, confidence, first_seen_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (guild_id, normalized_form) DO UPDATE SET
                lemma = COALESCE(lemma, excluded.lemma),
                associations_json = excluded.associations_json,
                supporting_event_ids_json = CASE
                    WHEN occurrence_count < 20 THEN json_insert(
                        supporting_event_ids_json,
                        '$[#]',
                        excluded.supporting_event_ids_json ->> '$[0]'
                    ) ELSE supporting_event_ids_json END,
                occurrence_count = occurrence_count + 1,
                confidence = MIN(0.95, confidence + 0.03),
                updated_at = excluded.updated_at
            """,
            (
                event.guild_id,
                analysis.normalized_form,
                analysis.lemma,
                analysis.selected_meaning,
                json.dumps(associations, ensure_ascii=False),
                json.dumps((event.event_id,), ensure_ascii=False),
                max(0.1, analysis.confidence.meaning * 0.55),
                observed_at,
                observed_at,
            ),
        )

    async def ensure_study_queue(self, event: Event, analysis: WordAnalysis) -> bool:
        """Ensure an unresolved word is visible in the persistent study queue."""

        if not analysis.requires_lexical_study or not analysis.normalized_form:
            return False
        await self._upsert_unknown(
            event,
            analysis,
            event.utc_timestamp.isoformat(),
            increment_occurrence=False,
        )
        return True

    async def queue_semantic_study(
        self,
        event: Event,
        terms: tuple[str, ...],
        *,
        reason: str,
        priority: float = 0.72,
    ) -> int:
        """Queue known forms whose *meaning or use* is insufficient for a complete question.

        This is deliberately separate from ``language_unknown_concepts``: pymorphy may know
        that ``робот`` is a noun while the bot still lacks properties needed to reason about
        robots.  Reference sources can then teach the internal lexicon in the background.
        """

        if event.channel_id is None or event.payload.get("author_is_bot") is True:
            return 0
        normalized_terms = tuple(
            dict.fromkeys(
                lemma
                for raw in terms[:12]
                if (surface := normalize_word(raw))
                if (lemma := DEFAULT_MORPHOLOGY.lemma(surface))
                if _SEMANTIC_STUDY_TERM_RE.fullmatch(lemma)
            )
        )
        if not normalized_terms:
            return 0
        now = event.utc_timestamp.isoformat()
        queued = 0
        for term in normalized_terms:
            cursor = await self._database.execute(
                """INSERT INTO language_semantic_study_queue (
                       guild_id, channel_id, concept_key, reason, priority,
                       first_seen_at, last_seen_at, source_event_ids_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (guild_id, channel_id, concept_key) DO UPDATE SET
                       reason = excluded.reason,
                       priority = MAX(priority, excluded.priority),
                       last_seen_at = excluded.last_seen_at,
                       occurrence_count = occurrence_count + 1,
                       source_event_ids_json = CASE
                           WHEN occurrence_count < 20 THEN json_insert(
                               source_event_ids_json,
                               '$[#]',
                               excluded.source_event_ids_json ->> '$[0]'
                           ) ELSE source_event_ids_json END,
                       study_attempts = CASE
                           WHEN study_status = 'exhausted' THEN 0 ELSE study_attempts END,
                       study_status = CASE
                           WHEN study_status = 'exhausted' THEN 'pending' ELSE study_status END
                   """,
                (
                    event.guild_id,
                    event.channel_id,
                    term,
                    reason[:80],
                    max(0.0, min(1.0, priority)),
                    now,
                    now,
                    json.dumps((event.event_id,), ensure_ascii=False),
                ),
            )
            queued += int(cursor.rowcount > 0)
        if queued and self._study_notifier is not None:
            self._study_notifier()
        return queued

    async def resolve_unknown(self, guild_id: int, channel_id: int, word: str) -> int:
        """Mark a queued concept resolved after a grounded human explanation."""

        normalized = normalize_word(word)
        cursor = await self._database.execute(
            """UPDATE language_unknown_concepts
               SET resolved = 1, curiosity_score = 0.0, study_status = 'learned'
               WHERE guild_id = ? AND channel_id IN (0, ?)
                 AND concept_key = ? AND resolved = 0""",
            (guild_id, channel_id, normalized),
        )
        return cursor.rowcount

    async def _record_unknown(self, event: Event, analysis: WordAnalysis, observed_at: str) -> None:
        await self._upsert_unknown(
            event,
            analysis,
            observed_at,
            increment_occurrence=True,
        )

    async def _upsert_unknown(
        self,
        event: Event,
        analysis: WordAnalysis,
        observed_at: str,
        *,
        increment_occurrence: bool,
    ) -> None:
        if not analysis.requires_lexical_study:
            return
        relevance = 0.8 if event.payload.get("bot_mentioned") is True else 0.25
        await self._database.execute(
            """
            INSERT INTO language_unknown_concepts (
                guild_id, channel_id, concept_key, unknown_kind, first_seen_at, last_seen_at,
                importance_score, novelty_score, conversation_relevance, curiosity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT (guild_id, channel_id, concept_key) DO UPDATE SET
                unknown_kind = excluded.unknown_kind,
                last_seen_at = excluded.last_seen_at,
                occurrence_count = occurrence_count + ?,
                conversation_relevance = MAX(
                    conversation_relevance, excluded.conversation_relevance
                ),
                curiosity_score = MIN(
                    1.0,
                    curiosity_score + ? + excluded.conversation_relevance * 0.08
                ),
                study_attempts = CASE
                    WHEN study_status = 'exhausted' THEN 0 ELSE study_attempts END,
                study_status = CASE
                    WHEN study_status = 'exhausted' THEN 'pending' ELSE study_status END
            """,
            (
                event.guild_id,
                event.channel_id or 0,
                analysis.normalized_form,
                analysis.unknown_kind.value,
                observed_at,
                observed_at,
                relevance,
                relevance,
                min(1.0, 0.22 + relevance * 0.25),
                int(increment_occurrence),
                0.12 if increment_occurrence else 0.0,
            ),
        )
        if event.channel_id is not None:
            await self._database.execute(
                """UPDATE language_conversation_sessions SET
                    active_unknowns_json = CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM json_each(active_unknowns_json)
                            WHERE value = ?
                        ) THEN json_insert(active_unknowns_json, '$[#]', ?)
                        ELSE active_unknowns_json
                    END
                   WHERE guild_id = ? AND channel_id = ? AND status = 'active'""",
                (
                    analysis.normalized_form,
                    analysis.normalized_form,
                    event.guild_id,
                    event.channel_id,
                ),
            )
        if self._study_notifier is not None:
            self._study_notifier()

    async def _slang(self, guild_id: int, channel_id: int | None, normalized: str):
        return await self._database.fetchone(
            """SELECT * FROM language_server_slang
               WHERE guild_id = ? AND normalized_form = ? AND channel_id IN (0, ?)
               ORDER BY CASE WHEN channel_id = ? THEN 0 ELSE 1 END, confidence DESC LIMIT 1""",
            (guild_id, normalized, channel_id or 0, channel_id or 0),
        )

    async def _personal(self, guild_id: int, normalized: str):
        return await self._database.fetchone(
            """SELECT * FROM language_personal_interpretations
               WHERE guild_id = ? AND normalized_form = ?""",
            (guild_id, normalized),
        )

    async def _neologism(self, guild_id: int, normalized: str):
        return await self._database.fetchone(
            """SELECT status FROM language_neologism_candidates
               WHERE guild_id = ? AND normalized_form = ?
                 AND status NOT IN ('rejected', 'expired')""",
            (guild_id, normalized),
        )

    async def _senses_for_lemma(self, lemma: str) -> tuple[DictionarySense, ...]:
        rows = await self._database.fetchall(
            """SELECT s.*, h.group_key
               FROM language_dictionary_senses AS s
               LEFT JOIN language_homonym_groups AS h ON h.sense_id = s.sense_id
               WHERE s.lemma = ?
               ORDER BY s.confidence DESC, s.sense_id""",
            (normalize_word(lemma),),
        )
        return tuple(
            DictionarySense(
                sense_id=str(row["sense_id"]),
                lemma=str(row["lemma"]),
                part_of_speech=(str(row["part_of_speech"]) if row["part_of_speech"] else None),
                definition=str(row["definition"]),
                context_terms=tuple(
                    str(item) for item in json.loads(str(row["context_terms_json"]))
                ),
                examples=tuple(str(item) for item in json.loads(str(row["examples_json"]))),
                source=str(row["source"]),
                confidence=float(row["confidence"]),
                homonym_group=str(row["group_key"]) if row["group_key"] else None,
            )
            for row in rows
        )

    def _resolve_sense_from_context(
        self, senses: tuple[DictionarySense, ...], context: str
    ) -> DictionarySense | None:
        if not senses:
            return None
        if len(senses) == 1:
            return senses[0]
        context_words = set(words_from_text(context))
        context_lemmas = set(context_words)
        if self._morph is not None:
            for word in context_words:
                parsed = self._morph.parse(word)
                if parsed:
                    context_lemmas.add(str(parsed[0].normal_form))
        ranked: list[tuple[float, DictionarySense]] = []
        for sense in senses:
            discriminators = set(sense.context_terms)
            overlap = len(context_lemmas & discriminators)
            definition_terms = set(words_from_text(sense.definition))
            definition_overlap = len(context_lemmas & definition_terms)
            score = overlap * 1.0 + definition_overlap * 0.25 + sense.confidence * 0.1
            ranked.append((score, sense))
        ranked.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        best_score, best = ranked[0]
        runner_score = ranked[1][0]
        if best_score < 0.5 or best_score - runner_score < 0.35:
            return None
        contextual_confidence = min(0.98, 0.62 + best_score * 0.08)
        return DictionarySense(
            best.sense_id,
            best.lemma,
            best.part_of_speech,
            best.definition,
            best.context_terms,
            best.examples,
            best.source,
            contextual_confidence,
            best.homonym_group,
        )

    async def _cached_reference(
        self,
        normalized: str,
        *,
        context: str | None = None,
    ) -> (
        tuple[
            str,
            str | None,
            dict[str, str],
            tuple[str, ...],
            tuple[str, ...],
            str,
            float,
        ]
        | None
    ):
        rows = await self._database.fetchall(
            """SELECT e.*, f.grammatical_features_json
               FROM language_dictionary_forms AS f
               JOIN language_dictionary_entries AS e ON e.lemma = f.lemma
               WHERE f.surface_form = ?
               ORDER BY e.confidence DESC LIMIT 32""",
            (normalized,),
        )
        reading = DEFAULT_MORPHOLOGY.reading_in_context(normalized, context) if context else None
        contextual = reading is not None and reading.is_known

        def compatible(item: Any) -> bool:
            if not contextual:
                return True
            part = str(item["part_of_speech"] or "")
            return str(item["lemma"]) == reading.lemma and (
                not part or POS_NAMES.get(part, part) == reading.part_of_speech
            )

        row = next((item for item in rows if compatible(item)), None)
        if row is None:
            row = await self._database.fetchone(
                """SELECT *, '{}' AS grammatical_features_json
                   FROM language_dictionary_entries WHERE lemma = ?""",
                (reading.lemma if contextual else normalized,),
            )
        if row is None or not compatible(row):
            return None
        features = {
            str(name): str(value)
            for name, value in json.loads(str(row["grammatical_features_json"])).items()
        }
        if contextual:
            # Case/tense belong to this occurrence, not to a cached headword.
            features = reading.features()
        return (
            str(row["lemma"]),
            str(row["part_of_speech"]) if row["part_of_speech"] else (
                reading.part_of_speech if contextual else None
            ),
            features,
            tuple(str(item) for item in json.loads(str(row["meanings_json"]))),
            tuple(str(item) for item in json.loads(str(row["style_labels_json"]))),
            "dictionary_cache",
            float(row["confidence"]),
        )

    def _reference_analysis(
        self,
        normalized: str,
        *,
        context: str | None = None,
    ) -> (
        tuple[
            str,
            str | None,
            dict[str, str],
            tuple[str, ...],
            tuple[str, ...],
            str,
            float,
        ]
        | None
    ):
        reading = DEFAULT_MORPHOLOGY.reading_in_context(normalized, context)
        raw_parses = self._morph.parse(normalized) if self._morph is not None else ()
        is_known = bool(raw_parses and any(item.is_known for item in raw_parses))
        lemma_guess = reading.lemma if reading is not None else _fallback_lemma(normalized)
        builtin = _BUILTIN.get(normalized) or _BUILTIN.get(lemma_guess)
        if builtin:
            part, meanings, labels = builtin
            return (
                lemma_guess,
                part,
                _fallback_features(normalized, lemma_guess, part),
                meanings,
                labels,
                "builtin",
                0.96,
            )
        abbreviation_expansions = abbreviation_meanings(normalized)
        if abbreviation_expansions:
            meanings = tuple(
                f"сокращение от «{expansion}»" for expansion in abbreviation_expansions
            )
            return (
                normalized,
                "NOUN",
                {"abbreviation": "yes"},
                meanings,
                ("abbreviation",),
                "builtin_abbreviation",
                0.98,
            )
        if reading is not None and reading.analysis_kind == "numeric_literal":
            return (
                reading.lemma,
                "NUMR",
                reading.features(),
                (),
                ("numeric_literal",),
                "numeric_literal",
                0.99,
            )
        if self._morph is None:
            return None
        if reading is None or (not is_known and reading.corrected_form is None):
            return None
        return (
            reading.lemma,
            reading.part_of_speech,
            reading.features(),
            (),
            (),
            "opencorpora_spelling" if reading.corrected_form else "opencorpora",
            min(0.95, 0.65 + reading.score * 0.3),
        )

    @staticmethod
    def _associations(row: Any | None) -> tuple[str, ...]:
        if row is None:
            return ()
        return tuple(str(item) for item in json.loads(str(row["associations_json"])))


class NeologismRepository:
    """Quarantine generated words until repeated human use gives them a stable role."""

    def __init__(
        self,
        database: Database,
        *,
        expiration_days: int = 30,
        min_response_interval: int = 75,
    ) -> None:
        self._database = database
        self.expiration_days = expiration_days
        self.min_response_interval = min_response_interval

    async def create_candidate(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        word: str,
        creator_context: str,
        source_components: tuple[str, ...],
        intended_meaning: str,
        example_usage: str,
        dictionary_match: str | None = None,
        similar_words: tuple[str, ...] = (),
        response_number: int | None = None,
        global_response_number: int | None = None,
        now: datetime | None = None,
    ) -> NeologismCandidate | None:
        if dictionary_match:
            return None
        if response_number is not None and not await self.cooldown_allows(
            guild_id,
            response_number=response_number,
            global_response_number=global_response_number,
        ):
            return None
        moment = now or datetime.now(UTC)
        candidate_id = f"neologism:{uuid4()}"
        cursor = await self._database.execute(
            """
            INSERT OR IGNORE INTO language_neologism_candidates (
                candidate_id, guild_id, channel_id, word, normalized_form, created_at,
                creator_context, source_components_json, intended_meaning, example_usage,
                dictionary_match, similar_words_json, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                guild_id,
                channel_id,
                word,
                normalize_word(word),
                moment.isoformat(),
                creator_context[:1000],
                json.dumps(source_components, ensure_ascii=False),
                intended_meaning[:500],
                example_usage[:1000],
                dictionary_match,
                json.dumps(similar_words, ensure_ascii=False),
                (moment + timedelta(days=self.expiration_days)).isoformat(),
            ),
        )
        if cursor.rowcount != 1:
            return None
        if response_number is not None:
            await self._record_cooldowns(
                guild_id,
                response_number=response_number,
                global_response_number=global_response_number,
                now=moment,
            )
        return NeologismCandidate(
            candidate_id, word, intended_meaning, "experimental", 0.1, 0, 0, 0, 0
        )

    async def cooldown_allows(
        self,
        guild_id: int,
        *,
        response_number: int,
        global_response_number: int | None = None,
    ) -> bool:
        scopes = ((f"guild:{guild_id}", response_number),)
        if global_response_number is not None:
            scopes = (*scopes, ("global", global_response_number))
        for scope, current in scopes:
            row = await self._database.fetchone(
                """SELECT last_response_number FROM language_neologism_cooldowns
                   WHERE scope_key = ?""",
                (scope,),
            )
            if row and current - int(row["last_response_number"]) < self.min_response_interval:
                return False
        return True

    async def _record_cooldowns(
        self,
        guild_id: int,
        *,
        response_number: int,
        global_response_number: int | None,
        now: datetime,
    ) -> None:
        scopes = [(f"guild:{guild_id}", response_number)]
        if global_response_number is not None:
            scopes.append(("global", global_response_number))
        async with self._database.transaction() as connection:
            await connection.executemany(
                """INSERT INTO language_neologism_cooldowns (
                    scope_key, last_response_number, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT (scope_key) DO UPDATE SET
                    last_response_number = excluded.last_response_number,
                    updated_at = excluded.updated_at""",
                ((scope, number, now.isoformat()) for scope, number in scopes),
            )

    async def record_feedback(
        self,
        candidate_id: str,
        *,
        human_repeat: bool = False,
        positive: bool = False,
        negative: bool = False,
        used: bool = False,
    ) -> str | None:
        await self._database.execute(
            """UPDATE language_neologism_candidates SET
                usage_count = usage_count + ?, human_repeat_count = human_repeat_count + ?,
                positive_reaction_count = positive_reaction_count + ?,
                negative_reaction_count = negative_reaction_count + ?,
                last_used_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE last_used_at END,
                confidence = MIN(1.0, MAX(0.0, confidence + ?))
               WHERE candidate_id = ?""",
            (
                int(used),
                int(human_repeat),
                int(positive),
                int(negative),
                int(used),
                int(human_repeat) * 0.18 + int(positive) * 0.12 - int(negative) * 0.25,
                candidate_id,
            ),
        )
        row = await self._database.fetchone(
            "SELECT * FROM language_neologism_candidates WHERE candidate_id = ?", (candidate_id,)
        )
        if row is None:
            return None
        status = str(row["status"])
        if int(row["negative_reaction_count"]) >= 2:
            status = "rejected"
        elif int(row["human_repeat_count"]) >= 2 and float(row["confidence"]) >= 0.55:
            status = "server_slang"
        elif int(row["usage_count"]) >= 2 and float(row["confidence"]) >= 0.35:
            status = "understood"
        now = datetime.now(UTC).isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                "UPDATE language_neologism_candidates SET status = ? WHERE candidate_id = ?",
                (status, candidate_id),
            )
            if status == "server_slang":
                await connection.execute(
                    """INSERT INTO language_server_slang (
                        guild_id, channel_id, normalized_form, meaning, confidence,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (guild_id, channel_id, normalized_form) DO UPDATE SET
                        meaning = excluded.meaning,
                        confidence = MAX(confidence, excluded.confidence),
                        updated_at = excluded.updated_at""",
                    (
                        int(row["guild_id"]),
                        int(row["channel_id"] or 0),
                        str(row["normalized_form"]),
                        str(row["intended_meaning"]),
                        float(row["confidence"]),
                        now,
                        now,
                    ),
                )
            elif status == "rejected":
                await connection.execute(
                    """INSERT INTO language_rejected_word_forms (
                        guild_id, normalized_form, reason, rejected_at
                    ) VALUES (?, ?, 'negative human feedback', ?)
                    ON CONFLICT (guild_id, normalized_form) DO UPDATE SET
                        reason = excluded.reason, rejected_at = excluded.rejected_at""",
                    (int(row["guild_id"]), str(row["normalized_form"]), now),
                )
        return status

    async def active(self, guild_id: int, limit: int = 20) -> tuple[NeologismCandidate, ...]:
        rows = await self._database.fetchall(
            """SELECT * FROM language_neologism_candidates
               WHERE guild_id = ? AND status NOT IN ('rejected', 'expired')
               ORDER BY created_at DESC LIMIT ?""",
            (guild_id, max(1, min(limit, 50))),
        )
        return tuple(
            NeologismCandidate(
                str(row["candidate_id"]),
                str(row["word"]),
                str(row["intended_meaning"]),
                str(row["status"]),
                float(row["confidence"]),
                int(row["usage_count"]),
                int(row["human_repeat_count"]),
                int(row["positive_reaction_count"]),
                int(row["negative_reaction_count"]),
            )
            for row in rows
        )

    async def expire(self, now: datetime | None = None) -> int:
        cursor = await self._database.execute(
            """UPDATE language_neologism_candidates SET status = 'expired'
               WHERE status IN ('experimental', 'understood') AND expires_at <= ?""",
            ((now or datetime.now(UTC)).isoformat(),),
        )
        return cursor.rowcount


class NeologismService:
    """Make a rare proposal only when a semantic reason precedes word generation."""

    def __init__(
        self,
        lexicon: LexiconService,
        character_markov: CharacterMarkovRepository,
        candidates: NeologismRepository,
    ) -> None:
        self._lexicon = lexicon
        self._character_markov = character_markov
        self._candidates = candidates

    async def propose(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        creator_context: str,
        source_components: tuple[str, ...],
        intended_meaning: str,
        response_number: int,
        global_response_number: int | None = None,
        missing_known_word: bool = False,
        recurring_situation: bool = False,
        creative_context: bool = False,
        positive_history: bool = False,
        seed: str = "",
    ) -> NeologismCandidate | None:
        semantic_reason = any(
            (
                missing_known_word,
                recurring_situation,
                creative_context,
                len(source_components) >= 2,
                positive_history,
            )
        )
        if not semantic_reason:
            return None
        if not await self._candidates.cooldown_allows(
            guild_id,
            response_number=response_number,
            global_response_number=global_response_number,
        ):
            return None
        existing = await self._lexicon.personal_words(guild_id)
        model = await self._character_markov.model(guild_id)
        word = model.generate(seed=seed, existing_words=existing)
        if word is None:
            return None
        if await self._lexicon.is_rejected_word(guild_id, word):
            return None
        analysis = await self._lexicon.analyze_word(
            word,
            guild_id=guild_id,
            channel_id=channel_id,
            context=creator_context,
        )
        if analysis.is_dictionary_word:
            return None
        similar = await self._lexicon.find_similar_words(word)
        return await self._candidates.create_candidate(
            guild_id=guild_id,
            channel_id=channel_id,
            word=word,
            creator_context=creator_context,
            source_components=source_components,
            intended_meaning=intended_meaning,
            example_usage=f"{word} — {intended_meaning}",
            similar_words=similar,
            response_number=response_number,
            global_response_number=global_response_number,
        )


def _select_meaning(
    lemma: str, meanings: tuple[str, ...], context: str
) -> tuple[str | None, float]:
    if len(meanings) <= 1:
        return (meanings[0] if meanings else None), (0.9 if meanings else 0.52)
    words = set(words_from_text(context))
    if lemma == "лук":
        weapon = len(words & _WEAPON_CONTEXT)
        food = len(words & _FOOD_CONTEXT)
        if weapon > food:
            return meanings[0], min(0.96, 0.65 + weapon * 0.1)
        if food > weapon:
            return meanings[1], min(0.96, 0.65 + food * 0.1)
    # Resolve arbitrary dictionary senses through their own semantic evidence.
    # Ties retain ambiguity; never mix the relations of competing meanings.
    context_terms = set(content_lemmas(context)) - {lemma}
    scores = tuple(len(context_terms & set(content_lemmas(meaning))) for meaning in meanings)
    best = max(scores, default=0)
    if best > 0 and scores.count(best) == 1:
        return meanings[scores.index(best)], min(0.9, 0.55 + best * 0.12)
    return None, 0.34


def _fallback_lemma(word: str) -> str:
    explicit = {
        "палкой": "палка",
        "палке": "палка",
        "палку": "палка",
        "нашла": "найти",
        "нашёл": "найти",
        "нашел": "найти",
    }
    return explicit.get(word, word)


def _meaning_overlap(left: str, right: str) -> float:
    left_terms = set(words_from_text(left))
    right_terms = set(words_from_text(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def _fallback_features(word: str, lemma: str, part: str) -> dict[str, str]:
    if word == "палкой":
        return {"gender": "feminine", "number": "singular", "case": "instrumental"}
    if word == "нашла":
        return {"gender": "feminine", "number": "singular", "tense": "past"}
    return {"part_of_speech": part, "lemma": lemma}


def _tag_features(tag: Any) -> dict[str, str]:
    names = {
        "gender": getattr(tag, "gender", None),
        "number": getattr(tag, "number", None),
        "case": getattr(tag, "case", None),
        "person": getattr(tag, "person", None),
        "tense": getattr(tag, "tense", None),
    }
    return {name: str(value) for name, value in names.items() if value}
