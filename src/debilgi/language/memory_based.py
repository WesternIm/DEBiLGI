"""Semi-language synthesis: retrieval, variable-order Markov, candidates, ranking."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import inf

from debilgi.cognition.dialogue_tasks import contact_response_binding
from debilgi.language.clarification_policy import is_contact_request
from debilgi.language.communication import reader_goal
from debilgi.language.content import content_lemmas
from debilgi.language.grammar import SoftGrammar, role
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.planning import UtterancePlan
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize
from debilgi.language.usage_learning import UsageGuidance
from debilgi.language.word_relations import (
    WordRelation,
    grammatical_tokens,
    relation_coherence,
    relation_fit,
)
from debilgi.social.models import PrimitiveVector
from debilgi.social.signals import reaction_affinity

_END = "\0end"
_START = "\0start"
_BLOCKED_TOKENS = {"<user>", "<role>", "<channel>", "<url>"}
_NO_SPACE_BEFORE = frozenset(",.!?:;)]}»…")
_NO_SPACE_AFTER = frozenset("([{«")
_SENTENCE_END = {".", "!", "?", "…"}
_OPENING_DELIMITERS = {"(": ")", "[": "]", "{": "}", "«": "»", "“": "”", "„": "“"}
_CLOSING_DELIMITERS = frozenset(_OPENING_DELIMITERS.values())
_COGNITIVE_STATE_ACTIONS = frozenset(
    {"понимать", "знать", "разбирать", "распознавать", "схватывать"}
)


@dataclass(frozen=True, slots=True)
class _Profile:
    chaos: float
    memory_attachment: float
    novelty: float
    coherence: float
    expressiveness: float
    target_tokens: int
    candidate_count: int
    reaction: PrimitiveVector | None


@dataclass(slots=True)
class _LanguageModel:
    starts: Counter[str]
    question_starts: Counter[str]
    transitions: dict[tuple[str, ...], Counter[str]]
    vocabulary: set[str]
    grammar: SoftGrammar
    question_grammar: SoftGrammar


@dataclass(frozen=True, slots=True)
class GeneratedUtterance:
    text: str
    variant: int
    renderer: str = "memory_graph"


@dataclass(frozen=True, slots=True)
class LearnedUtterance:
    text: str
    score: float
    variant: int
    source_event_ids: tuple[str, ...]
    relation_support: float = 0.0
    relation_conflicts: int = 0
    topicality: float = 0.0
    unresolved_reaction: bool = False
    cognitive_predicate: str | None = None


class MemoryPhraseGenerator:
    """Generate and rank many utterances using no authored response phrases."""

    def __init__(self, *, max_output_tokens: int = 72, max_output_chars: int = 1200) -> None:
        if max_output_tokens <= 0 or max_output_chars <= 0:
            raise ValueError("output limits must be positive")
        self.max_output_tokens = max_output_tokens
        self.max_output_chars = max_output_chars

    def generate_candidates(
        self, plan: UtterancePlan, *, limit: int = 8
    ) -> tuple[GeneratedUtterance, ...]:
        """Expose several deterministic surfaces instead of hiding the internal winner."""

        limit = max(1, min(limit, 24))
        seen: set[str] = set()
        generated: list[GeneratedUtterance] = []
        for variant in range(limit):
            context = replace(
                plan.context,
                seed=f"{plan.context.seed}|surface-variant:{variant}",
            )
            text = self.generate(replace(plan, context=context)).strip()
            key = " ".join(text.casefold().split())
            if not text or key in seen:
                continue
            seen.add(key)
            generated.append(GeneratedUtterance(text, variant))
        return tuple(generated)

    def generate_explorations(
        self, plan: UtterancePlan, *, recent_replies: tuple[str, ...] = (), limit: int = 12
    ) -> tuple[LearnedUtterance, ...]:
        """Construct language from observed transitions even with an incomplete parse.

        This is a statistical search over the bot's own corpus, not a bank of
        authored replies. Grammar and repetition affect ranking; input copies
        are excluded, including cosmetic changes of word order or perspective.
        The result is explicitly an exploration, never a verified proposition.
        """
        context = plan.context
        documents = tuple(
            dict.fromkeys(
                (
                    *(report.answer_text for report in context.dialogue_reports),
                    *(text for _, text in context.corpus_utterances),
                    *(
                        text.partition(": ")[2]
                        for text in context.consultation_messages
                        if text.startswith("user: ")
                    ),
                )
            )
        )
        documents = tuple(text for text in documents if internal_surface_reason(text) is None)
        if context.response_options:
            # Search only the domain supplied by the human, for any vocabulary.
            # Counts are a provisional choice, never evidence of truth or desire.
            terms = set(content_lemmas(context.prompt))
            connected = tuple(
                (source_id, text)
                for source_id, text in context.corpus_utterances
                if not terms or terms.intersection(content_lemmas(text))
            )
            ranked = []
            for option in context.response_options:
                lemma = DEFAULT_MORPHOLOGY.lemma(option)
                source_ids = tuple(
                    source_id
                    for source_id, text in connected
                    if lemma in {DEFAULT_MORPHOLOGY.lemma(word) for word in tokenize(text)}
                )
                count = sum(count for gram, count in context.generation_ngrams if option in gram)
                digest = hashlib.sha256(f"{context.prompt}|{option}".encode()).hexdigest()[:8]
                tie = int(digest, 16) / 2**32
                ranked.append(
                    LearnedUtterance(
                        option[:1].upper() + option[1:],
                        len(source_ids) + count * 0.01 + tie * 0.001,
                        0,
                        source_ids,
                    )
                )
            # The statistical rank decides; sentence-length scoring cannot
            # invent another answer or turn a binary choice into a question.
            return (max(ranked, key=lambda item: item.score),)
        prompt = self._usable_tokens(grammatical_tokens(context.prompt, human_perspective=True))
        prompt_terms = set(content_lemmas(prompt))
        contact_request = is_contact_request(context.prompt)
        active_task = (
            context.dialogue_situation.active_task
            if context.dialogue_situation is not None
            else None
        )
        open_generation = bool(
            active_task is not None
            and active_task.goal == "satisfy_directive"
            and any(
                value.endswith(("-нибудь", "-то", "-либо"))
                for _, value in active_task.objective.required_roles
            )
        )
        unresolved_input_recovery = bool(
            active_task is not None
            and active_task.goal == "respond_relevantly"
        )
        stored = context.generation_ngrams or (() if documents else context.memory_ngrams)
        observed_terms = {lemma for text in documents[:96] for lemma in content_lemmas(text)}
        observed_terms.update(lemma for gram, _ in stored for lemma in content_lemmas(gram))
        prompt_signal_terms = prompt_terms or {
            DEFAULT_MORPHOLOGY.lemma(token)
            for token in prompt
            if self._is_word(token)
        }
        unresolved_terms = {
            term
            for term in prompt_signal_terms
            if not any(same_lexical_family(term, observed) for observed in observed_terms)
        }
        anchors = prompt_terms | {
            lemma
            for lemma in observed_terms
            if any(same_lexical_family(lemma, term) for term in prompt_terms)
        }
        # A technical memory summary is not observed speech. Select complete
        # human sentences connected to the current words before building a model.
        all_sources = tuple(
            sentence
            for text in documents[:96]
            for sentence in self._sentences(
                self._usable_tokens(grammatical_tokens(text, human_perspective=True))
            )
        )
        topical_sources = tuple(
            sentence
            for sentence in all_sources
            if not prompt_terms or anchors.intersection(content_lemmas(sentence))
        )
        relation_neighbors = {
            lemma
            for link in context.word_relations
            if {link.governor, link.dependent}.intersection(anchors)
            for lemma in (link.governor, link.dependent)
        }
        topical_neighbors = {
            lemma
            for source in topical_sources
            for lemma in content_lemmas(source)
            if lemma not in anchors
        }
        connected_sources = tuple(
            source
            for source in all_sources
            if (anchors | relation_neighbors | topical_neighbors).intersection(
                content_lemmas(source)
            )
        )
        # A lexical walk must remain in the current connected component. The
        # previous backoff appended every sentence in the channel; that let a
        # mention of one word jump into an unrelated article or old argument.
        # Contact calls have no propositional topic, so they may inspect the
        # conversational corpus but still face the task-binding gate below.
        sources = tuple(
            dict.fromkeys(
                connected_sources
                or (
                    all_sources
                    if contact_request or open_generation or not prompt_terms
                    else ()
                )
            )
        )[:64]
        if contact_request:
            contact_surfaces = self._contact_explorations(
                context.word_relations,
                sources,
                context.seed,
                recent_replies,
                context.affect_features,
                limit,
            )
            if contact_surfaces:
                return contact_surfaces
        vocabulary = set(prompt).union(*(set(source) for source in sources))
        related_terms = anchors | {
            lemma
            for gram, _ in stored
            if anchors.intersection(content_lemmas(gram))
            for lemma in content_lemmas(gram)
        }
        related_terms.update(
            lemma
            for link in context.word_relations
            if {link.governor, link.dependent}.intersection(anchors)
            for lemma in (link.governor, link.dependent)
        )
        # Once a topic is known, disjoint learned walks cannot introduce an
        # unrelated subject merely because it was frequent in the old archive.
        stored = tuple(
            (gram, count)
            for gram, count in stored
            if not prompt_terms
            or open_generation
            or all(
                token in vocabulary
                or token in {_START, _END}
                or role(token)
                in {"pronoun", "preposition", "conjunction", "particle", "question", "punctuation"}
                or DEFAULT_MORPHOLOGY.lemma(token) in related_terms
                for token in gram
            )
        )
        model_sources = (*sources, *((prompt,) if prompt else ()))
        model = self._build_model(
            model_sources,
            stored,
            (),
            sentence_starts_only=True,
        )
        profile = self._profile(plan)
        goal = reader_goal(analyze_structured_utterance(context.prompt))
        prompt_words = tuple(token for token in prompt if self._is_word(token))
        recent_tokens = tuple(tokenize(text) for text in recent_replies)
        ranked: dict[str, tuple[float, int]] = {}
        prefixes = tuple(
            dict.fromkeys(
                source[index : index + 1]
                for source in model_sources
                for index in [
                    next(
                        (
                            i
                            for i, token in enumerate(source)
                            if DEFAULT_MORPHOLOGY.lemma(token) in anchors
                        ),
                        0,
                    )
                ]
            )
        )
        attempts = max(32, min(128, profile.candidate_count * 2)) if sources or stored else 0
        for variant in range(attempts):
            candidate = self._exploration_candidate(
                model,
                context.word_relations,
                f"{context.seed}|explore:{variant}",
                prefix=prefixes[(variant // 2) % len(prefixes)]
                if prefixes and variant % 2 == 0
                else (),
            )
            if not candidate:
                continue
            if any(self._is_source_copy(candidate, source) for source in sources):
                continue
            candidate = grammatical_tokens(self._detokenize(candidate), prune_unbound=True)
            if not candidate:
                continue
            if any(self._is_source_copy(candidate, source) for source in sources):
                continue
            question = (
                role(next((token for token in candidate if self._is_word(token)), "")) == "question"
            )
            if question:
                candidate = (
                    (*candidate[:-1], "?") if candidate[-1] in _SENTENCE_END else (*candidate, "?")
                )
            elif candidate[-1] not in _SENTENCE_END:
                candidate = (*candidate, ".")
            text = self._truncate_complete(self._detokenize(candidate), self.max_output_chars)
            if not text or internal_surface_reason(text) is not None:
                continue
            text = text[:1].upper() + text[1:]
            from debilgi.language.copying import is_input_copy

            if is_input_copy(text, context.prompt):
                continue
            terms = set(content_lemmas(candidate))
            topicality = min(1.0, len(terms & anchors) / max(1, len(prompt_terms)))
            surface = evaluate_surface(text)
            morphology = DEFAULT_MORPHOLOGY.analyze_sentence(text)
            relation = relation_coherence(text, context.word_relations)
            if surface.hard_issues or surface.sentence_completeness < 0.72:
                continue
            if relation.conflicts:
                continue
            if (
                prompt_terms
                and not contact_request
                and not open_generation
                and not terms.intersection(anchors)
            ):
                continue
            contact_binding = (
                contact_response_binding(text) if contact_request else 0.0
            )
            if contact_request and contact_binding < 0.80:
                continue
            actual_structure = analyze_structured_utterance(text)
            actual = reader_goal(actual_structure)
            unbound_arguments = sum(
                token.semantic_role == "unassigned_content"
                and token.part_of_speech in {"noun", "pronoun", "verb"}
                and dict(token.grammatical_features).get("case") != "genitive"
                for token in actual_structure.tokens
            )
            actual_subjects = set(actual.subject)
            for clause in actual_structure.clauses:
                if clause.subject_indices or clause.predicate_index is None:
                    continue
                features = dict(
                    actual_structure.tokens[clause.predicate_index].grammatical_features
                )
                inferred = {
                    ("first", "singular"): "я",
                    ("first", "plural"): "мы",
                    ("second", "singular"): "ты",
                    ("second", "plural"): "вы",
                }.get((features.get("person"), features.get("number")))
                if inferred:
                    actual_subjects.add(inferred)
            expected_subjects = {lemma for name, lemma in goal.known_roles if name == "subject"}
            subject_shift = bool(
                expected_subjects
                and actual_subjects
                and not expected_subjects.intersection(actual_subjects)
            )
            repetition = max(
                (self._sequence_similarity(candidate, prior) for prior in recent_tokens),
                default=0.0,
            )
            score = (
                self._score(
                    candidate,
                    prompt,
                    sources,
                    (),
                    model,
                    profile,
                    plan.speech_act,
                    context.usage_hints,
                    context.learned_syntax_patterns,
                    context.expected_response_acts,
                )
                + topicality * 3.0
                - 8.0 * bool(prompt_terms and not terms.intersection(anchors))
                + surface.score
                + morphology.agreement_score * 1.5
                + relation_fit(text, context.word_relations) * 2.0
                + relation.score * 1.5
                + contact_binding * 3.0
                - len(surface.hard_issues) * 1.5
                - len(morphology.issues) * 0.75
                - subject_shift * 6.0
                - unbound_arguments * 2.5
                - repetition * 2.0
                - 4.0
                * (
                    bool(prompt_words)
                    and tuple(token for token in candidate if self._is_word(token)) == prompt_words
                )
            )
            previous = ranked.get(text)
            if previous is None or previous[0] < score:
                ranked[text] = (score, variant)
        has_bound_ranked = bool(
            prompt_terms
            and any(
                set(content_lemmas(text)).intersection(anchors)
                for text in ranked
            )
        )
        if (
            unresolved_input_recovery
            and not contact_request
            and unresolved_terms
            and not has_bound_ranked
        ):
            unresolved_surfaces = self._unresolved_explorations(
                context.word_relations,
                all_sources,
                context.seed,
                recent_replies,
                context.affect_features,
                limit,
            )
            if unresolved_surfaces:
                return unresolved_surfaces
        if not ranked:
            # The bootstrap uses no authored words or fictional knowledge.
            # No lexical evidence is available for a new utterance. Punctuation
            # expresses an inquiry without disguising input as a new answer.
            ranked["?"] = (0.0, 0)
        result = sorted(ranked.items(), key=lambda item: (item[1][0], item[0]), reverse=True)
        return tuple(
            LearnedUtterance(
                text,
                score,
                variant,
                tuple(
                    event_id
                    for event_id, source in context.corpus_utterances
                    if set(tokenize(source)) & set(tokenize(text))
                ),
                relation_support=relation_coherence(text, context.word_relations).score,
                relation_conflicts=relation_coherence(text, context.word_relations).conflicts,
                topicality=min(
                    1.0,
                    len(set(content_lemmas(text)) & anchors) / max(1, len(prompt_terms)),
                ),
            )
            for text, (score, variant) in result[: max(1, min(24, limit))]
        )

    def _contact_explorations(
        self,
        relations: tuple[WordRelation, ...],
        sources: tuple[tuple[str, ...], ...],
        seed: str,
        recent_replies: tuple[str, ...],
        affect_features: tuple[tuple[str, float], ...],
        limit: int,
    ) -> tuple[LearnedUtterance, ...]:
        """Realize a contact intention from learned predicates, not reply text.

        The relation DB supplies the available actions and their observed
        subject role. Morphology supplies a first-person form. No remembered
        sentence is selected, and an accidentally identical source surface is
        still rejected by the ordinary copy guard.
        """

        counts: Counter[str] = Counter()
        for link in relations:
            if link.relation != "nsubj" or link.dependent not in {"я", "мы"}:
                continue
            counts[link.governor] += link.count
        if not counts:
            return ()

        affect = dict(affect_features)
        arousal = affect.get("emotion_arousal", 0.0)
        affiliation = affect.get("emotion_affiliation", 0.0)
        candidates: list[tuple[float, str]] = []
        seen = set()
        for lemma, count in counts.items():
            form = DEFAULT_MORPHOLOGY.inflect(
                lemma,
                person="first",
                number="singular",
                tense="present",
            )
            if not form:
                continue
            for explicit_subject, raw in (
                (False, f"{form}."),
                (True, f"Я {form}."),
            ):
                text = raw[:1].upper() + raw[1:]
                key = " ".join(text.casefold().split())
                if key in seen or contact_response_binding(text) < 0.80:
                    continue
                tokens = tokenize(text)
                if any(self._is_source_copy(tokens, source) for source in sources):
                    continue
                if any(
                    self._sequence_similarity(tokens, tokenize(prior)) >= 0.72
                    for prior in recent_replies
                ):
                    continue
                seen.add(key)
                tie = self._number(f"{seed}|contact:{lemma}:{explicit_subject}") / 2**64
                # Affect changes preference among learned actions; it never
                # supplies a word or turns a relation into a factual claim.
                score = (
                    float(count)
                    + tie * 0.25
                    + arousal * (0.10 if lemma in {"ответить", "сказать"} else 0.03)
                    + affiliation * (
                        0.12 if lemma in {"слушать", "слышать", "понимать"} else 0.02
                    )
                    - (0.04 if explicit_subject else 0.0)
                )
                candidates.append((score, text))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(
            LearnedUtterance(
                text,
                score,
                variant,
                (),
                relation_support=relation_coherence(text, relations).score,
                relation_conflicts=relation_coherence(text, relations).conflicts,
                topicality=1.0,
            )
            for variant, (score, text) in enumerate(
                candidates[: max(1, min(24, limit))]
            )
        )

    def _unresolved_explorations(
        self,
        relations: tuple[WordRelation, ...],
        sources: tuple[tuple[str, ...], ...],
        seed: str,
        recent_replies: tuple[str, ...],
        affect_features: tuple[tuple[str, float], ...],
        limit: int,
    ) -> tuple[LearnedUtterance, ...]:
        """Express a failed interpretation through a learned cognitive state.

        The memory supplies the predicate and proof that the agent has used it
        with a first-person subject. Negation is the result of the failed
        semantic lookup in the current turn. No stored reply is selected.
        """

        counts: Counter[str] = Counter()
        for link in relations:
            if (
                link.relation == "nsubj"
                and link.dependent in {"я", "мы"}
                and link.governor in _COGNITIVE_STATE_ACTIONS
            ):
                counts[link.governor] += link.count
        if not counts:
            return ()

        affect = dict(affect_features)
        arousal = affect.get("emotion_arousal", 0.0)
        confidence = affect.get("emotion_confidence", 0.0)
        candidates: list[tuple[float, str, str]] = []
        for lemma, count in counts.items():
            form = DEFAULT_MORPHOLOGY.inflect(
                lemma,
                person="first",
                number="singular",
                tense="present",
            )
            if not form:
                continue
            text = f"Я не {form}."
            tokens = tokenize(text)
            if any(self._is_source_copy(tokens, source) for source in sources):
                continue
            tie = self._number(f"{seed}|unresolved:{lemma}") / 2**64
            semantic_preference = {
                "понимать": 0.30,
                "разбирать": 0.22,
                "распознавать": 0.18,
                "знать": 0.12,
                "схватывать": 0.08,
            }.get(lemma, 0.0)
            repetition = max(
                (
                    self._sequence_similarity(tokens, tokenize(prior))
                    for prior in recent_replies
                ),
                default=0.0,
            )
            score = (
                float(count)
                + semantic_preference
                + tie * 0.20
                + confidence * 0.08
                - arousal * 0.03
                - repetition * 0.45
            )
            candidates.append((score, text, lemma))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(
            LearnedUtterance(
                text,
                score,
                variant,
                (),
                relation_support=relation_coherence(text, relations).score,
                relation_conflicts=relation_coherence(text, relations).conflicts,
                topicality=0.0,
                unresolved_reaction=True,
                cognitive_predicate=lemma,
            )
            for variant, (score, text, lemma) in enumerate(
                candidates[: max(1, min(24, limit))]
            )
        )

    @staticmethod
    def _sentences(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        sentences = []
        start = 0
        for index, token in enumerate(tokens):
            if token in _SENTENCE_END:
                if any(MemoryPhraseGenerator._is_word(item) for item in tokens[start : index + 1]):
                    sentences.append(tokens[start : index + 1])
                start = index + 1
        if any(MemoryPhraseGenerator._is_word(item) for item in tokens[start:]):
            sentences.append(tokens[start:])
        return tuple(sentences)

    def _exploration_candidate(
        self,
        model: _LanguageModel,
        relations: tuple[WordRelation, ...],
        seed: str,
        *,
        prefix: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        if not model.starts:
            return ()
        output = (
            list(prefix)
            if prefix
            else [self._weighted_pick(model.grammar.reweight(model.starts, ()), f"{seed}:start")]
        )
        while len(output) < self.max_output_tokens:
            available = tuple(
                model.transitions[tuple(output[-size:])]
                for size in range(min(3, len(output)), 0, -1)
                if tuple(output[-size:]) in model.transitions
            )
            if not available:
                break
            # Always taking the longest observed n-gram walks an old sentence
            # from start to finish.  Varying the history order lets separate
            # learned paths meet at shared words, which is where an actually
            # new sentence can emerge.  All choices still come from observed
            # transitions and remain deterministic for the event seed.
            options = available[
                self._number(f"{seed}:history:{len(output)}") % len(available)
            ]
            weighted = model.grammar.reweight(options, tuple(output))
            governors = {DEFAULT_MORPHOLOGY.lemma(token) for token in output[-3:]}
            partners = {link.dependent for link in relations if link.governor in governors}
            for token in weighted:
                if DEFAULT_MORPHOLOGY.lemma(token) in partners:
                    weighted[token] *= 1.5
            following = self._weighted_pick(weighted, f"{seed}:{len(output)}")
            if following in {_END, _START}:
                break
            if len(output) >= 2 and output[-1] == output[-2] == following:
                break
            output.append(following)
            # There is no minimum sentence length: reaching a boundary ends
            # the thought, even when the personality prefers a long response.
            if following in _SENTENCE_END:
                break
        return tuple(output)

    def generate(self, plan: UtterancePlan) -> str:
        context = plan.context
        if context.exact_answer:
            return self._truncate_complete(context.exact_answer, self.max_output_chars)
        prompt_tokens = self._usable_tokens(tokenize(context.prompt))
        retrieved = self._retrieve(context.memory_messages, prompt_tokens, limit=8)
        knowledge = context.knowledge_facts[:8]
        reply_heads = tuple(
            head for raw_head in context.reply_heads if (head := self._usable_tokens(raw_head))
        )
        if not retrieved and not knowledge and not reply_heads:
            return ""
        relevant_terms = set(content_lemmas(prompt_tokens))
        relevant_terms.update(
            lemma for message in (*retrieved, *knowledge) for lemma in content_lemmas(message)
        )
        stored_ngrams = tuple(
            item
            for item in context.memory_ngrams
            if len(relevant_terms) >= 2 and relevant_terms.intersection(content_lemmas(item[0]))
        )
        experience_transitions = tuple(
            item
            for item in context.experience_transitions
            if len(relevant_terms) >= 2 and relevant_terms.intersection(content_lemmas(item[0]))
        )
        tokenized = tuple(
            tokens
            for message in (*retrieved, *knowledge, context.prompt)
            if (tokens := self._usable_tokens(tokenize(message)))
        )
        model = self._build_model(
            tokenized,
            stored_ngrams,
            experience_transitions,
        )
        if not model.vocabulary and not reply_heads:
            return ""
        if len(model.vocabulary) == 1:
            return ""

        profile = self._profile(plan)
        source_tokens = tuple(self._usable_tokens(tokenize(message)) for message in retrieved)
        knowledge_tokens = tuple(self._usable_tokens(tokenize(fact)) for fact in knowledge)
        candidates: dict[tuple[str, ...], float] = {}
        for source_index, source in enumerate(source_tokens[:3]):
            if not source:
                continue
            is_question = source[-1:] == ("?",)
            if plan.speech_act == "clarifying_question" and not is_question:
                continue
            if plan.speech_act != "clarifying_question" and is_question:
                continue
            rendered_source = self._detokenize(source)
            source_preview = rendered_source
            if source[-1] not in _SENTENCE_END:
                source_preview += "."
            source_surface = evaluate_surface(source_preview)
            if source_surface.hard_issues or source_surface.score < 0.72:
                continue
            candidates[source] = self._score(
                source,
                prompt_tokens,
                source_tokens,
                knowledge_tokens,
                model,
                profile,
                plan.speech_act,
                context.usage_hints,
                context.learned_syntax_patterns,
                context.expected_response_acts,
            ) + max(0.15, 0.75 - source_index * 0.20)
        if plan.speech_act != "clarifying_question":
            prompt_words = set(content_lemmas(context.prompt))
            maximum = min(plan.max_tokens, self.max_output_tokens)
            for fact_index, fact in enumerate(knowledge_tokens[:3]):
                grounded = fact[:maximum]
                fact_words = set(content_lemmas(grounded))
                overlap = len(prompt_words & fact_words)
                if not grounded or (prompt_words and overlap == 0):
                    continue
                rank_bonus = max(0.0, 2.40 - fact_index * 0.80)
                recall_bonus = 1.35 + rank_bonus + min(1.20, overlap * 0.40)
                candidates[grounded] = (
                    self._score(
                        grounded,
                        prompt_tokens,
                        source_tokens,
                        knowledge_tokens,
                        model,
                        profile,
                        plan.speech_act,
                        context.usage_hints,
                        context.learned_syntax_patterns,
                        context.expected_response_acts,
                    )
                    + recall_bonus
                )
        for index in range(profile.candidate_count):
            seed = f"{context.seed}|{plan.speech_act}|{index}"
            candidate = self._generate_candidate(
                model,
                prompt_tokens,
                reply_heads,
                profile,
                seed,
                index,
                plan.speech_act == "clarifying_question",
            )
            if not candidate:
                continue
            preview = candidate
            if preview[-1] not in _SENTENCE_END:
                preview = (*preview, "?" if plan.speech_act == "clarifying_question" else ".")
            surface = evaluate_surface(self._detokenize(preview))
            if surface.hard_issues or surface.score < 0.72:
                continue
            score = self._score(
                candidate,
                prompt_tokens,
                source_tokens,
                knowledge_tokens,
                model,
                profile,
                plan.speech_act,
                context.usage_hints,
                context.learned_syntax_patterns,
                context.expected_response_acts,
            )
            candidates[candidate] = max(score, candidates.get(candidate, -inf))
        if not candidates:
            return ""
        selected = max(
            candidates.items(),
            key=lambda item: (item[1], self._number(f"{context.seed}|{item[0]}")),
        )[0]
        if plan.speech_act == "clarifying_question" and selected[-1] not in _SENTENCE_END:
            selected = (*selected, "?")
        elif selected[-1] not in _SENTENCE_END:
            selected = (*selected, ".")
        rendered = self._truncate_complete(self._detokenize(selected), self.max_output_chars)
        return rendered if not evaluate_surface(rendered).hard_issues else ""

    def _generate_candidate(
        self,
        model: _LanguageModel,
        prompt: tuple[str, ...],
        reply_heads: tuple[tuple[str, ...], ...],
        profile: _Profile,
        seed: str,
        index: int,
        question_mode: bool,
    ) -> tuple[str, ...]:
        output: list[str] = []
        starts = model.question_starts if question_mode and model.question_starts else model.starts
        grammar = model.question_grammar if question_mode else model.grammar
        if question_mode and model.question_starts:
            question_starts = grammar.reweight(model.question_starts, ())
            output.append(self._weighted_pick(question_starts, f"{seed}:question-start"))
        if not output and reply_heads and index % 4 == 0:
            output.extend(self._pick(reply_heads, f"{seed}:head"))
        if not output and starts and index % 3 == 1:
            grammatical_starts = grammar.reweight(starts, ())
            output.append(self._weighted_pick(grammatical_starts, f"{seed}:grammar-start"))
        if not output:
            following = self._next_token(model, prompt, profile.chaos, f"{seed}:prompt")
            if following != _END:
                output.append(following)
        if not output and starts:
            output.append(self._weighted_pick(starts, f"{seed}:start"))

        variation = self._number(f"{seed}:length") % 11 - 5
        maximum = max(
            4,
            min(self.max_output_tokens, profile.target_tokens + variation),
        )
        minimum = min(maximum, max(5, round(profile.target_tokens * 0.62)))
        target_sentences = 1
        sentence_ends = sum(token in _SENTENCE_END for token in output)
        while output and len(output) < maximum:
            history = (*prompt[-3:], *output)
            following = self._next_token(
                model,
                history,
                profile.chaos,
                f"{seed}:{len(output)}",
                grammar,
            )
            if following == _END:
                break
            if len(output) >= 2 and output[-1] == output[-2] == following:
                break
            output.append(following)
            if following in _SENTENCE_END:
                sentence_ends += 1
            if (
                len(output) >= minimum
                and sentence_ends >= target_sentences
                and following in _SENTENCE_END
            ):
                break
        return tuple(output)

    def _next_token(
        self,
        model: _LanguageModel,
        history: tuple[str, ...],
        chaos: float,
        seed: str,
        grammar: SoftGrammar | None = None,
    ) -> str:
        available: list[Counter[str]] = []
        for size in range(min(3, len(history)), 0, -1):
            options = model.transitions.get(tuple(history[-size:]))
            if options:
                available.append(options)
        if not available:
            return _END
        disorder = self._number(f"{seed}:order") % 10_000 / 10_000
        choice_index = 0
        if len(available) > 1 and disorder < chaos * 0.55:
            choice_index = 1 + self._number(f"{seed}:backoff") % (len(available) - 1)
        weighted = (grammar or model.grammar).reweight(available[choice_index], history)
        return self._weighted_pick(weighted, seed)

    @classmethod
    def _build_model(
        cls,
        messages: tuple[tuple[str, ...], ...],
        stored_ngrams: tuple[tuple[tuple[str, ...], int], ...],
        experience_transitions: tuple[tuple[tuple[str, ...], float], ...],
        *,
        sentence_starts_only: bool = False,
    ) -> _LanguageModel:
        starts: Counter[str] = Counter()
        transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()
        question_starts: Counter[str] = Counter()
        for tokens in messages:
            if cls._is_word(tokens[0]):
                starts[tokens[0]] += 1
            if tokens[-1:] == ("?",) or role(tokens[0]) == "question":
                question_starts[tokens[0]] += 1
            vocabulary.update(tokens)
            for position, following in enumerate((*tokens[1:], _END)):
                for size in range(1, min(3, position + 1) + 1):
                    history = tuple(tokens[position - size + 1 : position + 1])
                    transitions[history][following] += 1
        for gram, count in stored_ngrams:
            usable = cls._usable_tokens(gram)
            if len(usable) < 2 or count <= 0:
                continue
            if usable[0] == _START:
                starts[usable[1]] += count
                continue
            if not sentence_starts_only and cls._is_word(usable[0]):
                starts[usable[0]] += count
            vocabulary.update(usable)
            transitions[usable[:-1]][usable[-1]] += count
        for gram, adjustment in experience_transitions:
            usable = cls._usable_tokens(gram)
            if len(usable) < 2 or adjustment == 0.0:
                continue
            history, following = usable[:-1], usable[-1]
            current = float(transitions[history].get(following, 0.0))
            if current > 0.0 or adjustment > 0.0:
                transitions[history][following] = max(0.01, current + adjustment)
                vocabulary.update(usable)
        question_messages = tuple(
            tokens for tokens in messages if tokens[-1:] == ("?",) or role(tokens[0]) == "question"
        )
        return _LanguageModel(
            starts,
            question_starts,
            dict(transitions),
            vocabulary,
            SoftGrammar.learn(messages),
            SoftGrammar.learn(question_messages or messages),
        )

    @classmethod
    def _retrieve(
        cls, messages: tuple[str, ...], prompt: tuple[str, ...], *, limit: int
    ) -> tuple[str, ...]:
        prompt_words = set(content_lemmas(prompt))
        ranked: list[tuple[float, int, str]] = []
        total = max(1, len(messages))
        for index, message in enumerate(messages):
            if internal_surface_reason(message) is not None:
                continue
            if evaluate_surface(message).hard_issues:
                continue
            tokens = cls._usable_tokens(tokenize(message))
            words = set(content_lemmas(tokens))
            overlap = len(prompt_words & words) / max(1, len(prompt_words | words))
            if prompt_words and overlap <= 0.0:
                continue
            recency = (index + 1) / total
            ranked.append((overlap * 5.0 + recency * 0.35, index, message))
        selected = sorted(ranked, key=lambda item: (-item[0], -item[1]))[:limit]
        selected.sort(key=lambda item: item[1])
        return tuple(item[2] for item in selected)

    @classmethod
    def _score(
        cls,
        candidate: tuple[str, ...],
        prompt: tuple[str, ...],
        sources: tuple[tuple[str, ...], ...],
        knowledge_sources: tuple[tuple[str, ...], ...],
        model: _LanguageModel,
        profile: _Profile,
        speech_act: str,
        usage_hints: tuple[UsageGuidance, ...] = (),
        learned_syntax_patterns: tuple[tuple[str, ...], ...] = (),
        expected_response_acts: tuple[str, ...] = (),
    ) -> float:
        words = tuple(token for token in candidate if cls._is_word(token))
        unique = set(content_lemmas(candidate))
        prompt_words = set(content_lemmas(prompt))
        source_words = {lemma for source in sources[:8] for lemma in content_lemmas(source)}
        knowledge_words = {
            lemma for source in knowledge_sources for lemma in content_lemmas(source)
        }
        topic = len(unique & (prompt_words | source_words | knowledge_words)) / max(1, len(unique))
        grounding = len(unique & knowledge_words) / max(1, len(unique))
        novelty = len(unique - prompt_words) / max(1, len(unique))
        surface_unique = set(words)
        repetition = 1.0 - len(surface_unique) / max(1, len(words))
        coherent = sum(
            model.transitions.get((left,), Counter()).get(right, 0) > 0
            for left, right in zip(candidate, candidate[1:], strict=False)
        ) / max(1, len(candidate) - 1)
        copy_similarity = max(
            (cls._sequence_similarity(candidate, source) for source in sources), default=0.0
        )
        prompt_copy_similarity = cls._sequence_similarity(candidate, prompt)
        length_fit = 1.0 - min(
            1.0, abs(len(candidate) - profile.target_tokens) / max(1, profile.target_tokens)
        )
        coherence_fit = 1.0 - abs(coherent - profile.coherence)
        grammar = model.question_grammar if speech_act == "clarifying_question" else model.grammar
        grammar_fit = grammar.score(candidate)
        act_fit = 0.0
        if speech_act == "clarifying_question" and candidate[-1:] == ("?",):
            act_fit = 0.25
        elif speech_act != "clarifying_question" and candidate[-1:] != ("?",):
            act_fit = 0.10
        return (
            topic * (0.8 + profile.memory_attachment)
            + coherence_fit * (1.0 + profile.coherence)
            + novelty * profile.novelty
            + grounding * 1.15
            + grammar_fit * (1.15 + profile.coherence * 0.65)
            + length_fit * 0.75
            + act_fit
            + reaction_affinity(candidate, profile.reaction, speech_act) * 1.35
            + cls._usage_fit(candidate, usage_hints) * 0.55
            + cls._learned_syntax_fit(candidate, learned_syntax_patterns) * 0.45
            + (0.22 if speech_act in expected_response_acts else 0.0)
            - repetition * (1.35 - profile.chaos * 0.45)
            - copy_similarity * (0.45 + profile.novelty * 0.55)
            - prompt_copy_similarity * 2.10
        )

    @staticmethod
    def _learned_syntax_fit(
        candidate: tuple[str, ...], patterns: tuple[tuple[str, ...], ...]
    ) -> float:
        if not patterns:
            return 0.0
        candidate_pattern = tuple(
            reading.part_of_speech
            for token in candidate
            if (reading := DEFAULT_MORPHOLOGY.best(token)) is not None
            and reading.part_of_speech is not None
        )
        if not candidate_pattern:
            return 0.0
        best = 0.0
        for pattern in patterns:
            if not pattern:
                continue
            span = min(len(candidate_pattern), len(pattern))
            positional = sum(
                candidate_pattern[index] == pattern[index] for index in range(span)
            ) / max(len(candidate_pattern), len(pattern))
            shared = len(set(candidate_pattern) & set(pattern)) / max(
                1, len(set(candidate_pattern) | set(pattern))
            )
            best = max(best, positional * 0.72 + shared * 0.28)
        return best

    @staticmethod
    def _usage_fit(candidate: tuple[str, ...], hints: tuple[UsageGuidance, ...]) -> float:
        """Reward learned collocations softly; never reject an unseen construction."""

        if not hints:
            return 0.0
        readings = tuple(DEFAULT_MORPHOLOGY.best(token) for token in candidate)
        candidate_lemmas = {reading.lemma for reading in readings if reading is not None}
        scores: list[float] = []
        for hint in hints:
            target_readings = tuple(
                reading
                for token, reading in zip(candidate, readings, strict=True)
                if reading is not None
                and (reading.lemma == hint.lemma or token.casefold().replace("ё", "е") == hint.word)
            )
            if not target_readings:
                continue
            matches = 0.15
            if hint.preferred_case and any(
                item.case == hint.preferred_case for item in target_readings
            ):
                matches += 0.30
            if hint.preferred_number and any(
                item.number == hint.preferred_number for item in target_readings
            ):
                matches += 0.20
            neighbors = set(hint.neighbors)
            if neighbors & candidate_lemmas:
                matches += 0.35
            scores.append(matches * max(0.0, min(1.0, hint.confidence)))
        return max(scores, default=0.0)

    @staticmethod
    def _sequence_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
        left_pairs = set(zip(left, left[1:], strict=False))
        right_pairs = set(zip(right, right[1:], strict=False))
        return len(left_pairs & right_pairs) / max(1, len(left_pairs | right_pairs))

    @classmethod
    def _is_source_copy(cls, candidate: tuple[str, ...], source: tuple[str, ...]) -> bool:
        """Reject a stored sentence replay even when punctuation or one edge changes."""

        left = tuple(token for token in candidate if cls._is_word(token))
        right = tuple(token for token in source if cls._is_word(token))
        if not left or not right:
            return False
        if left == right:
            return True
        if len(left) == 1 or len(right) == 1:
            return left == right
        left_pairs = set(zip(left, left[1:], strict=False))
        right_pairs = set(zip(right, right[1:], strict=False))
        # A generated excerpt can be much shorter than its source and still be
        # a verbatim replay. Judge how much of the candidate came from the
        # source, instead of diluting that evidence by the source's length.
        containment = len(left_pairs & right_pairs) / max(1, len(left_pairs))
        if len(left) >= 4 and containment >= 0.80:
            return True
        length_ratio = min(len(left), len(right)) / max(len(left), len(right))
        return length_ratio >= 0.65 and cls._sequence_similarity(left, right) >= 0.72

    @staticmethod
    def _profile(plan: UtterancePlan) -> _Profile:
        personality = plan.context.personality
        reaction = plan.context.reaction.vector if plan.context.reaction else None
        reflection = plan.context.reflection
        if personality is None:
            return _Profile(0.5, 0.5, 0.5, 0.75, 0.5, 16, 32, reaction)
        arousal = reaction.arousal if reaction else 0.0
        curiosity = reaction.curiosity if reaction else 0.0
        resistance = reaction.resistance if reaction else 0.0
        agreement = reaction.agreement if reaction else 0.0
        trust = reaction.trust if reaction else 0.0
        length_scale = reflection.length_scale if reflection else 1.0
        exploration = reflection.exploration if reflection else 0.0
        caution = reflection.caution if reflection else 0.0
        reflection_confidence = reflection.confidence if reflection else 0.0
        return _Profile(
            chaos=max(
                0.0,
                min(1.0, personality.chaos + arousal * 0.08 - caution * 0.12),
            ),
            memory_attachment=max(
                0.0,
                min(
                    1.0,
                    personality.memory_attachment
                    + agreement * 0.10
                    + trust * 0.08
                    - resistance * 0.08,
                ),
            ),
            novelty=max(
                0.0,
                min(
                    1.0,
                    personality.novelty_drive + curiosity * 0.12 + exploration * 0.10,
                ),
            ),
            coherence=max(
                0.0,
                min(
                    1.0,
                    personality.coherence_drive
                    + trust * 0.06
                    - arousal * 0.04
                    + caution * 0.12 * reflection_confidence,
                ),
            ),
            expressiveness=max(0.0, min(1.0, personality.expressiveness + arousal * 0.10)),
            target_tokens=max(
                8,
                min(
                    56,
                    round(
                        (personality.target_tokens + curiosity * 4 + arousal * 3 - resistance * 2)
                        * length_scale
                    ),
                ),
            ),
            candidate_count=max(
                20,
                min(
                    96,
                    personality.candidate_count + round(curiosity * 10 + exploration * 12),
                ),
            ),
            reaction=reaction,
        )

    @staticmethod
    def _usable_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            token
            for token in tokens
            if token not in _BLOCKED_TOKENS and not token.startswith("<emoji:")
        )

    @staticmethod
    def _is_word(token: str) -> bool:
        return any(character.isalnum() for character in token)

    @classmethod
    def _weighted_pick(cls, values: Counter[str], seed: str) -> str:
        total = sum(values.values())
        position = cls._number(seed) % total
        for value, count in sorted(values.items()):
            if position < count:
                return value
            position -= count
        raise RuntimeError("weighted choice received an empty counter")

    @classmethod
    def _pick(cls, values: tuple[tuple[str, ...], ...], seed: str) -> tuple[str, ...]:
        return values[cls._number(seed) % len(values)] if values else ()

    @staticmethod
    def _number(seed: str) -> int:
        return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")

    @staticmethod
    def _detokenize(tokens: tuple[str, ...]) -> str:
        result = ""
        previous = ""
        for token in tokens:
            if not result or token in _NO_SPACE_BEFORE or previous in _NO_SPACE_AFTER:
                result += token
            else:
                result += f" {token}"
            previous = token
        return result

    @staticmethod
    def _truncate_complete(text: str, maximum: int) -> str:
        """Respect the cap, semantic boundaries and balanced quotes/brackets."""

        cleaned = text.strip()
        if len(cleaned) <= maximum:
            return _balance_delimiters(cleaned, maximum)
        if maximum <= 1:
            return "…"[:maximum]
        prefix = cleaned[:maximum]
        sentence_end = max(prefix.rfind(mark) for mark in _SENTENCE_END)
        if sentence_end >= min(120, maximum // 3):
            return _balance_delimiters(prefix[: sentence_end + 1].rstrip(), maximum)
        word_end = prefix.rfind(" ", 0, maximum - 1)
        if word_end <= 0:
            shortened = prefix[: maximum - 1].rstrip() + "…"
        else:
            shortened = prefix[:word_end].rstrip(" ,;:—-") + "…"
        return _balance_delimiters(shortened, maximum)


def _balance_delimiters(text: str, maximum: int) -> str:
    candidate = text[:maximum].rstrip()
    for _ in range(4):
        missing = _missing_closers(candidate)
        if not missing:
            return candidate
        if len(candidate) + len(missing) <= maximum:
            return candidate + missing
        budget = maximum - len(missing)
        if budget <= 1:
            return candidate[:maximum]
        prefix = candidate[:budget].rstrip()
        word_end = prefix.rfind(" ")
        candidate = (prefix[:word_end] if word_end > 0 else prefix).rstrip(" ,;:—-")
    return candidate[:maximum].rstrip()


def _missing_closers(text: str) -> str:
    stack: list[str] = []
    straight_quote_open = False
    for character in text:
        if character == '"':
            straight_quote_open = not straight_quote_open
            continue
        expected = _OPENING_DELIMITERS.get(character)
        if expected is not None:
            stack.append(expected)
        elif character in _CLOSING_DELIMITERS and stack and stack[-1] == character:
            stack.pop()
    closers = "".join(reversed(stack))
    return ('"' if straight_quote_open else "") + closers
