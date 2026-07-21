"""Semi-language synthesis: retrieval, variable-order Markov, candidates, ranking."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import inf

from debilgi.language.content import content_lemmas
from debilgi.language.grammar import SoftGrammar, role
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.planning import UtterancePlan
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize
from debilgi.language.usage_learning import UsageGuidance
from debilgi.social.models import PrimitiveVector
from debilgi.social.signals import reaction_affinity

_END = "\0end"
_BLOCKED_TOKENS = {"<user>", "<role>", "<channel>", "<url>"}
_NO_SPACE_BEFORE = frozenset(",.!?:;)]}»…")
_NO_SPACE_AFTER = frozenset("([{«")
_SENTENCE_END = {".", "!", "?", "…"}
_OPENING_DELIMITERS = {"(": ")", "[": "]", "{": "}", "«": "»", "“": "”", "„": "“"}
_CLOSING_DELIMITERS = frozenset(_OPENING_DELIMITERS.values())


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
            lemma
            for message in (*retrieved, *knowledge)
            for lemma in content_lemmas(message)
        )
        stored_ngrams = tuple(
            item
            for item in context.memory_ngrams
            if len(relevant_terms) >= 2
            and relevant_terms.intersection(content_lemmas(item[0]))
        )
        experience_transitions = tuple(
            item
            for item in context.experience_transitions
            if len(relevant_terms) >= 2
            and relevant_terms.intersection(content_lemmas(item[0]))
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
            candidates[source] = (
                self._score(
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
                )
                + max(0.15, 0.75 - source_index * 0.20)
            )
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
            if cls._is_word(usable[0]):
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
        source_words = {
            lemma for source in sources[:8] for lemma in content_lemmas(source)
        }
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
    def _usage_fit(
        candidate: tuple[str, ...], hints: tuple[UsageGuidance, ...]
    ) -> float:
        """Reward learned collocations softly; never reject an unseen construction."""

        if not hints:
            return 0.0
        readings = tuple(DEFAULT_MORPHOLOGY.best(token) for token in candidate)
        candidate_lemmas = {
            reading.lemma
            for reading in readings
            if reading is not None
        }
        scores: list[float] = []
        for hint in hints:
            target_readings = tuple(
                reading
                for token, reading in zip(candidate, readings, strict=True)
                if reading is not None
                and (
                    reading.lemma == hint.lemma
                    or token.casefold().replace("ё", "е") == hint.word
                )
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
