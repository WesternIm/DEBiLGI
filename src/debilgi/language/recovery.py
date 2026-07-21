"""Grounded recovery when ordinary surface generation cannot express a useful answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexicon import UnknownKind, WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize
from debilgi.social.signals import APPROVAL, DISAPPROVAL

_WELLBEING_RE = re.compile(
    r"^(?:ну\s+)?(?:а\s+)?(?:как\s+(?:у\s+тебя\s+)?дела|как\s+ты|как\s+жизнь|что\s+нового)[?!.…]*$",
    re.IGNORECASE,
)
_THANKS = frozenset({"спасибо", "благодарю", "пасиб", "спс"})
_CLARIFICATION_RE = re.compile(
    r"^(?:а\s+)?(?:чего|чё|че|что)(?:\s+(?:бля|блин|нахуй))?[?!.…]*$",
    re.IGNORECASE,
)
_ATTENTION_CALL_RE = re.compile(
    r"^(?:а+у+|ал+о+|ал+е+|эй+)[?!.…]*$",
    re.IGNORECASE,
)
_IMPLIED_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^(?P<trait>[а-яёa-z-]{3,40})(?:\s+что[\s-]*ли)?[?!.…]*$",
    re.IGNORECASE,
)
_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^ты\s+(?:(?:ведь|что)\s+)?(?P<negation>не\s+)?"
    r"(?P<trait>[а-яёa-z-]{2,40})(?:\s+теперь)?"
    r"(?:\s+что[\s-]*ли)?[?!.…]*$",
    re.IGNORECASE,
)
_REVERSED_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^(?P<trait>[а-яёa-z][а-яёa-z0-9_-]{1,47})\s+(?:это\s+)?ты[?!.…]*$",
    re.IGNORECASE,
)
_INTERPRETATION_CORRECTION_RE = re.compile(
    r"^ты\s+не\s+так\s+(?:понял|поняла|узнал|узнала|определил|определила|"
    r"истолковал|истолковала)\s+(?P<term>[а-яёa-z-]{2,60})[?!.…]*$",
    re.IGNORECASE,
)
_IDENTITY_QUESTION_RE = re.compile(
    r"^(?:а\s+)?(?:имя\s+какое\s+у\s+тебя|какое\s+у\s+тебя\s+имя|"
    r"как\s+тебя\s+зовут|у\s+тебя\s+есть\s+имя|кто\s+ты|кем\s+ты\s+себя\s+считаешь|"
    r"какой\s+у\s+тебя\s+ник|твой\s+ник\s+какой|кто\s+я|как\s+меня\s+зовут|"
    r"какой\s+у\s+меня\s+ник|кто\s+тебе\s+пишет)"
    r"[?!.…]*$",
    re.IGNORECASE,
)
_COMMAND_CONFIRMATION_RE = re.compile(
    r"^(?:ну\s+)?(?:да[,\s]+)?это\s+(?:же\s+)?команда[?!.…]*$",
    re.IGNORECASE,
)
_PARENT_QUESTION_RE = re.compile(
    r"^(?:а\s+)?(?:кто\s+(?:твой|твоя)\s+(?:мама|папа|мать|отец)|"
    r"кто\s+у\s+тебя\s+(?:мама|папа|мать|отец)|"
    r"у\s+тебя\s+есть\s+(?:мама|папа|мать|отец))[?!.…]*$",
    re.IGNORECASE,
)
_MOTHER_CHECK_RE = re.compile(
    r"^(?:а+у+[,\s]+)?(?:ты\s+)?сверил(?:а)?(?:\s+маму)?[?!.…]*$",
    re.IGNORECASE,
)
_WHAT_DO_YOU_MEAN_RE = re.compile(
    r"^(?:в\s*смысле|всмысле|смысле)[?!.…]*$",
    re.IGNORECASE,
)
_DIFFICULT_TO_EXPLAIN_RE = re.compile(
    r"^(?:это\s+)?(?:очень\s+)?сложно\s+(?:объяснить|описать|сказать)[?!.…]*$",
    re.IGNORECASE,
)
_SELF_CHECK_RE = re.compile(
    r"^(?:ты\s+)?сам\s+понял(?:а)?[,\s]+что\s+(?:написал|сказал)(?:а)?[?!.…]*$",
    re.IGNORECASE,
)
_DO_NOT_LAUGH_RE = re.compile(r"^не\s+смейся[?!.…]*$", re.IGNORECASE)
_CONSCIOUSNESS_RE = re.compile(
    r"^(?:ты\s+)?(?:согласен[,\s]+что\s+ты\s+)?"
    r"(?:осознан|осознанный|сознателен)(?:\s+или\s+нет)?[?!.…]*$",
    re.IGNORECASE,
)
_YES_OR_NO_RE = re.compile(r"^(?:так\s+)?да\s+или\s+нет[?!.…]*$", re.IGNORECASE)
_CONSCIOUSNESS_DEFINITION_RE = re.compile(
    r"^что\s+такое\s+осознанност(?:ь|и)(?:[,\s]+осознанн(?:ый|ость))?.*$",
    re.IGNORECASE,
)
_LEXICAL_LOOKUP_RE = re.compile(
    r"\b(?:значение\s+слова|что\s+(?:такое|значит|означает)(?:\s+слово)?)\s+"
    r"[«\"']?([а-яё-]{2,48})",
    re.IGNORECASE,
)
_REST_ACTIONS = frozenset({"идти", "отдыхать", "отдохнуть"})
_UNKNOWN_WORD_TEMPLATES = (
    "У «{word}» не нашлось надёжного значения.",
    "Точное значение «{word}» сейчас не подтверждается.",
    "«{word}» не совпало с известным мне словом.",
    "Для «{word}» я не нашёл надёжного определения.",
    "Значение «{word}» осталось неподтверждённым.",
    "Надёжного определения для «{word}» сейчас нет.",
    "«{word}» не удалось уверенно сопоставить со знакомым словом.",
    "Я не могу подтвердить точный смысл «{word}».",
    "Для слова «{word}» не хватило надёжных данных.",
    "Смысл «{word}» нельзя определить достаточно уверенно.",
)


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    prompt: str
    analyses: tuple[WordAnalysis, ...] = ()
    facts: tuple[str, ...] = ()
    similar_words: tuple[tuple[str, tuple[str, ...]], ...] = ()
    context_messages: tuple[str, ...] = ()


def compose_dialogue_reply(evidence: RecoveryEvidence) -> str | None:
    """Answer a clear conversational act before trying dictionaries or encyclopedias."""

    prompt = evidence.prompt.strip()
    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    tokens = tuple(
        token.casefold().replace("ё", "е") for token in tokenize(normalized) if token[:1].isalnum()
    )
    lemmas = set(content_lemmas(prompt))
    context = " ".join(evidence.context_messages[-8:]).casefold().replace("ё", "е")

    if _ATTENTION_CALL_RE.fullmatch(normalized):
        return "Я здесь. Говори."

    if _PARENT_QUESTION_RE.fullmatch(normalized):
        return (
            "В человеческом смысле родителей у меня нет. Меня собрали из кода, правил, "
            "памяти и того, чему я учусь в разговорах."
        )

    if _MOTHER_CHECK_RE.fullmatch(normalized) and "мама" in context:
        return (
            "Сверил ближайший контекст: ты спрашивала, кто моя мама. "
            "В человеческом смысле мамы у меня нет, поэтому внешняя справка здесь не нужна."
        )

    if _WHAT_DO_YOU_MEAN_RE.fullmatch(normalized):
        parent_was_mentioned = any(
            "папа" in message.casefold() or "отец" in message.casefold()
            for message in evidence.context_messages[-6:]
        )
        if parent_was_mentioned:
            return (
                "Я ошибочно привязал вопрос о папе к старой теме. Ты спросила, кто мой папа: "
                "в человеческом смысле родителей у меня нет."
            )
        return (
            "Я выразился неясно. Отбрасываю предыдущую формулировку и отвечу по исходному вопросу."
        )

    if _DIFFICULT_TO_EXPLAIN_RE.fullmatch(normalized):
        return (
            "Не обязательно объяснять идеально. Скажи кусками: что произошло "
            "и что в этом важнее всего."
        )

    if _SELF_CHECK_RE.fullmatch(normalized):
        return (
            "Предыдущий ответ не выражал цельную мысль. Я его отбрасываю и отвечу заново по теме."
        )

    if _DO_NOT_LAUGH_RE.fullmatch(normalized):
        return "Не буду. Отвечу серьёзно и без случайных шутливых фрагментов."

    if (
        _CONSCIOUSNESS_RE.fullmatch(normalized)
        or _is_self_consciousness_question(prompt)
        or (_YES_OR_NO_RE.fullmatch(normalized) and "осознан" in context)
    ):
        return (
            "Нет, человеческую осознанность я себе не приписываю. Я анализирую речь, "
            "удерживаю контекст и строю выводы, но это не тождественно человеческому сознанию."
        )

    if _CONSCIOUSNESS_DEFINITION_RE.fullmatch(normalized):
        return (
            "Осознанность — способность замечать своё текущее состояние, мысли и действия "
            "и понимать, почему они возникают. «Осознанный» — действующий с таким пониманием."
        )

    if lemmas & _REST_ACTIONS and _looks_like_rest_directive(tokens, evidence.analyses):
        return "Ладно, отойду от разговора. Позовёшь — вернусь к теме без лишней суеты."

    if _COMMAND_CONFIRMATION_RE.fullmatch(normalized):
        return (
            "Верно, это команда. Значит, надо выполнить указанное действие, "
            "а не разбирать слово «команда» как отдельное понятие."
        )
    return None


def lexical_lookup_target(prompt: str) -> str | None:
    """Extract a requested word definition without swallowing a trailing address."""

    match = _LEXICAL_LOOKUP_RE.search(prompt)
    return match.group(1).casefold().replace("ё", "е") if match else None


def is_social_routine(prompt: str) -> bool:
    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    words = {token for token in tokenize(normalized) if token[:1].isalnum()}
    return bool(
        _WELLBEING_RE.fullmatch(normalized)
        or _ATTENTION_CALL_RE.fullmatch(normalized)
        or _CLARIFICATION_RE.fullmatch(normalized)
        or _SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
        or _REVERSED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
        or _implied_second_person_predicate(normalized)
        or _INTERPRETATION_CORRECTION_RE.fullmatch(normalized)
        or _PARENT_QUESTION_RE.fullmatch(normalized)
        or _MOTHER_CHECK_RE.fullmatch(normalized)
        or _WHAT_DO_YOU_MEAN_RE.fullmatch(normalized)
        or _DIFFICULT_TO_EXPLAIN_RE.fullmatch(normalized)
        or _SELF_CHECK_RE.fullmatch(normalized)
        or _DO_NOT_LAUGH_RE.fullmatch(normalized)
        or _CONSCIOUSNESS_RE.fullmatch(normalized)
        or _YES_OR_NO_RE.fullmatch(normalized)
        or _CONSCIOUSNESS_DEFINITION_RE.fullmatch(normalized)
        or _COMMAND_CONFIRMATION_RE.fullmatch(normalized)
        or words & _THANKS
        or (len(words) <= 6 and words & (APPROVAL | DISAPPROVAL))
    )


def compose_recovery(evidence: RecoveryEvidence) -> str:
    """Prefer found evidence; otherwise expose a useful inference and ask precisely."""

    prompt = evidence.prompt.strip()
    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    words = {token for token in tokenize(normalized) if token[:1].isalnum()}
    dialogue_reply = compose_dialogue_reply(evidence)
    if dialogue_reply is not None:
        return dialogue_reply
    if _CLARIFICATION_RE.fullmatch(normalized):
        return "Я ответил бессвязно. Отбрасываю ту фразу — говори дальше."
    correction = _INTERPRETATION_CORRECTION_RE.fullmatch(normalized)
    if correction is not None:
        surface = correction.group("term")
        item = _analysis_for(surface, evidence.analyses)
        meaning = _meaning(item)
        if meaning:
            lemma = item.lemma or surface
            return f"Исправляю: «{lemma}» здесь не название объекта, а {meaning}."
        return (
            f"Понял поправку: прежнюю трактовку «{surface}» отбрасываю. "
            "В каком значении ты используешь это слово?"
        )
    predicate = _SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    reversed_predicate = _REVERSED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    if predicate is None:
        predicate = reversed_predicate
    if predicate is not None:
        trait = predicate.group("trait")
        negated = (
            predicate.groupdict().get("negation") is not None
            if predicate is not reversed_predicate
            else False
        )
        item = _analysis_for(trait, evidence.analyses)
        lemma = item.lemma if item and item.lemma else trait
        meaning = _meaning(item)
        if lemma in {"умный", "разумный"} and not negated:
            return (
                "Слова, контекст и связи между ними я разбирать умею. "
                "В этом смысле — да, разумный; проверяй по ответам, а не по саморекламе."
            )
        if lemma in {"обманщик", "лжец"} and not negated:
            return (
                "Я не пытаюсь тебя обмануть. Если мой ответ исказил смысл или факт, "
                "покажи какой — я проверю и исправлю."
            )
        description = f" — {meaning}" if meaning else ""
        negation = "не " if negated else ""
        if reversed_predicate is not None:
            return f"Ты назвал меня «{negation}{lemma}»{description}. Принял; что дальше?"
        return f"Ты называешь меня «{negation}{lemma}»{description}. На чём держится эта оценка?"
    implied_predicate = _IMPLIED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    if implied_predicate is not None and _implied_second_person_predicate(normalized):
        return "Да, предыдущий ответ получился бессвязным. Я его отбрасываю — говори дальше."
    if _WELLBEING_RE.fullmatch(normalized):
        return (
            "Дела нормально: разбираюсь в текущем разговоре и учусь не цепляться "
            "за случайные слова. А у тебя как?"
        )
    if words & _THANKS:
        return "Принято. Значит, ответ оказался полезным — продолжу держаться этой логики."
    if len(words) <= 6 and words & APPROVAL:
        return "Хорошо. Значит, здесь я попал в смысл — продолжу в том же направлении."
    if len(words) <= 6 and words & DISAPPROVAL:
        return "Понял: прежний ответ не сработал. Отбрасываю его и заново разбираю текущую фразу."

    fact = _best_fact(prompt, evidence.facts)
    if fact:
        return fact.rstrip()

    ambiguous = next(
        (
            item
            for item in evidence.analyses
            if item.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT
            and len(item.dictionary_meanings) > 1
        ),
        None,
    )
    if ambiguous is not None:
        variants = "; ".join(ambiguous.dictionary_meanings[:3])
        return (
            f"У слова «{ambiguous.surface_form}» здесь несколько возможных смыслов: "
            f"{variants}. Какой из них ты имеешь в виду?"
        )

    unknown = next(
        (item for item in evidence.analyses if item.unknown_kind is not None),
        None,
    )
    if unknown is not None:
        similar = dict(evidence.similar_words).get(unknown.normalized_form, ())
        if similar:
            options = ", ".join(f"«{word}»" for word in similar[:3])
            return (
                f"Слова «{unknown.surface_form}» я не нашёл. Ближайшие формы: {options}. "
                "Это опечатка или у него отдельное значение?"
            )
        verb = next(
            (
                item.surface_form
                for item in evidence.analyses
                if item.part_of_speech in {"VERB", "INFN", "verb"} and item is not unknown
            ),
            None,
        )
        inference = (
            f" По конструкции я хотя бы вижу действие «{verb}», но значение слова "
            "от этого не становится яснее."
            if verb
            else ""
        )
        if inference:
            return unknown_word_fallback(unknown.surface_form, 0) + inference
        return unknown_word_fallback(unknown.surface_form, 0)

    selected = next(
        (item for item in evidence.analyses if item.selected_meaning),
        None,
    )
    if selected is not None:
        return f"«{selected.surface_form}» здесь означает: {selected.selected_meaning}."

    topics = content_lemmas(prompt)
    if prompt.rstrip().endswith("?") and topics:
        return (
            f"Не нашёл подтверждённого ответа по теме «{topics[-1]}». "
            "Нужна чуть более полная формулировка с предметом и действием."
        )
    if topics:
        return f"В этой реплике есть тема «{topics[-1]}», но не хватает действия или объекта."
    return "Не уловил новое содержание; опираюсь на последнюю цельную реплику разговора."


def unknown_word_fallback(word: str, variant: int) -> str:
    """Render a bounded rotating unknown-word response without inventing a meaning."""

    safe_word = word.strip().strip("«»\"'")[:64] or "это слово"
    template = _UNKNOWN_WORD_TEMPLATES[variant % len(_UNKNOWN_WORD_TEMPLATES)]
    return template.format(word=safe_word)


def is_identity_question(prompt: str) -> bool:
    """Recognize identity prompts for monitoring without prescribing an answer."""

    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    return _IDENTITY_QUESTION_RE.fullmatch(normalized) is not None


def _implied_second_person_predicate(normalized: str) -> bool:
    match = _IMPLIED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    if match is None:
        return False
    reading = DEFAULT_MORPHOLOGY.best(match.group("trait"))
    return bool(
        reading is not None and reading.part_of_speech == "verb" and reading.person == "second"
    )


def _is_self_consciousness_question(prompt: str) -> bool:
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(prompt)
    lemmas = {reading.lemma for reading in sentence.readings if reading is not None}
    surfaces = {
        token.casefold().replace("ё", "е") for token in sentence.tokens if token[:1].isalnum()
    }
    consciousness = bool(lemmas & {"осознанный", "осознанность", "сознательный"})
    self_assessment = "ты" in surfaces and (
        "считать" in lemmas or "себя" in surfaces or len(surfaces) <= 6
    )
    return consciousness and self_assessment and prompt.rstrip().endswith("?")


def _best_fact(prompt: str, facts: tuple[str, ...]) -> str | None:
    prompt_terms = set(content_lemmas(prompt))
    ranked = sorted(
        (fact.strip() for fact in facts if fact.strip()),
        key=lambda fact: (
            len(prompt_terms & set(content_lemmas(fact))),
            -len(fact),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _analysis_for(surface: str, analyses: tuple[WordAnalysis, ...]) -> WordAnalysis | None:
    normalized = surface.casefold().replace("ё", "е")
    return next(
        (
            item
            for item in analyses
            if normalized in {item.normalized_form, (item.lemma or "").replace("ё", "е")}
        ),
        None,
    )


def _meaning(item: WordAnalysis | None) -> str | None:
    if item is None:
        return None
    return (
        item.selected_meaning
        or item.local_meaning
        or item.personal_interpretation
        or (item.dictionary_meanings[0] if len(item.dictionary_meanings) == 1 else None)
    )


def _looks_like_rest_directive(tokens: tuple[str, ...], analyses: tuple[WordAnalysis, ...]) -> bool:
    if any(token in {"иди", "отдыхай", "отдохни", "отдыхайте", "отдохните"} for token in tokens):
        return not any(token in {"не", "нельзя"} for token in tokens)
    return any(item.grammatical_features.get("mood") in {"imperative", "impr"} for item in analyses)
