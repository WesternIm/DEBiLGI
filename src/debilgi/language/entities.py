"""Entity and identity context extracted before lexical uncertainty handling."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.character_markov import normalize_word
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize

if TYPE_CHECKING:
    from debilgi.language.productive_nlg import ClauseThought

_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
_SPEAKER_IDENTITY_RE = re.compile(
    r"^(?:а\s+)?(?:кто\s+я|как\s+меня\s+зовут|какое\s+у\s+меня\s+имя|"
    r"какой\s+у\s+меня\s+ник|мой\s+ник\s+какой|кто\s+тебе\s+пишет)\??$",
    re.IGNORECASE,
)
_BOT_IDENTITY_RE = re.compile(
    r"^(?:а\s+)?(?:кто\s+ты|ты\s+кто|как\s+тебя\s+зовут|какое\s+у\s+тебя\s+имя|"
    r"какой\s+у\s+тебя\s+ник|твой\s+ник\s+какой)\??$",
    re.IGNORECASE,
)
_WHO_IS_RE = re.compile(r"^(?:а\s+)?кто\s+(?:такой|такая|такие)?\s*(.+?)\??$", re.IGNORECASE)
_IDENTITY_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:(?:а|ну|так|слушай|скажи(?:-ка)?)[,\s]+)+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EntityMention:
    surface: str
    canonical: str
    kind: str
    role: str
    entity_id: int | None
    confidence: float


@dataclass(frozen=True, slots=True)
class EntityAnalysis:
    mentions: tuple[EntityMention, ...]
    action_terms: tuple[str, ...]
    object_terms: tuple[str, ...]

    @property
    def identity_terms(self) -> frozenset[str]:
        return frozenset(
            mention.canonical
            for mention in self.mentions
            if mention.kind in {"person", "nickname", "address"}
            if mention.canonical
        )


def render_message_content(content: str, payload: Mapping[str, Any]) -> str:
    """Remove the bot address while retaining semantic references to other entities."""

    bot_id = _identity_id(payload.get("bot_identity")) or _integer(payload.get("bot_user_id"))
    users = _records_by_id(payload.get("mentioned_users"))
    roles = _records_by_id(payload.get("mentioned_roles"))
    channels = _records_by_id(payload.get("mentioned_channels"))

    removed_implicit_bot_address = False

    def user_replacement(match: re.Match[str]) -> str:
        nonlocal removed_implicit_bot_address
        entity_id = int(match.group(1))
        if bot_id is not None and entity_id == bot_id:
            return " "
        record = users.get(entity_id)
        if (
            bot_id is None
            and payload.get("bot_mentioned") is True
            and record is None
            and not removed_implicit_bot_address
        ):
            removed_implicit_bot_address = True
            return " "
        return f" {_identity_label(record) if record else 'пользователь'} "

    def role_replacement(match: re.Match[str]) -> str:
        record = roles.get(int(match.group(1)))
        return f" {_identity_label(record) if record else 'роль'} "

    def channel_replacement(match: re.Match[str]) -> str:
        record = channels.get(int(match.group(1)))
        return f" {_identity_label(record) if record else 'канал'} "

    rendered = _USER_MENTION_RE.sub(user_replacement, content)
    rendered = _ROLE_MENTION_RE.sub(role_replacement, rendered)
    return _CHANNEL_MENTION_RE.sub(channel_replacement, rendered)


def analyze_entities(text: str, payload: Mapping[str, Any]) -> EntityAnalysis:
    """Classify known identities, addresses, actions and object-like terms."""

    normalized_text = text.casefold().replace("ё", "е")
    found: list[EntityMention] = []
    identity_records = (
        ("speaker", payload.get("author_identity")),
        ("addressee", payload.get("bot_identity")),
        ("replied_person", payload.get("reply_to_author_identity")),
        *(
            ("mentioned_person", item)
            for item in _mapping_sequence(payload.get("mentioned_users"))
        ),
        *(("mentioned_role", item) for item in _mapping_sequence(payload.get("mentioned_roles"))),
        *(
            ("mentioned_channel", item)
            for item in _mapping_sequence(payload.get("mentioned_channels"))
        ),
    )
    for role, raw_record in identity_records:
        record = raw_record if isinstance(raw_record, Mapping) else None
        if record is None:
            continue
        entity_id = _identity_id(record)
        kind = (
            "person"
            if "person" in role or role in {"speaker", "addressee", "replied_person"}
            else role.removeprefix("mentioned_")
        )
        for alias in _identity_aliases(record):
            normalized_alias = alias.casefold().replace("ё", "е").lstrip("@#")
            if not normalized_alias or not _contains_alias(normalized_text, normalized_alias):
                continue
            found.append(
                EntityMention(alias, normalized_alias, kind, role, entity_id, 0.99)
            )

    addressing = DEFAULT_ADDRESSING.analyze(text)
    for span in addressing.addresses:
        for lemma in span.lemmas:
            found.append(
                EntityMention(
                    span.text, lemma, "address", "addressee", None, span.confidence
                )
            )

    identity_terms = {item.canonical for item in found}
    actions: list[str] = []
    objects: list[str] = []
    for token in tokenize(addressing.content_text):
        if not token[:1].isalnum():
            continue
        readings = DEFAULT_MORPHOLOGY.parse_word(token)
        reading = readings[0] if readings else None
        lemma = reading.lemma if reading is not None else normalize_word(token)
        if lemma in identity_terms:
            continue
        if reading is not None and reading.part_of_speech == "verb":
            actions.append(lemma)
            continue
        if reading is not None and reading.part_of_speech == "noun":
            named = bool(reading.grammemes & {"Name", "Surn", "Patr"})
            if named:
                found.append(EntityMention(token, lemma, "person", "named_person", None, 0.88))
            else:
                # Grammatical animacy covers people, animals and many other
                # living objects. It is not evidence of a Discord identity or
                # a named person; keep the noun as an object until context
                # supplies a real identity link.
                objects.append(lemma)
        elif _looks_like_nickname(token):
            found.append(
                EntityMention(
                    token,
                    normalize_word(token),
                    "nickname",
                    "named_entity",
                    None,
                    0.72,
                )
            )

    return EntityAnalysis(
        mentions=tuple(dict.fromkeys(found)),
        action_terms=tuple(dict.fromkeys(actions)),
        object_terms=tuple(dict.fromkeys(objects)),
    )


def identity_thought(prompt: str, payload: Mapping[str, Any]) -> ClauseThought | None:
    """Realize a gateway name as a proposition rather than an authored reply."""
    # The structural parser consumes entities; load realization only at call time.
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )
    normalized = _normalized_identity_prompt(prompt)
    if _BOT_IDENTITY_RE.fullmatch(normalized):
        subject, record = "я", payload.get("bot_identity")
    elif _SPEAKER_IDENTITY_RE.fullmatch(normalized):
        subject, record = "ты", payload.get("author_identity")
    else:
        return None
    if not isinstance(record, Mapping) or not _identity_aliases(record):
        return None
    name = _identity_label(record)
    return ClauseThought(
        predicate=ConceptLexeme("называться", "называться", "verb", (("tense", "present"),)),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(
                ConceptLexeme(subject, subject, "pronoun"))),
            SemanticDependency("predicate_complement", ConceptArgument(
                ConceptLexeme(name, name, "noun", entity_kind="discord_identity"))),
        ), predicate_class="copular", confidence=0.99,
        evidence=(f"gateway_identity:{record.get('id')}:{name}",),
    )


def identity_reply(prompt: str, payload: Mapping[str, Any]) -> str | None:
    """Answer explicit Discord identity questions from gateway facts, not guesses."""

    normalized = _normalized_identity_prompt(prompt)
    if _SPEAKER_IDENTITY_RE.fullmatch(normalized):
        record = payload.get("author_identity")
        return f"Ты — {_describe_identity(record)}." if isinstance(record, Mapping) else None
    if _BOT_IDENTITY_RE.fullmatch(normalized):
        record = payload.get("bot_identity")
        return (
            f"В этом Discord я — {_describe_identity(record)}."
            if isinstance(record, Mapping)
            else None
        )
    match = _WHO_IS_RE.fullmatch(normalized)
    if match is None:
        return None
    target = match.group(1).strip(" @#.,!?…").casefold()
    candidates = (
        payload.get("author_identity"),
        payload.get("bot_identity"),
        payload.get("reply_to_author_identity"),
        *_mapping_sequence(payload.get("mentioned_users")),
    )
    for record in candidates:
        if not isinstance(record, Mapping):
            continue
        aliases = {alias.casefold().lstrip("@") for alias in _identity_aliases(record)}
        if target in aliases:
            return f"Это {_describe_identity(record)}."
    return None


def _describe_identity(raw: Mapping[str, Any]) -> str:
    display = _identity_label(raw)
    username = raw.get("username")
    if isinstance(username, str) and username and username != display:
        return f"{display} (@{username})"
    return f"@{username}" if isinstance(username, str) and username else display


def _normalized_identity_prompt(prompt: str) -> str:
    normalized = " ".join(
        prompt.casefold().replace("ё", "е").strip(" .!…,:;").split()
    )
    return _IDENTITY_DISCOURSE_PREFIX_RE.sub("", normalized)


def _identity_aliases(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value.strip()
            for key in ("display_name", "global_name", "username", "name")
            if isinstance((value := record.get(key)), str) and value.strip()
        )
    )


def _identity_label(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "пользователь"
    aliases = _identity_aliases(record)
    return aliases[0] if aliases else "пользователь"


def _identity_id(raw: object) -> int | None:
    return _integer(raw.get("id")) if isinstance(raw, Mapping) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping_sequence(raw: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _records_by_id(raw: object) -> dict[int, Mapping[str, Any]]:
    return {
        entity_id: item
        for item in _mapping_sequence(raw)
        if (entity_id := _identity_id(item)) is not None
    }


def _contains_alias(text: str, alias: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", text, re.IGNORECASE) is not None


def _looks_like_nickname(token: str) -> bool:
    return bool(
        len(token) >= 3
        and (
            "_" in token
            or any(character.isdigit() for character in token)
            or (
                any(character.islower() for character in token)
                and any(character.isupper() for character in token)
            )
        )
    )
