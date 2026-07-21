"""Predicate valency profiles used by the grounded response planner.

The profiles describe grammatical argument slots, not reply templates.  A
profile can therefore reject an isolated incomplete fact without deciding what
the bot should say.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValencySlot:
    role: str
    required_in_isolated_fact: bool = False
    grammatical_case: str | None = None
    prepositions: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PredicateValency:
    lemma: str
    slots: tuple[ValencySlot, ...]
    complement_roles: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ValencyAssessment:
    lemma: str
    profile_source: str
    missing_roles: tuple[str, ...]
    unexpected_roles: tuple[str, ...]
    confidence: float

    @property
    def complete(self) -> bool:
        return not self.missing_roles


_SUBJECT = ValencySlot("subject", required_in_isolated_fact=True, grammatical_case="nominative")
_DIRECT = ValencySlot(
    "direct_object",
    required_in_isolated_fact=True,
    grammatical_case="accusative",
)
_GENITIVE_OBJECT = ValencySlot(
    "direct_object",
    required_in_isolated_fact=True,
    grammatical_case="genitive",
)
_INSTRUMENTAL_OBJECT = ValencySlot(
    "direct_object",
    required_in_isolated_fact=True,
    grammatical_case="instrumental",
)
_INDIRECT = ValencySlot("indirect_object", grammatical_case="dative")
_REQUIRED_INDIRECT = ValencySlot(
    "indirect_object",
    required_in_isolated_fact=True,
    grammatical_case="dative",
)
_SOURCE = ValencySlot("source", prepositions=frozenset({"из", "от", "с", "со"}))
_DESTINATION = ValencySlot(
    "destination",
    prepositions=frozenset({"в", "во", "на", "к", "ко"}),
)
_LOCATION = ValencySlot(
    "location",
    prepositions=frozenset({"в", "во", "на", "под", "над", "за", "около", "возле"}),
)
_TOPIC = ValencySlot("topic", prepositions=frozenset({"о", "об", "обо", "про"}))
_COMPANION = ValencySlot("companion", prepositions=frozenset({"с", "со"}))
_INSTRUMENT = ValencySlot("instrument", prepositions=frozenset({"с", "со"}))
_ACTION_COMPLEMENT = ValencySlot(
    "action_complement",
    required_in_isolated_fact=True,
)
_OPTIONAL_ACTION_COMPLEMENT = ValencySlot("action_complement")


def _profile(
    lemma: str,
    *slots: ValencySlot,
    complement_roles: dict[str, str] | None = None,
) -> PredicateValency:
    return PredicateValency(
        lemma,
        slots,
        tuple((complement_roles or {}).items()),
    )


# A deliberately small bootstrap lexicon.  It describes Russian government
# patterns and can later be extended from the grammar repository.
_PROFILES = {
    item.lemma: item
    for item in (
        _profile(
            "брать", _SUBJECT, _DIRECT, _SOURCE, complement_roles={"с": "source", "со": "source"}
        ),
        _profile(
            "взять", _SUBJECT, _DIRECT, _SOURCE, complement_roles={"с": "source", "со": "source"}
        ),
        _profile("дать", _SUBJECT, _DIRECT, _INDIRECT),
        _profile(
            "положить",
            _SUBJECT,
            _DIRECT,
            _DESTINATION,
            complement_roles={"в": "destination", "во": "destination", "на": "destination"},
        ),
        _profile(
            "спрятать",
            _SUBJECT,
            _DIRECT,
            _DESTINATION,
            complement_roles={"в": "destination", "во": "destination", "на": "destination"},
        ),
        _profile("наносить", _SUBJECT, _DIRECT, _INDIRECT),
        _profile("любить", _SUBJECT, _DIRECT),
        _profile("иметь", _SUBJECT, _DIRECT),
        _profile("мочь", _SUBJECT, _ACTION_COMPLEMENT),
        _profile("уметь", _SUBJECT, _ACTION_COMPLEMENT),
        _profile("мешать", _SUBJECT, _INDIRECT, _ACTION_COMPLEMENT),
        _profile("нравиться", _SUBJECT, _INDIRECT),
        _profile("хотеть", _SUBJECT, _DIRECT, _OPTIONAL_ACTION_COMPLEMENT),
        _profile("понять", _SUBJECT, _DIRECT),
        _profile("предполагать", _SUBJECT, _DIRECT),
        _profile("предположить", _SUBJECT, _DIRECT),
        _profile(
            "считать",
            _SUBJECT,
            ValencySlot("direct_object", grammatical_case="accusative"),
            ValencySlot("predicate_complement"),
        ),
        _profile("поперхнуться", _SUBJECT),
        _profile(
            "иметься",
            _SUBJECT,
            ValencySlot("direct_object", grammatical_case="nominative"),
            _LOCATION,
        ),
        _profile("бояться", _SUBJECT, _GENITIVE_OBJECT),
        _profile("избегать", _SUBJECT, _GENITIVE_OBJECT),
        _profile("управлять", _SUBJECT, _INSTRUMENTAL_OBJECT),
        _profile("интересоваться", _SUBJECT, _INSTRUMENTAL_OBJECT),
        _profile("удивляться", _SUBJECT, _REQUIRED_INDIRECT),
        _profile("радоваться", _SUBJECT, _REQUIRED_INDIRECT),
        _profile("сочувствовать", _SUBJECT, _REQUIRED_INDIRECT),
        _profile("завидовать", _SUBJECT, _REQUIRED_INDIRECT),
        _profile(
            "жить",
            _SUBJECT,
            _LOCATION,
            complement_roles={"в": "location", "во": "location", "на": "location"},
        ),
        _profile(
            "лежать",
            _SUBJECT,
            _LOCATION,
            complement_roles={"в": "location", "во": "location", "на": "location"},
        ),
        _profile(
            "находиться",
            _SUBJECT,
            _LOCATION,
            complement_roles={"в": "location", "во": "location", "на": "location"},
        ),
        _profile(
            "говорить",
            _SUBJECT,
            _TOPIC,
            _COMPANION,
            complement_roles={"с": "companion", "со": "companion"},
        ),
        _profile(
            "работать",
            _SUBJECT,
            _INSTRUMENT,
            _COMPANION,
            complement_roles={"с": "instrument_or_companion", "со": "instrument_or_companion"},
        ),
    )
}


def predicate_valency(lemma: str) -> PredicateValency | None:
    return _PROFILES.get(lemma.casefold())


def resolve_complement_role(
    lemma: str,
    preposition: str,
    default_role: str,
) -> str:
    """Resolve an ambiguous prepositional role using predicate government."""

    profile = predicate_valency(lemma)
    if profile is None:
        return default_role
    roles = dict(profile.complement_roles)
    return roles.get(preposition.casefold(), default_role)


def governed_object_case(
    lemma: str,
    role: str,
    default: str,
) -> str:
    """Return the grammatical case governed by a lexical predicate slot."""

    profile = predicate_valency(lemma)
    if profile is None:
        return default
    return next(
        (
            slot.grammatical_case
            for slot in profile.slots
            if slot.role == role and slot.grammatical_case is not None
        ),
        default,
    )


def assess_predicate_valency(
    lemma: str,
    *,
    present_roles: frozenset[str],
    complement_roles: frozenset[str] = frozenset(),
    implicit_subject: bool = False,
    imperative: bool = False,
) -> ValencyAssessment:
    """Check whether an isolated proposition supplies its core arguments."""

    normalized = lemma.casefold()
    profile = predicate_valency(normalized)
    available = set(present_roles)
    available.update(complement_roles)
    if implicit_subject or imperative:
        available.add("subject")
    if profile is None:
        missing = () if "subject" in available else ("subject",)
        # Productive Russian reflexives are normally intransitive with regard
        # to an accusative object.  Unknown lexical exceptions may later get a
        # learned profile, but blindly accepting the object yields phrases
        # such as ``поперхнулся мозг``.
        unexpected = (
            ("direct_object",)
            if normalized.endswith(("ся", "сь")) and "direct_object" in present_roles
            else ()
        )
        return ValencyAssessment(
            normalized,
            "generic",
            missing,
            unexpected,
            0.36 if unexpected else 0.78 if not missing else 0.28,
        )

    expected = {slot.role for slot in profile.slots}
    missing = tuple(
        slot.role
        for slot in profile.slots
        if slot.required_in_isolated_fact and slot.role not in available
    )
    # Complements outside a lexical profile are still legal adjuncts.  Only
    # core object roles are useful as an anomaly signal here.
    unexpected = tuple(
        role
        for role in ("direct_object", "indirect_object", "action_complement")
        if role in present_roles and role not in expected
    )
    confidence = 0.96 - 0.34 * len(missing) - 0.08 * len(unexpected)
    return ValencyAssessment(
        normalized,
        "lexical_profile",
        missing,
        unexpected,
        max(0.0, min(1.0, confidence)),
    )
