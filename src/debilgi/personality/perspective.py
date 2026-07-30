"""A unified, inspectable point of view used before opinion and wording.

The genome, momentary phenotype, relationship history and learned outcome
guidance previously influenced separate stages.  This module combines them
into one transient perspective.  It contains drives and appraisals, never
authored replies.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from debilgi.personality.expression import PersonalityExpression
from debilgi.reflection.models import ReflectionGuidance
from debilgi.social.models import RelationState, SocialReaction

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")

# Semantic qualities are connected to abstract personality drives.  These are
# value associations, not polarity labels or response strings: the resulting
# score still depends on the current perspective.
_VALUE_CUES: dict[str, tuple[tuple[str, float, float], ...]] = {
    "интересный": (("curiosity", 1.0, 0.78),),
    "необычный": (("novelty", 1.0, 0.82), ("curiosity", 1.0, 0.34)),
    "новый": (("novelty", 1.0, 0.72),),
    "сложный": (("curiosity", 1.0, 0.56), ("patience", 1.0, 0.28)),
    "оригинальный": (("novelty", 1.0, 0.80),),
    "яркий": (("expressiveness", 1.0, 0.76),),
    "эмоциональный": (("expressiveness", 1.0, 0.66),),
    "смелый": (("assertiveness", 1.0, 0.74),),
    "дерзкий": (("assertiveness", 1.0, 0.62), ("playfulness", 1.0, 0.30)),
    "свободный": (("independence", 1.0, 0.76),),
    "независимый": (("independence", 1.0, 0.82),),
    "логичный": (("coherence", 1.0, 0.74),),
    "связный": (("coherence", 1.0, 0.68),),
    "ясный": (("coherence", 1.0, 0.62),),
    "надежный": (("trust", 1.0, 0.70), ("coherence", 1.0, 0.24)),
    "открытый": (("sociability", 1.0, 0.64),),
    "дружелюбный": (("warmth", 1.0, 0.72),),
    "крутой": (
        ("expressiveness", 1.0, 0.56),
        ("assertiveness", 1.0, 0.46),
        ("playfulness", 1.0, 0.30),
    ),
    "умный": (("curiosity", 1.0, 0.54), ("coherence", 1.0, 0.48)),
    "герой": (("assertiveness", 1.0, 0.64), ("sociability", 1.0, 0.22)),
    "скучный": (("novelty", -1.0, 0.82), ("curiosity", -1.0, 0.34)),
    "однообразный": (("novelty", -1.0, 0.78),),
    "предсказуемый": (("novelty", -1.0, 0.46), ("coherence", 1.0, 0.24)),
    "хаотичный": (("playfulness", 1.0, 0.34), ("coherence", -1.0, 0.48)),
    "навязчивый": (("independence", -1.0, 0.76), ("resistance", 1.0, 0.34)),
    "принудительный": (("independence", -1.0, 0.86),),
    "фальшивый": (("trust", -1.0, 0.78), ("skepticism", 1.0, 0.32)),
    "опасный": (("caution", -1.0, 0.62), ("assertiveness", 1.0, 0.20)),
    "рискованный": (("caution", -1.0, 0.54), ("playfulness", 1.0, 0.22)),
}


@dataclass(frozen=True, slots=True)
class PerspectiveAppraisal:
    score: float
    confidence: float
    dimensions: tuple[str, ...]
    terms: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersonalOpinion:
    """A digested stance that exists before a dialogue move or sentence."""

    subject: str
    quality: str
    stance: str
    score: float
    confidence: float
    evidence: tuple[str, ...]
    value_dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersonalityPerspective:
    """The agent's current point of view, prior to any semantic decision."""

    genome_signature: str
    sociability: float
    curiosity: float
    expressiveness: float
    novelty: float
    coherence: float
    assertiveness: float
    independence: float
    skepticism: float
    patience: float
    warmth: float
    playfulness: float
    memory_attachment: float
    trust: float
    affinity: float
    resistance: float
    arousal: float
    caution: float
    exploration: float
    confidence: float
    evidence: tuple[str, ...]

    def drive(self, name: str) -> float:
        aliases = {
            "novelty_drive": "novelty",
            "coherence_drive": "coherence",
            "suspicion": "skepticism",
        }
        value = getattr(self, aliases.get(name, name), 0.5)
        return _clamp(float(value))

    def appraise(self, terms: tuple[str, ...]) -> PerspectiveAppraisal:
        """Evaluate semantic qualities through this perspective."""

        lemmas = tuple(
            dict.fromkeys(
                token.casefold().replace("ё", "е")
                for term in terms
                for token in _TOKEN_RE.findall(term)
            )
        )
        contributions: list[tuple[float, str, str]] = []
        for lemma in lemmas:
            for dimension, orientation, intensity in _VALUE_CUES.get(lemma, ()):
                drive = self.drive(dimension)
                contribution = orientation * intensity * (0.30 + drive * 0.70)
                contributions.append((contribution, dimension, lemma))
        if not contributions:
            return PerspectiveAppraisal(
                0.0,
                min(0.52, self.confidence * 0.68),
                (),
                (),
                ("personality:no_semantic_value_link",),
            )
        score = _clamp(
            sum(item[0] for item in contributions)
            / max(1.0, len(contributions) ** 0.62),
            -1.0,
            1.0,
        )
        dimensions = tuple(dict.fromkeys(item[1] for item in contributions))
        matched_terms = tuple(dict.fromkeys(item[2] for item in contributions))
        confidence = _clamp(
            0.42 + self.confidence * 0.28 + min(0.24, len(contributions) * 0.07)
        )
        return PerspectiveAppraisal(
            score,
            confidence,
            dimensions,
            matched_terms,
            tuple(
                dict.fromkeys(
                    f"personality:{dimension}:{lemma}:{value:+.3f}"
                    for value, dimension, lemma in contributions
                )
            ),
        )

    def candidate_fit(self, move: str) -> float:
        """Score a communicative move by how naturally it follows from ``self``."""

        normalized = move.casefold().replace("-", "_")
        if any(marker in normalized for marker in ("qualify", "withhold", "evidence")):
            return _clamp(self.caution * 0.52 + self.skepticism * 0.34 + self.coherence * 0.14)
        if any(
            marker in normalized
            for marker in ("position", "disagree", "challenge", "reject", "decide")
        ):
            return _clamp(
                self.assertiveness * 0.48
                + self.independence * 0.34
                + self.coherence * 0.18
            )
        if any(
            marker in normalized
            for marker in ("clarify", "probe", "question", "continue", "inspect")
        ):
            return _clamp(
                self.curiosity * 0.46
                + self.exploration * 0.34
                + self.patience * 0.20
            )
        if any(marker in normalized for marker in ("acknowledge", "reciprocate", "support")):
            return _clamp(self.warmth * 0.46 + self.affinity * 0.34 + self.trust * 0.20)
        if "play" in normalized:
            return _clamp(
                self.playfulness * 0.58
                + self.assertiveness * 0.22
                + self.arousal * 0.20
            )
        if any(marker in normalized for marker in ("answer", "inform", "explain")):
            return _clamp(
                self.coherence * 0.44
                + self.assertiveness * 0.26
                + self.curiosity * 0.18
                + self.trust * 0.12
            )
        return _clamp(self.coherence * 0.55 + self.warmth * 0.20 + self.assertiveness * 0.25)

    def trace(self) -> dict[str, object]:
        return asdict(self)


def build_personality_perspective(
    expression: PersonalityExpression,
    *,
    reaction: SocialReaction | None = None,
    relation: RelationState | None = None,
    reflection: ReflectionGuidance | None = None,
) -> PersonalityPerspective:
    """Digest innate, momentary, relational and learned signals into one self."""

    current = reaction.vector if reaction is not None else None
    baseline = relation.vector if relation is not None else None
    relation_weight = (
        min(0.72, 0.18 + relation.observations / 24.0)
        if relation is not None
        else 0.0
    )

    def social(name: str, fallback: float) -> float:
        current_value = float(getattr(current, name)) if current is not None else fallback
        if baseline is None:
            return _clamp(current_value)
        return _clamp(
            current_value * (1.0 - relation_weight)
            + float(getattr(baseline, name)) * relation_weight
        )

    agreement = social("agreement", 0.5)
    approval = social("approval", 0.5)
    trust = social("trust", expression.warmth * 0.62)
    resistance = social("resistance", expression.suspicion * 0.46)
    aversion = social("aversion", expression.suspicion * 0.28)
    arousal = social("arousal", expression.expressiveness * 0.44)
    affinity = _clamp(
        0.5
        + (agreement + approval + trust - resistance - aversion - 0.5) * 0.24
    )
    reflection_caution = reflection.caution if reflection is not None else 0.0
    reflection_exploration = reflection.exploration if reflection is not None else 0.0
    caution = _clamp(
        expression.suspicion * 0.38
        + (1.0 - expression.assertiveness) * 0.18
        + reflection_caution * 0.44
    )
    exploration = _clamp(
        expression.novelty_drive * 0.40
        + expression.curiosity * 0.32
        + reflection_exploration * 0.28
    )
    warmth = _clamp(
        expression.warmth * 0.48 + affinity * 0.30 + trust * 0.22
    )
    skepticism = _clamp(
        expression.suspicion * 0.48
        + resistance * 0.28
        + expression.uncertainty * 0.24
    )
    assertiveness = _clamp(
        expression.assertiveness * 0.58
        + arousal * 0.18
        + (1.0 - caution) * 0.24
    )
    independence = _clamp(
        assertiveness * 0.42
        + expression.novelty_drive * 0.28
        + expression.suspicion * 0.20
        + (1.0 - affinity) * 0.10
    )
    playfulness = _clamp(
        expression.playfulness * 0.56
        + arousal * 0.20
        + exploration * 0.24
    )
    confidence = _clamp(
        0.48
        + (reaction.confidence if reaction is not None else 0.0) * 0.14
        + min(0.18, (relation.observations if relation is not None else 0) / 80.0)
        + (reflection.confidence if reflection is not None else 0.0) * 0.16
    )
    evidence = (
        f"genome:{expression.genome_signature}",
        "phenotype:current_cognitive_state",
        *(
            (
                f"reaction:{reaction.blend_id}:{reaction.confidence:.3f}",
            )
            if reaction is not None
            else ()
        ),
        *(
            (
                f"relation:observations:{relation.observations}",
            )
            if relation is not None
            else ()
        ),
        *(
            (
                f"reflection:{reflection.strongest_strategy or 'general'}:"
                f"{reflection.confidence:.3f}",
            )
            if reflection is not None
            else ()
        ),
    )
    return PersonalityPerspective(
        genome_signature=expression.genome_signature,
        sociability=expression.sociability,
        curiosity=expression.curiosity,
        expressiveness=expression.expressiveness,
        novelty=expression.novelty_drive,
        coherence=expression.coherence_drive,
        assertiveness=assertiveness,
        independence=independence,
        skepticism=skepticism,
        patience=expression.patience,
        warmth=warmth,
        playfulness=playfulness,
        memory_attachment=expression.memory_attachment,
        trust=trust,
        affinity=affinity,
        resistance=resistance,
        arousal=arousal,
        caution=caution,
        exploration=exploration,
        confidence=confidence,
        evidence=evidence,
    )


def form_personal_opinion(
    subject: str,
    quality: str,
    *,
    perspective: PersonalityPerspective | None,
    situational_evidence: tuple[tuple[float, float, str], ...] = (),
) -> PersonalOpinion:
    """Digest values and situational evidence into an owned, revisable stance."""

    weighted = list(situational_evidence)
    appraisal = perspective.appraise((quality,)) if perspective is not None else None
    if (
        appraisal is not None
        and appraisal.dimensions
        and abs(appraisal.score) >= 0.10
    ):
        weighted.append(
            (
                appraisal.score,
                0.34 + appraisal.confidence * 0.34,
                "personality_value_appraisal",
            )
        )
    if not weighted:
        return PersonalOpinion(
            subject=subject,
            quality=quality,
            stance="withhold",
            score=0.0,
            confidence=0.38,
            evidence=("opinion:no_personal_or_situational_basis",),
            value_dimensions=(),
        )
    numerator = sum(score * weight for score, weight, _ in weighted)
    denominator = sum(weight for _, weight, _ in weighted)
    score = _clamp(numerator / max(0.01, denominator), -1.0, 1.0)
    stance = "support" if score >= 0.12 else "oppose" if score <= -0.12 else "withhold"
    confidence = _clamp(
        0.38
        + min(0.36, denominator / max(1.0, len(weighted)) * 0.34)
        + (perspective.confidence if perspective is not None else 0.0) * 0.18
    )
    return PersonalOpinion(
        subject=subject,
        quality=quality,
        stance=stance,
        score=score,
        confidence=confidence,
        evidence=tuple(
            dict.fromkeys(
                (
                    *(label for _, _, label in weighted),
                    *(appraisal.evidence if appraisal is not None else ()),
                )
            )
        ),
        value_dimensions=appraisal.dimensions if appraisal is not None else (),
    )


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
