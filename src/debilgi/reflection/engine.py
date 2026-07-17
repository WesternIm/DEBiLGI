"""Compare predicted and actual reply outcomes and derive reusable strategy lessons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from debilgi.experience.models import PreparedReply, UtteranceExperience
from debilgi.experience.repository import ExperienceRepository
from debilgi.reflection.models import Reflection, ReflectionGuidance
from debilgi.reflection.repository import ReflectionRepository


@dataclass(frozen=True, slots=True)
class ReflectionConfig:
    surprise_threshold: float = 0.25
    backfill_limit: int = 500


class ReflectionEngine:
    def __init__(
        self,
        repository: ReflectionRepository,
        experiences: ExperienceRepository,
        config: ReflectionConfig | None = None,
    ) -> None:
        self.repository = repository
        self._experiences = experiences
        self.config = config or ReflectionConfig()

    async def predict(self, guild_id: int, reply: PreparedReply) -> float:
        return await self.repository.predict(
            guild_id,
            strategies_for(
                reply.tokens,
                initiative=reply.initiative,
                reply_candidates=reply.candidate_count,
            ),
        )

    async def reflect(
        self,
        experience: UtteranceExperience,
        *,
        outcome_type: str,
        actual_reward: float,
        created_at: datetime,
    ) -> bool:
        reflection = build_reflection(
            experience,
            outcome_type=outcome_type,
            actual_reward=actual_reward,
            created_at=created_at,
            surprise_threshold=self.config.surprise_threshold,
        )
        return await self.repository.save(reflection)

    async def guidance(self, guild_id: int) -> ReflectionGuidance:
        return await self.repository.guidance(guild_id)

    async def backfill(self) -> int:
        inserted = 0
        experiences = await self._experiences.unreflected(self.config.backfill_limit)
        for experience in experiences:
            if (
                experience.reward is None
                or experience.outcome_type is None
                or experience.resolved_at is None
            ):
                continue
            inserted += await self.reflect(
                experience,
                outcome_type=experience.outcome_type,
                actual_reward=experience.reward,
                created_at=experience.resolved_at,
            )
        return inserted


def strategies_for(
    tokens: tuple[str, ...], *, initiative: bool, reply_candidates: int
) -> tuple[str, ...]:
    word_count = sum(any(character.isalnum() for character in token) for token in tokens)
    if word_count <= 8:
        length = "length:concise"
    elif word_count <= 24:
        length = "length:balanced"
    else:
        length = "length:extended"
    strategies = [
        "mode:initiative" if initiative else "mode:responsive",
        "act:question" if tokens[-1:] == ("?",) else "act:statement",
        length,
        "search:wide" if reply_candidates >= 48 else "search:normal",
    ]
    if any(token in {"!", "?!", "!?"} for token in tokens):
        strategies.append("tone:expressive")
    return tuple(strategies)


def build_reflection(
    experience: UtteranceExperience,
    *,
    outcome_type: str,
    actual_reward: float,
    created_at: datetime,
    surprise_threshold: float = 0.25,
) -> Reflection:
    expected = max(-1.0, min(1.0, experience.expected_reward))
    actual = max(-1.0, min(1.0, actual_reward))
    error = actual - expected
    surprise = abs(error)
    if actual < -0.10:
        lesson = "revise_strategy"
    elif error > surprise_threshold:
        lesson = "better_than_expected"
    elif error < -surprise_threshold:
        lesson = "worse_than_expected"
    else:
        lesson = "expectation_confirmed"
    strategies = strategies_for(
        experience.response_tokens,
        initiative=experience.initiative,
        reply_candidates=experience.candidate_count,
    )
    evidence = (
        f"outcome:{outcome_type}",
        f"expected:{expected:.3f}",
        f"actual:{actual:.3f}",
        *strategies,
    )
    return Reflection(
        reflection_id=f"reflection:{uuid4()}",
        experience_id=experience.experience_id,
        guild_id=experience.guild_id,
        user_id=experience.user_id,
        outcome_type=outcome_type,
        expected_reward=expected,
        actual_reward=actual,
        prediction_error=error,
        surprise=surprise,
        lesson_kind=lesson,
        strategies=strategies,
        evidence=evidence,
        created_at=created_at,
    )
