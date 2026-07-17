"""Show the social primitive vector for a few diagnostic utterances."""

from __future__ import annotations

from datetime import UTC, datetime

from debilgi.events.models import Event, EventType
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.social.engine import assess_reaction

PROBES = (
    "да, это отличный результат",
    "нет, я отказываюсь",
    "почему это мерзко?",
    "мне интересно, но противно",
)


def main() -> None:
    for index, phrase in enumerate(PROBES):
        event = Event(
            event_id=f"social-probe:{index}",
            guild_id=1,
            channel_id=1,
            actor_id=1,
            target_id=None,
            event_type=EventType.MESSAGE_CREATED,
            timestamp=datetime.now(UTC),
            payload={"content": phrase},
            source="local-social-probe",
        )
        reaction = assess_reaction(event, None, DEBILGI_GENOME)
        strengths = ", ".join(
            f"{name}={value:.2f}" for name, value in reaction.vector.ranked()[:3]
        )
        print(
            f"{phrase} -> {reaction.blend_id} | {strengths} | "
            f"confidence={reaction.confidence:.2f}"
        )


if __name__ == "__main__":
    main()
