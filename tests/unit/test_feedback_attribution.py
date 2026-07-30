from datetime import UTC, datetime

from debilgi.events.models import Event, EventType
from debilgi.experience.feedback import FeedbackAttributor

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def message(content: str, *, reply_to: int = 99) -> Event:
    return Event(
        event_id=f"feedback:{content}",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW,
        payload={
            "content": content,
            "reply_to_message_id": reply_to,
            "author_is_bot": False,
        },
        source="test",
    )


def test_plain_reply_is_attributed_but_neutral() -> None:
    feedback = FeedbackAttributor().attribute(message("А почему ты так думаешь?"))

    assert feedback.target_message_id == 99
    assert feedback.kind == "conversation_continued"
    assert feedback.reward == 0.0
    assert not feedback.is_learnable()


def test_explicit_rejection_wins_over_direct_reply_shape() -> None:
    feedback = FeedbackAttributor().attribute(
        message("Нет, ты вообще ничего не поняла")
    )

    assert feedback.kind == "explicit_rejection"
    assert feedback.reward == -1.0
    assert feedback.attribution_confidence == 0.95
    assert feedback.is_learnable()


def test_explicit_confirmation_is_the_only_strong_positive_text_signal() -> None:
    feedback = FeedbackAttributor().attribute(message("Да, всё правильно"))

    assert feedback.kind == "explicit_confirmation"
    assert feedback.reward == 1.0
    assert feedback.is_learnable()
