from datetime import UTC, datetime, timedelta

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.context_gate import WorkingMemoryContextGate

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


class RelatedConcepts:
    def __init__(self, relations: dict[str, tuple[str, ...]] | None = None) -> None:
        self.relations = relations or {}

    async def related_concepts(
        self, guild_id: int, concept: str, *, limit: int = 20
    ) -> tuple[str, ...]:
        assert guild_id == 1
        del limit
        return self.relations.get(concept, ())


def message(
    message_id: int,
    text: str,
    *,
    author_id: int = 20,
    author_is_bot: bool = False,
    mentioned: bool = True,
):
    return normalize_message_created(
        message_id=message_id,
        guild_id=1,
        channel_id=10,
        author_id=author_id,
        created_at=NOW + timedelta(seconds=message_id),
        content=text,
        author_is_bot=author_is_bot,
        bot_mentioned=mentioned,
        bot_user_id=999,
    )


async def test_working_focus_is_bounded_and_keeps_turn_metadata() -> None:
    gate = WorkingMemoryContextGate(RelatedConcepts(), window_size=3)  # type: ignore[arg-type]
    await gate.observe(message(1, "обсудим старую музыку", mentioned=False))
    await gate.observe(message(2, "теперь поговорим про Байкал", mentioned=False))
    await gate.observe(message(3, "какая глубина у озера", mentioned=False))
    await gate.observe(message(4, "проверь этот факт", mentioned=True))
    await gate.set_pending_question(1, 10, 20, "батискаф")

    focus = await gate.snapshot(1, 10)

    assert len(focus.turns) == 3
    assert all("музыка" not in turn.concepts for turn in focus.turns)
    assert {"байкал", "озеро", "батискаф"} <= set(focus.concepts)
    assert focus.last_actor_id == 20
    assert focus.last_addressed_to_bot is True


async def test_reactive_candidate_is_passed_or_replaced_by_one_gate() -> None:
    gate = WorkingMemoryContextGate(RelatedConcepts())  # type: ignore[arg-type]
    await gate.observe(message(1, "расскажи про озеро Байкал"))

    relevant = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Байкал — глубокое пресноводное озеро.",
        source="retrieval",
        now=NOW + timedelta(minutes=1),
    )
    unrelated = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Бухта приключений находится далеко отсюда.",
        source="graph",
        now=NOW + timedelta(minutes=1),
    )

    assert relevant.action == "passed"
    assert "байкал" in relevant.matched_concepts
    assert unrelated.action == "replaced"
    assert unrelated.text is not None and unrelated.text.endswith("?")
    assert any(topic in unrelated.text.casefold() for topic in ("байкал", "озеро"))
    assert "Бухта" not in unrelated.text


async def test_graph_related_concept_counts_as_relevant() -> None:
    dataset = RelatedConcepts({"клоунесса": ("артистка", "цирк")})
    gate = WorkingMemoryContextGate(dataset)  # type: ignore[arg-type]
    await gate.observe(message(1, "что такое клоунесса"))

    decision = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Артистка исполняет комические роли в цирке.",
        source="retrieval",
    )

    assert decision.action == "passed"
    assert {"артистка", "цирк"} & set(decision.matched_concepts)


async def test_candidate_pool_receives_graded_relevance_from_one_focus() -> None:
    dataset = RelatedConcepts({"футбол": ("команда", "мяч")})
    gate = WorkingMemoryContextGate(dataset)  # type: ignore[arg-type]
    await gate.observe(message(1, "А футбол любишь?"))

    relevant, related, unrelated = await gate.measure_candidates(
        guild_id=1,
        channel_id=10,
        candidate_texts=(
            "Футбол мне интересен как тема.",
            "Командная игра с мячом бывает напряжённой.",
            "Американский телесериал вышел в 1987 году.",
        ),
    )

    assert relevant.score > related.score > unrelated.score
    assert relevant.focus_concepts == related.focus_concepts == unrelated.focus_concepts
    assert {"команда", "мяч"} & set(related.related_concepts)


async def test_background_fact_waits_for_channel_pause_when_unrelated() -> None:
    gate = WorkingMemoryContextGate(
        RelatedConcepts(),  # type: ignore[arg-type]
        initiative_idle=timedelta(minutes=5),
    )
    await gate.observe(message(1, "обсуждаем косинусы и углы"))

    busy = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Кстати, разобрался, что значит календарь.",
        source="background",
        now=NOW + timedelta(minutes=1),
    )
    idle = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Кстати, разобрался, что значит календарь.",
        source="background",
        now=NOW + timedelta(minutes=10),
    )

    assert busy.action == "deferred"
    assert idle.action == "passed"
    assert idle.reason == "channel_idle"


async def test_short_act_reads_the_same_last_bot_turn_and_bypasses_topic_guessing() -> None:
    gate = WorkingMemoryContextGate(RelatedConcepts())  # type: ignore[arg-type]
    await gate.observe(message(1, "объясни слово ключ"))
    await gate.commit_outgoing(
        guild_id=1,
        channel_id=10,
        message_id=500,
        text="Ключ здесь означает предмет для отпирания замка.",
        bot_user_id=999,
        target_user_id=20,
        observed_at=NOW + timedelta(seconds=2),
    )

    last_bot = await gate.last_bot_turn(1, 10, user_id=20)
    decision = await gate.evaluate(
        guild_id=1,
        channel_id=10,
        candidate_text="Принято. Предыдущий ответ подтверждён.",
        source="short_act",
    )

    assert last_bot is not None and last_bot.message_id == 500
    assert decision.action == "passed"
    assert decision.reason == "bound_to_current_turn"
