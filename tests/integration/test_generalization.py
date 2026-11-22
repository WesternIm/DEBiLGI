"""Independent humans, lexical hypotheses, privacy and immutable priors."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.lexicon import LexiconService
from debilgi.language.proposition_memory import (
    PropositionMemory,
    count_meaning_evidence,
    meaning_key,
)
from debilgi.language.semantic_graph import canonical_meanings, parse_proposition, proposition_key
from debilgi.personality.genome import DEBILGI_GENOME, SemanticProtection
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path):
    db = Database(tmp_path / "generalization.sqlite3", Path("migrations"))
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(index, text, actor=10, guild=1, channel=2):
    return Event(
        str(index),
        guild,
        channel,
        actor,
        None,
        EventType.MESSAGE_CREATED,
        NOW + timedelta(seconds=index),
        {"content": text},
        "discord",
    )


async def observe(database, index, text, actor=10, guild=1, channel=2):
    item = event(index, text, actor, guild, channel)
    await EventRepository(database).append(item)
    await PropositionMemory(database).observe(item)
    return item


async def test_independent_people_world_dispute_and_isolation(database):
    memory = PropositionMemory(database)
    for index in range(10):
        await observe(database, index, "Медведь ест рыбу.")
    rows = await database.fetchall("SELECT * FROM meaning_claims")
    assert len(rows) == 1
    key = (await database.fetchone("SELECT meaning_key FROM meaning_evidence"))[0]
    support = await memory.support(1, key)
    assert (support.people, support.supports, support.status) == (1, 1, "hypothesis")
    query = event(100, "Медведь ест рыбу?", 99)
    report = await memory.retrieve_claims(query)
    assert report.status == "Unknown" and report.reports[0].epistemic_status == "hypothesis"
    for actor in (11, 12):
        await observe(database, actor, "Медведь ест рыбу.", actor)
    result = await memory.retrieve_claims(query)
    assert result.status == "Known" and len(result.claims) == 1
    assert result.claims[0].epistemic_status == "fact"
    assert (await memory.retrieve_claims(replace(query, guild_id=2))).status == "Unknown"
    assert (await memory.retrieve_claims(replace(query, channel_id=3))).status == "Known"
    await observe(database, 20, "Медведь не ест рыбу.", 13)
    support = await memory.support(1, key)
    assert support.status == "disputed" and support.prevailing_polarity == 1
    assert (await memory.retrieve_claims(query)).status == "Unknown"
    assert (await memory.retrieve_claims(query)).reports[0].epistemic_status == "disputed"
    await database.close()
    await database.initialize()
    assert (await memory.support(1, key)).status == "disputed"
    for actor in (14, 15, 16):
        await observe(database, actor + 20, "Медведь не ест рыбу.", actor)
    assert (await memory.support(1, key)).prevailing_polarity == -1


async def test_personal_reports_never_cross_people_or_servers(database):
    await observe(database, 1, "Я читаю книгу.")
    own = event(20, "Я читаю книгу?")
    known = await PropositionMemory(database).retrieve_claims(own)
    assert known.status == "Known"
    assert (
        await PropositionMemory(database).query_claims(replace(known.query, speaker_id=99))
    ).status == "Unknown"
    other = event(20, "Он читает книгу?", 99)
    assert (await PropositionMemory(database).retrieve_claims(other)).status == "Unknown"
    assert (
        await PropositionMemory(database).retrieve_claims(replace(own, guild_id=2))
    ).status == "Unknown"


async def test_synonyms_require_humans_and_rekey_prior_evidence(database):
    memory, lexicon = PropositionMemory(database), LexiconService(database)
    await observe(database, 1, "Медведь ест рыбу.")
    await observe(database, 2, "Мишка ест рыбу.", 11)
    for index in (3, 4):
        item = event(index, "мишка то же что медведь", 10)
        await EventRepository(database).append(item)
        assert not await lexicon.observe_synonym(item, "мишка", "медведь")
    item = event(5, "мишка это медведь", 11)
    await EventRepository(database).append(item)
    assert await lexicon.observe_synonym(item, "мишка", "медведь")
    canon = await lexicon.canonical_lemmas(1)
    a, b = [parse_proposition(t) for t in ("Мишка ест рыбу.", "Медведь ест рыбу.")]
    assert a and b
    assert proposition_key(a, canonical=canon) == proposition_key(b, canonical=canon)
    assert proposition_key(a) == proposition_key(a, canonical={})
    assert not await lexicon.canonical_lemmas(2)
    assert (await memory.support(1, meaning_key(a, canon))).supports == 2
    assert (await database.fetchone("SELECT COUNT(*) FROM meaning_claims"))[0] == 1
    result = await memory.retrieve_claims(event(100, "Мишка ест рыбу?", 99))
    assert result.status == "Known"
    # Revoking a lexical bridge must not leave its merged authors attached
    # to the surviving word, including after raw input has been redacted.
    await database.execute(
        "UPDATE events SET payload_json=json_remove(payload_json,'$.content') "
        "WHERE event_id IN ('1','2')"
    )
    denial = event(6, "нет", 12)
    await EventRepository(database).append(denial)
    assert not await lexicon.observe_synonym(denial, "мишка", "медведь", contradicts=True)
    assert (await memory.support(1, meaning_key(a), public=True)).people == 1
    assert (await memory.support(1, meaning_key(b), public=True)).people == 1
    assert (await memory.retrieve_claims(event(100, "Мишка ест рыбу?", 99))).status == "Unknown"
    assert (await memory.retrieve_claims(event(100, "Медведь ест рыбу?", 99))).status == "Unknown"
    for pair in (("одеть", "надеть"), ("бежать", "прибежать")):
        assert not await lexicon.observe_synonym(item, *pair, explicit=True)
    await database.execute(
        "INSERT INTO language_synonym_links(guild_id,lemma,synonym_lemma,sense_id,source) "
        "VALUES (1,'левый','иной','sense-a','test'),(1,'правый','другой','sense-b','test')"
    )
    assert not await lexicon.observe_synonym(item, "левый", "правый", explicit=True)


async def test_explicit_confirmation_and_distribution_are_distinct(database):
    lexicon = LexiconService(database)
    item = event(1, "объяснение")
    await EventRepository(database).append(item)
    assert not await lexicon.observe_synonym(item, "мишка", "медведь", distributional=True)
    assert not await lexicon.canonical_lemmas(1)
    assert await lexicon.observe_synonym(item, "мишка", "медведь", explicit=True)
    item = event(2, "нет", 11)
    await EventRepository(database).append(item)
    assert not await lexicon.observe_synonym(item, "мишка", "медведь", contradicts=True)
    assert not await lexicon.canonical_lemmas(1)
    retraction = event(3, "да", 11)
    await EventRepository(database).append(retraction)
    assert not await lexicon.observe_synonym(retraction, "мишка", "медведь", explicit=True)
    assert not await lexicon.canonical_lemmas(1)
    with pytest.raises(ValueError):
        replace(retraction, guild_id=0)
    assert not await lexicon.canonical_lemmas(2)


async def test_experience_cannot_change_innate_weights(database):
    assert DEBILGI_GENOME.semantic_protections == ()
    rule = SemanticProtection(frozenset({"test:feature"}), 0.9)
    genome = replace(DEBILGI_GENOME, semantic_protections=(rule,))
    for index in range(20):
        await observe(database, index, "Медведь ест рыбу.", 10 + index)
    assert genome.protection_weight({"test:feature"}) == 0.9
    with pytest.raises(FrozenInstanceError):
        rule.weight = 0.1


async def test_source_reliability_is_capped_and_separate(database):
    memory = PropositionMemory(database)
    await observe(database, 1, "Медведь ест рыбу.", 10)
    for actor in (11, 12):
        await observe(database, actor, "Медведь не ест рыбу.", actor)
    assert 0.25 <= await memory.source_reliability(1, 10) < 1.0
    assert await memory.source_reliability(2, 10) == 1.0


async def test_ten_synonymous_paraphrases_are_one_human_vote(database):
    lexicon, memory = LexiconService(database), PropositionMemory(database)
    taught = event(1, "мишка это медведь", 90)
    await EventRepository(database).append(taught)
    assert await lexicon.observe_synonym(taught, "мишка", "медведь", explicit=True)
    variants = ("Мишка ест рыбу.", "Медведь ест рыбу.", "Рыбу ест медведь.")
    for index in range(10, 20):
        await observe(database, index, variants[index % 3])
    rows = await database.fetchall("SELECT DISTINCT meaning_key FROM meaning_evidence")
    assert len(rows) == 1
    support = await memory.support(1, rows[0][0])
    assert support.people == support.supports == 1 and not support.hypothesis_confirmed
    result = await memory.retrieve_claims(event(100, "Медведь ест рыбу?", 99))
    assert result.status == "Unknown" and result.reports


async def test_yes_must_reply_to_the_synonym_proposal(database):
    proposal = replace(
        event(1, "мишка это медведь"), payload={"content": "мишка это медведь", "message_id": 100}
    )
    await EventRepository(database).append(proposal)
    await PropositionMemory(database).observe(proposal)
    yes = replace(event(2, "да", 11), payload={"content": "да", "reply_to_message_id": 100})
    await EventRepository(database).append(yes)
    await PropositionMemory(database).observe(yes)
    assert (await LexiconService(database).canonical_lemmas(1))["мишка"] == "медведь"


async def test_canon_scope_is_restored_and_does_not_leak_between_tasks(database):
    import asyncio

    thought = parse_proposition("Мишка ест рыбу.")
    old_key = proposition_key(thought)

    async def scoped(mapping):
        with canonical_meanings(mapping):
            await asyncio.sleep(0)
            return proposition_key(thought)

    first, second = await asyncio.gather(scoped({"мишка": "медведь"}), scoped({}))
    assert first != second and second == old_key
    assert proposition_key(thought) == old_key


def test_established_quorum_requires_more_independent_challengers():
    rows = []

    def add(actor, polarity):
        index = len(rows)
        rows.append(
            {
                "actor_id": actor,
                "polarity": polarity,
                "source_event_id": str(index),
                "observed_at": (NOW + timedelta(days=index)).isoformat(),
            }
        )

    for actor in (10, 11, 12):
        add(actor, 1)
    for _ in range(10):
        add(13, -1)
    support = count_meaning_evidence(rows)
    assert (support.supports, support.contradicts, support.people) == (3, 1, 4)
    assert support.days == 13 and support.prevailing_polarity == 1
    # A supporter changes their mind: current 2:3 is still insufficient to
    # overturn the established three-person quorum.
    for actor in (10, 14):
        add(actor, -1)
    assert count_meaning_evidence(rows).prevailing_polarity == 1
    add(15, -1)
    support = count_meaning_evidence(rows)
    assert support.prevailing_polarity == -1 and support.status == "disputed"


async def test_shared_government_proposes_but_does_not_confirm(database):
    await database.execute(
        "INSERT INTO language_generation_word_relations VALUES "
        "(1,2,'видеть','медведь','direct_object','noun','accusative',3),"
        "(1,2,'видеть','мишка','direct_object','noun','accusative',3)"
    )
    item = event(1, "Вижу медведя.")
    await EventRepository(database).append(item)
    lexicon = LexiconService(database)
    await lexicon.observe_synonym_text(item, "Вижу медведя.")
    assert not await database.fetchone("SELECT 1 FROM continuous_cognition_hypotheses")
    await lexicon.discover_synonym_candidates()
    await lexicon.stop_synonym_candidates()
    row = await database.fetchone("SELECT * FROM continuous_cognition_hypotheses")
    assert row and row["signature"] == "synonym:медведь:мишка" and row["status"] == "working"
    assert row["positive_evidence"] == 0
    assert not await LexiconService(database).canonical_lemmas(1)


async def test_personal_reaffirmation_preserves_update_chronology(database):
    for index, text in enumerate(("Я читаю книгу.", "Я не читаю книгу.", "Я читаю книгу."), 1):
        await observe(database, index, text)
    result = await PropositionMemory(database).retrieve_claims(event(100, "Я читаю книгу?"))
    assert result.status == "Known" and len(result.claims) == 1
    assert not result.claims[0].thought.negated and result.claims[0].source_event_id == "3"


async def test_possessives_keep_concept_subjects_private(database):
    from debilgi.language.productive_nlg import ConceptLexeme
    from debilgi.language.proposition_memory import world_thought

    for actor in (10, 11):
        await observe(database, actor, "Мой кот ест рыбу.", actor)
    thought = parse_proposition("Кот ест рыбу.")
    thought = replace(
        thought,
        dependencies=tuple(
            replace(d, argument=replace(d.argument, modifiers=(ConceptLexeme("мой", "мой"),)))
            if d.role == "subject"
            else d
            for d in thought.dependencies
        ),
    )
    assert not world_thought(thought)
    result = await PropositionMemory(database).retrieve_claims(event(100, "Кот ест рыбу?", 99))
    assert result.status == "Unknown" and not result.reports


async def test_private_bindings_cannot_vote_on_an_identical_public_key(database):
    import json
    from dataclasses import asdict

    from debilgi.language.canonical_claims import argument_ref
    from debilgi.language.dialogue_obligations import thought_from_data

    memory = PropositionMemory(database)
    await observe(database, 1, "Медведь ест рыбу.", 10)
    original = dict(await database.fetchone("SELECT * FROM meaning_claims"))
    template = thought_from_data(json.loads(original["thought_json"]))
    # Participant bindings are attribution, deliberately outside the legacy
    # lexical key. A private participant can have the same displayed noun.
    for index, actor, owner in ((2, 11, 11), (3, 12, 12), (4, 10, 13)):
        source = event(index, "Личное сообщение", actor)
        await EventRepository(database).append(source)
        private = replace(
            template,
            negated=True,
            dependencies=tuple(
                replace(
                    d,
                    argument=replace(
                        d.argument,
                        head=replace(d.argument.head, participant_ref=f"discord:user:{owner}"),
                    ),
                )
                if d.role == "subject"
                else d
                for d in template.dependencies
            ),
        )
        claim = {
            **original,
            "claim_id": f"private:{owner}",
            "source_event_id": source.event_id,
            "subject_ref": f"discord:user:{owner}",
            "speaker_id": actor,
            "source_time": source.utc_timestamp.isoformat(),
            "source_kind": "self_report" if actor == owner else "other_report",
            "polarity": -1,
            "thought_json": json.dumps(asdict(private)),
        }
        await database.execute(
            "INSERT INTO meaning_claims ("
            + ",".join(claim)
            + ") VALUES ("
            + ",".join("?" for _ in claim)
            + ")",
            tuple(claim.values()),
        )
        for position, dep in enumerate(private.dependencies):
            await database.execute(
                "INSERT INTO meaning_arguments VALUES (?,?,?,?,?)",
                (
                    claim["claim_id"],
                    position,
                    dep.role,
                    argument_ref(dep.argument),
                    json.dumps(asdict(dep.argument)),
                ),
            )
        await database.execute(
            "INSERT INTO meaning_evidence(claim_id,source_event_id,source_key) VALUES (?,?,?)",
            (claim["claim_id"], source.event_id, "event"),
        )
    await memory.reindex_evidence()
    key = meaning_key(template)
    public = await memory.support(1, key, public=True)
    assert (public.people, public.supports, public.contradicts) == (1, 1, 0)
    assert await memory.source_reliability(1, 10) == 1.0
    result = await memory.retrieve_claims(event(100, "Медведь ест рыбу?", 99))
    assert result.status == "Unknown" and result.reports[0].epistemic_status == "hypothesis"
    other_person = replace(result.query, speaker_id=10, arguments=(("subject", "discord:user:13"),))
    assert (await memory.query_claims(other_person)).status == "Unknown"


async def test_dead_tables_are_absent_and_support_uses_index(database):
    tables = {row[0] for row in await database.fetchall("SELECT name FROM sqlite_master")}
    assert "meaning_evidence_support" in tables
    assert not tables & {
        "language_generation_ngrams",
        "language_generation_reader_ngrams",
        "language_generation_observations",
        "language_character_transitions",
        "language_experience_transitions",
    }
    plan = await database.fetchall(
        "EXPLAIN QUERY PLAN SELECT * FROM meaning_evidence WHERE guild_id=? AND meaning_key=?",
        (1, "test"),
    )
    assert "meaning_evidence_support" in str([tuple(r) for r in plan])
