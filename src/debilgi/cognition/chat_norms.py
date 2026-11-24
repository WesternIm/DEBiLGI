"""C — the mentality of the chat (4.43, flag ``DEBILGI_THOUGHT_CONTENT`` only).

What a chat treats as register and what as an attack is read from how *other people* react to a
turn — never from a word list and never from one person. Source: ``language_observed_exchanges``
(a turn of one live human and the answer of another; seed corpus and bots excluded — C starts
empty). For every word of the turn the reacting person casts one vote: friendly (laughter) or
hostile (an attack on the speaker, ``hostility_cues``). A word has a sign here only when ≥2 distinct
people voted the same way and they outweigh the other side.

Storage: ``continuous_cognition_hypotheses`` (signature ``chat:<kind>:<key>``, subject
``discord:guild:<id>``, status ``hypothesis``). A norm is a hypothesis, never a claim, never
carried between servers; it stores counts of people, never who. Written by sleep (consolidation);
the day only reads.

Horizon: a vote weighs ``2^(-age/T_S)`` with T_S ≈ 3.6 days (the horizon of what people did) and is
read over ``6·T_S`` — the same convention as the irritation window (``irritation_reading``): a
week-old vote still weighs ≈ 0.26, a month-old one is gone.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SIGNATURE = "chat:"
FRIENDLY, HOSTILE = "friendly", "hostile"


def _lemmas(text: str) -> frozenset[str]:
    """Words of a turn a reaction can be about: content lemmas and dictionary slang."""
    from debilgi.cognition.affect import strip_mentions
    from debilgi.language.content import content_lemmas
    from debilgi.language.dialogue_learning import exchange_analysis
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    text = strip_mentions(text)
    if not text:
        return frozenset()
    found = set(content_lemmas(text))
    for token in exchange_analysis(text)[0].tokens:
        if token.surface[:1].isalpha() and any(
                "Slng" in r.grammemes for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)):
            found.add(token.lemma)
    return frozenset(found)


def reaction_kind(prompt: str, answer: str) -> str | None:
    """How the other person took a turn: hostile (an attack on the speaker — read after an attack
    when the turn itself was one), friendly (laughter) or None (no signal)."""
    from debilgi.cognition.affect import hostility_cues
    from debilgi.language.pragmatics import analyze_pragmatics

    if hostility_cues(answer, bool(hostility_cues(prompt))):
        return HOSTILE
    if "amusement" in analyze_pragmatics(answer).functions:
        return FRIENDLY
    return None


@dataclass(frozen=True)
class Norm:
    norm_id: str
    kind: str
    key: str
    sign: str
    people: int
    confidence: float


@dataclass(frozen=True)
class ChatNorms:
    norms: tuple[Norm, ...] = ()
    _words: dict = field(default_factory=dict, compare=False, repr=False)

    def word(self, lemma: str) -> Norm | None:
        return self._words.get(lemma)

    def signs(self) -> tuple[tuple[str, int], ...]:
        """Hashable (lemma, +1 friendly / −1 hostile) for ``hostility_cues``."""
        return tuple(sorted((n.key, 1 if n.sign == FRIENDLY else -1)
                            for n in self.norms if n.kind == "word"))

    def tones(self) -> tuple[str, ...]:
        return tuple(sorted(n.key for n in self.norms if n.kind == "tone"))

    def barred(self, lemmas) -> tuple[Norm, ...]:
        """Hostile-signed words among these: topics people here answer with hostility."""
        return tuple(n for n in (self._words.get(lemma) for lemma in sorted(lemmas))
                     if n is not None and n.sign == HOSTILE)

    def about(self, lemmas) -> list[dict]:
        """For the choice basis: norm ids and how many people, not who."""
        return [{"id": n.norm_id, "kind": n.kind, "word": n.key, "sign": n.sign,
                 "people": n.people, "confidence": round(n.confidence, 4)}
                for n in (self._words.get(lemma) for lemma in sorted(lemmas)) if n is not None]


async def load(database, guild_id: int | None) -> ChatNorms:
    if guild_id is None:
        return ChatNorms()
    rows = await database.fetchall(
        "SELECT hypothesis_id, predicate_key, object_key, roles_json, confidence "
        "FROM continuous_cognition_hypotheses WHERE guild_id=? AND status!='rejected' "
        "AND subject_key=? AND predicate_key LIKE 'chat_norm:%' ORDER BY hypothesis_id",
        (guild_id, f"discord:guild:{guild_id}"))
    norms = []
    for row in rows:
        data = json.loads(row["roles_json"])
        norms.append(Norm(row["hypothesis_id"], data["kind"], data["key"], row["object_key"],
                          int(data["people"]), float(row["confidence"])))
    return ChatNorms(tuple(norms), {n.key: n for n in norms if n.kind == "word"})


async def consolidate(database, guild_id: int, now: datetime) -> int:
    """Sleep: recount the norms of one server from the exchanges of the last 6·T_S."""
    from debilgi.experience.meaning_intents import social_half_life

    half_life = social_half_life()
    rows = await database.fetchall(
        """SELECT p.content AS prompt, a.content AS answer, a.actor_id AS reactor,
                  x.observed_at AS observed_at
           FROM language_observed_exchanges x
           JOIN language_discourse_turns p ON p.source_id = x.prompt_event_id
           JOIN language_discourse_turns a ON a.source_id = x.answer_event_id
           JOIN events pe ON pe.event_id = p.source_id
           JOIN events ae ON ae.event_id = a.source_id
           WHERE x.guild_id = ? AND x.observed_at < ? AND x.observed_at >= ?
             AND p.actor_id IS NOT NULL AND a.actor_id IS NOT NULL AND p.actor_id != a.actor_id
             AND COALESCE(json_extract(pe.payload_json,'$.author_is_bot'),0)=0
             AND COALESCE(json_extract(ae.payload_json,'$.author_is_bot'),0)=0
             AND pe.source NOT LIKE 'seed%' AND pe.source NOT LIKE '%corpus%'
             AND ae.source NOT LIKE 'seed%' AND ae.source NOT LIKE '%corpus%'
           ORDER BY x.observed_at, x.answer_event_id LIMIT 8192""",
        (guild_id, now.isoformat(), (now - timedelta(seconds=6 * half_life)).isoformat()))
    # one person, one vote per word and side (the latest reaction of theirs weighs)
    votes: dict[str, dict[str, dict[int, float]]] = {}
    for row in rows:
        kind = reaction_kind(row["prompt"] or "", row["answer"] or "")
        if kind is None:
            continue
        age = max(0.0, (now - datetime.fromisoformat(row["observed_at"])).total_seconds())
        weight = 2.0 ** (-age / half_life)
        for lemma in _lemmas(row["prompt"] or ""):
            side = votes.setdefault(lemma, {FRIENDLY: {}, HOSTILE: {}})[kind]
            side[row["reactor"]] = max(side.get(row["reactor"], 0.0), weight)
    wanted: dict[str, tuple] = {}
    for lemma, sides in votes.items():
        mass = {k: sum(v.values()) for k, v in sides.items()}
        sign = FRIENDLY if mass[FRIENDLY] > mass[HOSTILE] else HOSTILE
        other = HOSTILE if sign == FRIENDLY else FRIENDLY
        people = len(sides[sign])
        if people < 2 or mass[sign] <= mass[other]:
            continue  # one person is not a chat; a balance is no sign
        wanted[f"chat:word:{lemma}"] = ("word", lemma, sign, people, len(sides[other]),
                                        mass[sign] / (mass[sign] + mass[other] + 1.0))
    wanted.update(await _tone_norms(database, guild_id, now, half_life))
    return await _store(database, guild_id, now, wanted)


async def _tone_norms(database, guild_id: int, now: datetime, half_life: float) -> dict:
    """The tone that ≥2 different people each keep in at least half of their turns (the chat's
    habitual tone, a background for «this person changed tone», not a rule)."""
    from debilgi.language.dialogue_acts import analyze_turn_form

    rows = await database.fetchall(
        """SELECT d.actor_id AS actor, d.content AS content
           FROM language_discourse_turns d JOIN events e ON e.event_id = d.source_id
           WHERE d.guild_id = ? AND d.observed_at < ? AND d.observed_at >= ?
             AND d.actor_id IS NOT NULL
             AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0
             AND e.source NOT LIKE 'seed%' AND e.source NOT LIKE '%corpus%'
           ORDER BY d.observed_at DESC LIMIT 4096""",
        (guild_id, now.isoformat(), (now - timedelta(seconds=6 * half_life)).isoformat()))
    per_person: dict[int, list[frozenset]] = {}
    for row in rows:
        per_person.setdefault(row["actor"], []).append(
            frozenset(analyze_turn_form(row["content"] or "").tone))
    habitual: dict[str, set[int]] = {}
    for actor, turns in per_person.items():
        if len(turns) < 2:
            continue  # a habit is not one turn
        for tone in set().union(*turns):
            if sum(tone in t for t in turns) / len(turns) >= 0.5:
                habitual.setdefault(tone, set()).add(actor)
    return {f"chat:tone:{tone}": ("tone", tone, FRIENDLY, len(people), 0, 1.0 - 1.0 / (
        len(people) + 1.0)) for tone, people in habitual.items() if len(people) >= 2}


async def _store(database, guild_id: int, now: datetime, wanted: dict) -> int:
    subject = f"discord:guild:{guild_id}"
    changed = 0
    existing = {r["signature"]: r["status"] for r in await database.fetchall(
        "SELECT signature, status FROM continuous_cognition_hypotheses "
        "WHERE guild_id=? AND signature LIKE 'chat:%'", (guild_id,))}
    for signature, (kind, key, sign, people, against, confidence) in sorted(wanted.items()):
        hypothesis_id = "chat-norm:" + hashlib.sha256(
            f"{guild_id}|{signature}".encode()).hexdigest()[:32]
        roles = json.dumps({"kind": kind, "key": key, "people": people, "against": against},
                           ensure_ascii=False, sort_keys=True)
        await database.execute(
            """INSERT INTO continuous_cognition_hypotheses
               (hypothesis_id, guild_id, signature, subject_key, predicate_key, object_key,
                roles_json, epistemic_status, confidence, positive_evidence, negative_evidence,
                status, first_seen_at, last_reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'hypothesis', ?, ?, ?, 'working', ?, ?)
               ON CONFLICT (guild_id, signature) DO UPDATE SET object_key=excluded.object_key,
                 roles_json=excluded.roles_json, confidence=excluded.confidence,
                 positive_evidence=excluded.positive_evidence,
                 negative_evidence=excluded.negative_evidence, status='working',
                 last_reviewed_at=excluded.last_reviewed_at""",
            (hypothesis_id, guild_id, signature, subject, f"chat_norm:{kind}", sign, roles,
             min(1.0, max(0.0, confidence)), people, against, now.isoformat(), now.isoformat()))
        changed += 1
    for signature, status in existing.items():
        if signature not in wanted and status != "rejected":
            # the votes faded or were outweighed: the norm is withdrawn, not kept as a habit
            await database.execute(
                "UPDATE continuous_cognition_hypotheses SET status='rejected', last_reviewed_at=? "
                "WHERE guild_id=? AND signature=?", (now.isoformat(), guild_id, signature))
            changed += 1
    return changed
