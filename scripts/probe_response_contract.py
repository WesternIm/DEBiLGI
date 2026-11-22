"""Offline response audit on SQLite backups; never logs in to Discord.

Uses the existing seed_harness.build and a local transport with receipts.
Run with PYTHONPATH=src (plus a local runner workaround if needed).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from debilgi import config
from debilgi.discord.reply_workflow import ReplyWorkflow
from debilgi.events.normalizer import normalize_message_created

PROMPTS = (
    "ку бро", "о чем думаешь", "как дела", "чем занимаешься?", "ты любишь собак?",
    "я сегодня работал весь день", "привет, расскажи о себе", "что ты сейчас делаешь?",
    "мне нравится рисовать", "какие книги ты читаешь?", "сегодня идёт дождь",
    "объясни, почему небо голубое",
)
DIALOGUES = (
    *((f"reactive_{index}", (text,)) for index, text in enumerate((
        "лошара", "ты тупой", "ну ты бля крутой", "красавчик", "я читаю книгу",
        "мне нравится рисовать", "я сегодня работал весь день", "как дела", "скучно", "ахахах",
    ))),
    ("reactive_repeat", ("лошара", "лошара")),
    ("reactive_smalltalk_episodes", ("как дела", {"text": "как дела", "seconds": 700},
                                     {"text": "как дела", "seconds": 1400})),
    ("reactive_smalltalk_repeat", ("как дела", "как дела")),
    ("reactive_shaving", ("я брил ноги",)),
    ("reactive_bad_news", ("я провалил экзамен",)),
    ("clarification", ("ты любишь собак?", "я спрашиваю, любишь ли ты собак?",
                       "так да или нет?", "что ты сейчас делаешь?")),
    ("memory", ("я читаю книгу.", "что я читаю?", "нет, я читаю журнал.", "что я читаю?")),
    ("social", ("ку бро", "спасибо", "ты кто?", "как дела")),
    ("people", ("я читаю книгу.", {"text": "я читаю журнал.", "person": 1},
                "что я читаю?", {"text": "что я читаю?", "person": 1})),
    ("dates", ({"text": "я сегодня читал книгу.", "day": 0},
               {"text": "я сегодня читал журнал.", "day": 1},
               {"text": "что я вчера читал?", "day": 1},
               {"text": "что я сегодня читал?", "day": 1},
               {"text": "что я вчера читал?", "day": 2})),
    ("unknown", ("я читаю книгу.", "что я покупаю?", "я покупаю журнал?")),
    ("ambiguity", ("Дракон встретил рыцаря.", "Он ушёл?", "Он ушёл?", "Дракон")),
    ("ambiguity_female", ("Маша встретила Олю.", "Она читала?", "Она читала?", "Оля")),
    ("shaving", ("я вчера брил ноги", "что я делал вчера?",
                 "нет, я брил руки", "что я брил?")),
    ("state_exception", ("я вчера болел", "что я делал вчера?")),
    ("stale_greeting", ("так ты согласен или нет?", "ку бро", "че делаешь")),
    ("stale_call", ("так ты согласен или нет?",
                    {"text": "але", "seconds": 700},
                    {"text": "ауу", "seconds": 740},
                    {"text": "ты где?", "seconds": 780})),
    ("stranger", ("что Петя делал вчера?", "какие книги Петя читал?")),
    ("self", ("ку бро", "что ты делаешь?", "что ты сегодня делал?",
              "какие книги ты читал?", "ты где?",
              {"text": "как тебя зовут?", "identity": "Дебильги"})),
    *((f"stage48_extra_{index}", (text,)) for index, text in enumerate((
        "я читаю журнал", "мне нравится рисовать", "я вчера на шашлыках был", "Дракон",
    ))),
    ("stage48_extra_names", ("Маша встретила Олю.", "Она читала?")),
    ("hypothesis_accept", ("я вчера болел", "да", "я вчера болел")),
    ("hypothesis_reject", ("я вчера болел", "не, ты чё", "я вчера болел")),
    ("live_unknown_why", ("че делаешь?", "почему?", "ну так почему?")),
    ("live_greeting_why", ("але", "и к чему это?")),
    ("live_greeting_misunderstood", ("але", "что здорово? я не понял тебя", "я не понял тебя")),
    ("live_historical_greeting", ({"text": "але", "historical_surface": "вот здравствуй."},
                                  "и к чему это?")),
    ("live_historical_ambiguous", ({"text": "але", "historical_surface": "даже здорово.."},
                                   "что здорово? я не понял тебя", "я не понял тебя")),
    ("live_self_polar", ("ты крутой математик?",)),
    ("stage48b_tense_past", ("я сегодня читал книгу",)),
    ("stage48b_world_accept", ("я вчера на шашлыках был", "да", "я вчера отдыхал?")),
    ("stage48b_world_reject", ("я вчера на шашлыках был", "не, ты чё", "я вчера отдыхал?")),
    ("stage49_attribute_after_action", ("ку бро", "что ты сегодня делал?", "так какую?")),
    ("stage49_attribute_after_question", ("я читаю книгу", "так какую?")),
    ("stage49_reason_after_unknown", ("че делаешь?", "а че так?")),
    ("stage49_causal_after_greeting", ("але", "потому что ты?")),
)
# The user's complete live 4.10 conversation, kept as one receipt-backed chain.
STAGE411_LIVE = (
    "ну ку бро", "я ферзевина", "а ты кто?", "а если прикинуть кто ты", "ну ок",
    "а что ты делаешь?", "кто бегает быстрее муравей или медведь?",
    "а ты ответь на вопрос", "ответишь?", "и какой ответ на мой вопрос?",
    "кто быстрее муравей или медведь?", "кто быстрее?", "быстрее из них медведь",
    "понял?", "и почему же я не понял тебя?", "я это понял", "о обоих",
    "кто быстрее? медведь или муравей?", "преположи", "медведь или муравей?",
)
DIALOGUES += (("stage411_live_20261116", STAGE411_LIVE),)

NOW = datetime(2026, 11, 14, 14, tzinfo=UTC)
FORM_TURNS = (
    "эй", "алоу бро", "а ты попробуй дать наугад", "аузын жап котакбас", "дырка?",
    "а что есть у тебя?", "ответишь?", "а че так?", "потому что ты?", "так какую?",
    "я вчера болел", "скучно", "ахахах", "лошара", "как дела", "я читаю книгу",
    "ну ты бля крутой", "красавчик",
)


def backup(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError(f"probe target must be a new copy: {target}")
    with (
        closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as original,
        closing(sqlite3.connect(target)) as copied,
    ):
        original.backup(copied)


class OfflineTransport:
    """A successful local send proves only simulated delivery, never Discord delivery."""

    def __init__(self):
        self.receipts = []

    async def send(self, event, reply):
        receipt = SimpleNamespace(
            id=2_999_000_000_000_000_000 + len(self.receipts),
            created_at=event.utc_timestamp + timedelta(seconds=1),
        )
        self.receipts.append((receipt, reply.text))
        return receipt


async def probe(harness, source: Path, work: Path, *, baseline: bool = False,
                scenario_prefix: str = "", stage48_only: bool = False, scenario_names=None,
                stage48b: bool = False, stage49: bool = False):
    # A relocated old-code snapshot must read exactly the same local settings.
    config._project_root = lambda: Path(__file__).resolve().parents[1]
    results = []
    # Keep the historical prepare-only sequence intact. Each multi-turn dialogue
    # gets a separate seed copy, preventing cross-scenario memory/feedback leakage.
    scenarios = (
        *((f"form_{i}", (text,)) for i, text in enumerate(FORM_TURNS)),
        ("form_self_followup", ("что ты сегодня делал?", "так какую?",
                                 "какую конструкцию ты изучил?", "потому что ты?")),
        ("form_self_grounded", ("ку бро", "что ты сегодня делал?", "так какую?",
                                 "какую конструкцию ты изучил?", "потому что ты?")),
        ("base", PROMPTS), *DIALOGUES,
    )
    scenarios = tuple(s for s in scenarios if s[0].startswith(scenario_prefix))
    if scenario_names is not None:
        scenarios = tuple(s for s in scenarios if s[0] in scenario_names)
    if stage48_only:
        scenarios = tuple(s for s in scenarios if s[0] in {
            "memory", "people", "dates", "shaving", "self",
            "hypothesis_accept", "hypothesis_reject",
        } or s[0].startswith("stage48_extra_") or (
            s[0].startswith("form_") and s[0][5:].isdigit()) or (
                stage48b and s[0].startswith(("live_", "stage48b_"))) or (
                    stage49 and s[0].startswith("stage49_")))
    for scenario, prompts in scenarios:
        target = work / f"{scenario}.sqlite3"
        backup(source, target)
        _, database, _, bot = harness.build(str(target))
        try:
            await database.initialize()
            await bot.pipeline.start()
            workflow = ReplyWorkflow(
                SimpleNamespace(user=SimpleNamespace(id=harness.BOT_ID)), bot.pipeline,
                bot.reply_service, bot.context_gate, bot.response_traces,
            )
            transport = OfflineTransport()
            previous = None
            hypothesis_rule = None
            for index, turn in enumerate(prompts):
                spec = turn if isinstance(turn, dict) else {"text": turn}
                prompt = spec["text"]
                actor = harness.USER + spec.get("person", 0)
                event = normalize_message_created(
                    message_id=1_999_000_000_000_000_000 + index,
                    guild_id=harness.GUILD, channel_id=harness.CHANNEL, author_id=actor,
                    created_at=NOW + timedelta(
                        days=spec.get("day", 0), seconds=spec.get("seconds", index * 40),
                    ),
                    content=f"<@{harness.BOT_ID}>{prompt}", author_is_bot=False,
                    bot_mentioned=True, bot_user_id=harness.BOT_ID,
                    bot_identity={"id": harness.BOT_ID, "username": spec["identity"]}
                    if "identity" in spec else None,
                    reply_to_message_id=previous.id if previous else None,
                    reply_to_author_id=harness.BOT_ID if previous else None,
                )
                sql_start = getattr(database, "probe_sql", 0)
                query_start = getattr(database, "probe_queries", 0)
                rows_start = getattr(database, "probe_rows", 0)
                await bot.pipeline.publish(event)
                preparation_started = perf_counter()
                prepared = await asyncio.wait_for(workflow.prepare(event), 600)
                preparation_seconds = perf_counter() - preparation_started
                hot_sql = getattr(database, "probe_sql", 0) - sql_start
                hot_queries = getattr(database, "probe_queries", 0) - query_start
                hot_rows = getattr(database, "probe_rows", 0) - rows_start
                if prepared and spec.get("historical_surface"):
                    # Offline fixture of the user's observed 4.8 send. It is not training data.
                    prepared = replace(prepared, text=spec["historical_surface"])
                    await bot.response_traces.step(
                        event.event_id, stage="prepared", component="historical_surface_fixture",
                        status="observed", summary="user-supplied historical bot surface",
                        details={"text": prepared.text, "synthetic_receipt": True},
                        created_at=event.utc_timestamp,
                    )
                validated = None
                receipt = None
                if scenario != "base" and prepared is not None:
                    validated = (prepared if spec.get("historical_surface")
                                 else await workflow.validate(event, prepared))
                    if validated is not None:
                        receipt = await transport.send(event, validated)
                        await workflow.record_sent(event, validated, receipt)
                        previous = receipt
                await bot.pipeline.join()
                turn_sql = getattr(database, "probe_sql", 0) - sql_start
                turn_queries = getattr(database, "probe_queries", 0) - query_start
                turn_rows = getattr(database, "probe_rows", 0) - rows_start
                trace = await database.fetchone(
                    "SELECT * FROM language_response_traces WHERE source_event_id = ?",
                    (event.event_id,),
                )
                steps = await bot.response_traces.steps_for_event(event.event_id)
                diagnostic = None if baseline else await bot.response_traces.diagnostics_for_event(
                    event.event_id,
                )
                claim_result = None
                if hasattr(bot.proposition_memory, "retrieve_claims"):
                    result = await bot.proposition_memory.retrieve_claims(event)
                    claim_result = {"status": result.status, "query": asdict(result.query),
                                    "claims": [asdict(c) for c in result.claims]}
                reaction_audit = None
                if prepared and prepared.response_source == "arbitrated:learned_dialogue_structure":
                    source_answers = await database.fetchall(
                        """SELECT d.content FROM language_discourse_turns d
                           JOIN events e ON e.event_id = d.source_id
                           WHERE d.guild_id = ? AND d.observed_at < ?
                             AND COALESCE(json_extract(e.payload_json, '$.author_is_bot'), 0) = 0
                             AND (e.source LIKE 'seed_corpus_%' OR EXISTS (
                               SELECT 1 FROM language_observed_exchanges x
                               WHERE x.answer_event_id = d.source_id))""",
                        (event.guild_id, event.utc_timestamp.isoformat()),
                    )
                    def normalize(text):
                        return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))
                    intention = await database.fetchone(
                        "SELECT plan_json FROM language_meaning_intentions WHERE intention_id = ?",
                        (prepared.meaning_intention_id,),
                    )
                    reaction_audit = {
                        "source_answer_events": len(source_answers),
                        "corpus_copy": any(normalize(prepared.text) == normalize(r["content"])
                                           for r in source_answers),
                        "selected_plan": json.loads(intention["plan_json"]) if intention else None,
                    }
                selected_meaning = None
                if prepared and prepared.meaning_intention_id:
                    stored = await database.fetchone(
                        "SELECT plan_json FROM language_meaning_intentions WHERE intention_id=?",
                        (prepared.meaning_intention_id,))
                    selected_meaning = json.loads(stored[0]) if stored else None
                row = {
                    "scenario": scenario, "turn": index + 1, "prompt": prompt,
                    "event_id": event.event_id,
                    "prepared_text": prepared.text if prepared else None,
                    "source": prepared.response_source if prepared else None,
                    "gate_text": validated.text if validated else None,
                    "sent_text": validated.text if receipt else None,
                    "receipt": receipt.id if receipt else None,
                    "status": trace["status"], "failure_stage": trace["failure_stage"],
                    "failure_reason": trace["failure_reason"], "diagnostic": diagnostic,
                    "steps": steps,
                    "claim_result": claim_result, "reaction_audit": reaction_audit,
                    "selected_meaning_plan": selected_meaning,
                    "historical_surface_fixture": bool(spec.get("historical_surface")),
                    "preparation_ms": preparation_seconds * 1000,
                    "sql_statements_before_answer": hot_sql,
                    "sql_statements_per_turn": turn_sql,
                    "sql_queries_before_answer": hot_queries,
                    "sql_rows_before_answer": hot_rows,
                    "sql_queries_per_turn": turn_queries, "sql_rows_per_turn": turn_rows,
                    "deliberation_steps": getattr(prepared, "deliberation_steps", 0),
                }
                from debilgi.discord import delivery

                row["pacing_remaining_seconds"] = (
                    delivery.reply_pacing_seconds(prepared.text, prepared.deliberation_steps,
                                                  preparation_seconds)
                    if prepared and hasattr(delivery, "reply_pacing_seconds") else None)
                decision = response_decision(row)
                actual_rule = ((diagnostic or {}).get("content") or {}).get("reaction_rule")
                if actual_rule and actual_rule.get("rule_id"):
                    decision = {**decision, **actual_rule}
                if decision.get("rule_id"):
                    learned_rules = await database.fetchall(
                        """SELECT x.answer_event_id, x.prompt_event_id, j.value
                           FROM language_observed_exchanges x, json_each(x.topic_links_json) j
                           WHERE x.guild_id = ? AND x.observed_at < ?
                             AND json_extract(j.value, '$.rule_id') = ?""",
                        (event.guild_id, event.utc_timestamp.isoformat(), decision["rule_id"]),
                    )
                    row["applied_rule"] = {
                        "rule_id": decision["rule_id"], "weight": decision["rule_weight"],
                        "activation": decision["activation"],
                        "source_ids": sorted({r[k] for r in learned_rules
                                              for k in ("answer_event_id", "prompt_event_id")}
                                             | {source for r in learned_rules
                                                for source in json.loads(r["value"]).get(
                                                    "source_ids", ())}),
                        "support": sum(json.loads(r["value"])["support"] for r in learned_rules),
                        "transformations": [json.loads(r["value"]) for r in learned_rules[:3]],
                    }
                if scenario.startswith("hypothesis_"):
                    await bot.pipeline.join()
                    decision = response_decision(row)
                    if actual_rule and actual_rule.get("rule_id"):
                        decision = {**decision, **actual_rule}
                    if decision.get("confidence_level") == "hypothesis":
                        hypothesis_rule = decision.get("rule_id")
                    rule_rows = await database.fetchall(
                        "SELECT topic_links_json FROM language_observed_exchanges WHERE guild_id=?",
                        (event.guild_id,),
                    )
                    observed = [link for r in rule_rows for link in json.loads(r[0])
                                if hypothesis_rule and link.get("rule_id") == hypothesis_rule]
                    row["hypothesis_rule_state"] = {
                        "rule_id": hypothesis_rule, "edges": len(observed),
                        "weights": sorted({r["weight"] for r in observed}),
                        "support": sum(r["support"] for r in observed),
                        "exceptions": sorted({e for r in observed for e in r["exceptions"]}),
                        "feedback_ids": sorted({e for r in observed for e in r["feedback_ids"]}),
                    }
                row["response_hypotheses"] = [dict(r) for r in await database.fetchall(
                    "SELECT hypothesis_id,signature,epistemic_status,confidence,positive_evidence,"
                    "negative_evidence,status,roles_json FROM continuous_cognition_hypotheses "
                    "WHERE signature LIKE 'response:%' ORDER BY hypothesis_id")]
                row["confirmed_hypothesis_claims"] = [dict(r) for r in await database.fetchall(
                    "SELECT c.claim_id,c.resolved,c.predicate,e.source_event_id,e.source_key "
                    "FROM meaning_claims c JOIN meaning_evidence e ON e.claim_id=c.claim_id "
                    "WHERE e.source_key LIKE 'confirmed_hypothesis:%' ORDER BY c.claim_id")]
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in {
                    "steps", "diagnostic", "claim_result", "reaction_audit",
                }}, ensure_ascii=False), flush=True)
            feedback = await database.fetchall(
                "SELECT outcome_type, external_value FROM language_meaning_consequences",
            )
            for row in results:
                if row["scenario"] == scenario:
                    row["scenario_feedback"] = [dict(item) for item in feedback]
                    if not baseline:
                        row["diagnostic"] = await bot.response_traces.diagnostics_for_event(
                            row["event_id"],
                        )
        finally:
            await bot.pipeline.stop(timeout=5)
            await database.close()
            await bot.close()
            # A full seed is several GB. Scenarios are isolated but sequential;
            # retain the JSON evidence, never a DB copy per scenario.
            for suffix in ("", "-wal", "-shm"):
                target.with_name(target.name + suffix).unlink(missing_ok=True)
    return results


def behavior(rows):
    """Compare observable choices; random intention/decision IDs and timings are excluded."""
    keys = (
        "scenario", "turn", "prompt", "prepared_text", "source", "gate_text", "sent_text",
        "receipt", "status", "failure_stage", "failure_reason", "scenario_feedback",
    )
    return [{key: row[key] for key in keys} for row in rows]


def verify_contract(rows):
    """Acceptance checks over every attempted candidate, including rejected ones."""
    count = 0
    for row in rows:
        diagnostic = row["diagnostic"]
        assert diagnostic["category"] in {
            "social", "informative", "epistemic", "clarification", "withheld", "error",
        }
        assert diagnostic["plan_id"] and diagnostic["meaning_id"]
        if diagnostic["category"] in {"withheld", "error"}:
            assert diagnostic["withhold_reason"]["kind"] in diagnostic["reasons"]
            assert diagnostic["withhold_reason"]["code"]
        for step in row["steps"]:
            details = step["details"]
            selection = details if step["stage"] == "selection" else details.get("selection")
            candidates = list(details.get("candidates", ()))
            if isinstance(selection, dict):
                candidates.extend(selection.get("scores", ()))
            for candidate in candidates:
                info = (
                    candidate.get("diagnostic") or candidate.get("metadata", {}).get("diagnostic")
                )
                assert info and info["content_source"] and info["producer"] and info["renderer"]
                assert info["plan_id"] and info["meaning_id"]
                count += 1
    print(f"contract verified: {len(rows)} turns, {count} candidate observations", flush=True)
    verify_mode_contract(rows)
    verify_form_contract(rows)
    if any(r["scenario"].startswith("reactive_") for r in rows):
        verify_reactive_contract(rows)
    if all(row.get("claim_result") is not None for row in rows):
        verify_claim_contract(rows)
        if any(ref.startswith("discord:agent:")
               for row in rows for role, ref in row["claim_result"]["query"]["arguments"]
               if role == "subject"):
            verify_self_contract(rows)


def verify_claim_contract(rows):
    """Facts, attribution and dated intervals are checked apart from reply modes."""
    indexed = {(row["scenario"], row["turn"]): row for row in rows}

    def single(scenario, turn, object_lemma):
        row = indexed[scenario, turn]
        result = row["claim_result"]
        assert result["status"] == "Known" and len(result["claims"]) == 1
        claim = result["claims"][0]
        assert any(d["role"] == "direct_object" and
                   d["argument"]["head"]["lemma"] == object_lemma
                   for d in claim["thought"]["dependencies"])
        assert claim["claim_id"] in claim["proof"]["claim_ids"]
        assert row["sent_text"] and "«" not in row["sent_text"]
        return claim

    book = single("memory", 2, "книга")
    journal = single("memory", 4, "журнал")
    assert indexed["memory", 2]["sent_text"] == "Ты читаешь книгу."
    assert tuple(journal["supersedes"]) == (book["claim_id"],)
    first = single("people", 3, "книга")
    second = single("people", 4, "журнал")
    assert first["subject_ref"] != second["subject_ref"]
    yesterday = single("dates", 3, "книга")
    today = single("dates", 4, "журнал")
    later = single("dates", 5, "журнал")
    assert yesterday["valid_to"] == today["valid_from"]
    assert later["claim_id"] == today["claim_id"]
    for turn in (2, 3):
        row = indexed["unknown", turn]
        assert row["claim_result"]["status"] == "Unknown"
        assert row["claim_result"]["claims"] == []
        assert row["sent_text"] and "не знаю" in row["sent_text"].casefold()
        assert response_decision(row)["mode"] == "silence_or_unknown"
    print("claim contract verified: corrections, people, source dates, Unknown", flush=True)


def response_decision(row):
    decisions = [s["details"]["decision"] for s in row["steps"]
                 if s["component"] == "response_mode"]
    assert len(decisions) == 1, (row["scenario"], row["turn"], decisions)
    return decisions[0]


def verify_form_contract(rows):
    indexed = {(row["scenario"], row["turn"]): row for row in rows}
    if not all((f"form_{i}", 1) in indexed for i in range(len(FORM_TURNS))):
        return
    for i in range(len(FORM_TURNS)):
        row = indexed[f"form_{i}", 1]
        decision = response_decision(row)
        if not row["sent_text"]:
            assert decision["reason"]
            continue
        assert decision["mode"] == "free_form"
        assert decision["confidence_level"] in {"answer", "hypothesis"}
        content = row["diagnostic"]["content"]
        assert content["reaction_parts"] and content["evidence_ids"]
        assert content["reverse_check"]["passed"]
        plan = row["reaction_audit"]["selected_plan"]
        assert plan["claim"]["predicate_class"] == "dialogue_act"
        assert not plan["claim"]["dependencies"]
    illness = indexed["form_10", 1]["diagnostic"]["content"]
    assert illness["reaction_meaning"] in {"empathize", "check_condition"}
    assert illness["event_valence"] is None or illness["event_valence"] < 0
    for scenario, turns in (("form_self_followup", (2, 3, 4)),
                            ("form_self_grounded", (3, 4, 5))):
        for turn in turns:
            row = indexed[scenario, turn]
            assert row["claim_result"]["status"] == "Unknown"
            assert not row["claim_result"]["claims"]
    delivered = sum(bool(indexed[f"form_{i}", 1]["sent_text"]) for i in range(len(FORM_TURNS)))
    print(f"form contract verified: {delivered}/18 delivered, scoped self Unknown, reverse checked",
          flush=True)


def verify_mode_contract(rows):
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    clear_unknown, informative, ambiguities = 0, 0, 0
    for row in rows:
        decision = response_decision(row)
        assert decision["mode"] in {"answer", "free_form", "silence_or_unknown", "clarification"}
        assert decision["reason"] and decision["parse_confidence"] is not None
        assert decision["expression_confidence"] is None  # Not measured before wording.
        mode_step = next(s["sequence"] for s in row["steps"]
                         if s["component"] == "response_mode")
        assert all(s["sequence"] > mode_step for s in row["steps"]
                   if s["component"] in {"corpus_generation", "evidence_planner"}
                   and s["stage"] == "candidate_generation")
        if decision["mode"] == "silence_or_unknown":
            assert row["diagnostic"]["category"] != "clarification"
            clear_unknown += 1
        if row["claim_result"]["status"] == "Known" and decision["mode"] == "answer":
            informative += not row["scenario"].startswith("form_")
            assert row["prepared_text"] and row["diagnostic"]["category"] == "informative"
        if decision["mode"] == "clarification":
            ambiguities += 1
            assert decision["ambiguous_slot"] == "subject"
            assert len(decision["alternatives"]) >= 2 and decision["expected_information_gain"] > 0
            assert row["sent_text"] and " или " in row["sent_text"]
    for scenario in ("ambiguity", "ambiguity_female"):
        assert response_decision(indexed[scenario, 2])["mode"] == "clarification"
        assert response_decision(indexed[scenario, 3])["mode"] != "clarification"
        assert indexed[scenario, 2]["receipt"] and not indexed[scenario, 3]["receipt"]
        assert any(f["outcome_type"] == "requested_information_received"
                   and f["external_value"] is None
                   for f in indexed[scenario, 2]["scenario_feedback"])
    for scenario, turn in (("stale_greeting", 2), ("stale_call", 2), ("stale_call", 3)):
        row = indexed[scenario, turn]
        assert response_decision(row)["mode"] == "free_form" and row["sent_text"]
        assert row["diagnostic"]["category"] == "social"
        assert row["diagnostic"]["content"]["plan_basis"] == "semantic_plan"
    assert "не знаю" in indexed["stale_greeting", 3]["sent_text"].casefold()
    shaving = indexed["shaving", 2]
    assert shaving["claim_result"]["status"] == "Known" and "ноги" in shaving["sent_text"]
    assert "руки" in indexed["shaving", 4]["sent_text"]
    assert indexed["state_exception", 2]["claim_result"]["status"] == "Unknown"
    assert "болел" not in (indexed["state_exception", 2]["sent_text"] or "")
    assert informative == 11
    print(f"mode contract verified: {clear_unknown} silence/Unknown decisions, "
          f"{informative} informative, "
          f"{ambiguities} slot clarifications; no consecutive clarification", flush=True)


def verify_reactive_contract(rows, *, require_coverage=False):
    """Keep data gaps visible; test fixtures must never inflate corpus coverage."""
    covered, missing = [], []
    for row in rows:
        if not row["scenario"].startswith("reactive_"):
            continue
        mode = response_decision(row)
        if row["sent_text"]:
            assert mode["mode"] == "free_form" and mode["reason"] == "activated_transformation"
            assert row["source"] == "arbitrated:learned_dialogue_structure"
            content = row["diagnostic"]["content"]
            assert content["evidence_ids"] and content["exchange_pattern_id"]
            assert content["reaction_meaning"] and content["reaction_parts"]
            assert all(p["part_id"].startswith("exchange-part:") for p in content["reaction_parts"])
            assert row["reaction_audit"] and not row["reaction_audit"]["corpus_copy"]
            plan = row["reaction_audit"]["selected_plan"]
            assert plan["claim"]["predicate_class"] == "dialogue_act"
            assert plan["claim"]["predicate"]["lemma"] == content["reaction_meaning"]
            assert not plan["claim"]["dependencies"]  # a speech act, never invented self state
            assert content["plan_basis"] == "semantic_plan"
            assert row["diagnostic"]["category"] == "social"
            assert len(row["sent_text"].split()) <= 7
            covered.append(row)
        else:
            assert row["failure_reason"] in {"no_reactive_evidence", "no_observed_social_form"}
            missing.append(row["prompt"])
    repeat = [r for r in covered if r["scenario"] == "reactive_repeat"]
    if len(repeat) == 2:
        assert repeat[0]["sent_text"] != repeat[1]["sent_text"]
        assert (repeat[0]["diagnostic"]["content"]["exchange_pattern_id"]
                != repeat[1]["diagnostic"]["content"]["exchange_pattern_id"])
    total = sum(r["scenario"].startswith("reactive_") for r in rows)
    print(f"reactive corpus coverage: {len(covered)}/{total}; missing={missing}", flush=True)
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    if ("reactive_smalltalk_repeat", 2) in indexed:
        first, second = (indexed["reactive_smalltalk_repeat", i] for i in (1, 2))
        assert first["sent_text"]
        if first["diagnostic"]["content"]["exchange_clause_kind"] == "question":
            assert second["diagnostic"].get("content", {}).get("exchange_clause_kind") != "question"
    if require_coverage:
        assert not missing, f"reactive corpus coverage incomplete: {missing}"


def verify_self_contract(rows):
    """Operational content and Unknown are checked independently of reply modes."""
    indexed = {row["turn"]: row for row in rows if row["scenario"] == "self"}
    for turn in (2, 4, 5):
        assert indexed[turn]["claim_result"]["status"] == "Unknown"
        assert indexed[turn]["claim_result"]["claims"] == []
    for turn in (3, 6):
        row = indexed[turn]
        assert row["claim_result"]["status"] == "Known" and row["sent_text"]
        assert row["source"] == "arbitrated:semantic_reasoning"
        for claim in row["claim_result"]["claims"]:
            assert claim["source_kind"] == "agent_observed"
            assert claim["subject_ref"].startswith("discord:agent:")
            assert claim["proof"]["source_event_ids"] and claim["thought"]["evidence"]
    assert indexed[6]["claim_result"]["claims"][0]["predicate"] == "называться"
    print("self contract verified: source IDs, current/past, books/location Unknown, identity",
          flush=True)


def verify_stage48_contract(rows):
    """Semantic computation may legitimately abstain; delivered acts must be proved."""
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    for i in range(len(FORM_TURNS)):
        row = indexed[f"form_{i}", 1]
        decision = response_decision(row)
        assert decision["confidence_level"] in {"answer", "hypothesis", "silence"}
        assert decision["input_features"] and decision["expression_confidence"] is None
        assert "не знаю" not in (row["sent_text"] or "").casefold()
    for row in rows:
        if row["source"] != "arbitrated:learned_dialogue_structure":
            continue
        content = row["diagnostic"]["content"]
        assert content["reverse_check"]["passed"]
        assert row["applied_rule"]["source_ids"] and row["applied_rule"]["weight"] > 0
        claim = row["reaction_audit"]["selected_plan"]["claim"]
        assert claim["predicate_class"] == "dialogue_act" and not claim["dependencies"]
    journal = indexed["stage48_extra_0", 1]["sent_text"]
    assert journal and "журнал" in journal and "какие журнал" not in journal
    assert "нравишься" not in (indexed["stage48_extra_1", 1]["sent_text"] or "")
    names = indexed["stage48_extra_names", 2]["sent_text"]
    assert names and "о Маше" in names and "об Оле" in names
    known = [r for r in rows if r["claim_result"]["status"] == "Known"
             and response_decision(r)["mode"] == "answer"]
    assert len(known) == 11
    for row in known:
        assert row["sent_text"] and all(c["proof"]["source_event_ids"]
                                        for c in row["claim_result"]["claims"])
    for scenario, direction in (("hypothesis_accept", 1), ("hypothesis_reject", -1)):
        first, second, third = (indexed[scenario, t] for t in (1, 2, 3))
        assert first["sent_text"] and response_decision(first)["confidence_level"] == "hypothesis"
        before, after = (r["hypothesis_rule_state"] for r in (first, second))
        assert before["rule_id"] == after["rule_id"] and after["feedback_ids"]
        assert (sum(after["weights"]) - sum(before["weights"])) * direction > 0
        assert bool(after["exceptions"]) == (direction < 0)
        assert response_decision(third)["confidence_level"] != "hypothesis"
    print(f"stage 4.8 verified: {len(rows)} turns, 11 facts, two attributed learning episodes",
          flush=True)


def verify_stage48b_contract(rows, *, flexible_world_hypotheses=False):
    """Check delivered meaning, receipt-scoped repairs and separate hypotheses."""
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    original = [indexed[f"form_{i}", 1] for i in range(18)]
    assert sum(bool(r["sent_text"]) for r in original) > 9
    for row in rows:
        decision = response_decision(row)
        assert decision["confidence_level"] in {"answer", "hypothesis", "silence"}
        if (row["source"] == "arbitrated:learned_dialogue_structure"
                and not row.get("historical_surface_fixture")):
            content = row["diagnostic"]["content"]
            assert content["reverse_check"]["passed"], row["scenario"]
            assert row["applied_rule"]["source_ids"] and row["applied_rule"]["weight"] > 0
            mixture = dict(content["response_mixture"])
            assert abs(sum(mixture.values()) - 1) < 1e-6
            assert content["hypothetical_assertions"] == 0
        if row["scenario"].startswith("live_"):
            assert row["sent_text"], (row["scenario"], row["turn"])
            if row["turn"] > 1:
                assert decision["act"] in {"repair_own_turn", "inform"}
            elif not row["historical_surface_fixture"]:
                assert "здорово" not in row["sent_text"].casefold()
    for scenario in ("form_15", "stage48_extra_0"):
        assert "читаешь" in indexed[scenario, 1]["sent_text"]
    past = indexed["stage48b_tense_past", 1]["sent_text"]
    assert "читал" in past and "сегодня" in past and "читаешь" not in past
    conditions = [indexed[s, 1]["sent_text"] for s in (
        "form_10", "form_11", "stage48_extra_2")]
    assert all(conditions) and len(set(conditions)) == 3
    assert "болел" in conditions[0] and "скучно" in conditions[1] and "отдыхал" in conditions[2]
    names = indexed["stage48_extra_names", 2]["sent_text"]
    assert "о Маше" in names and "об Оле" in names
    known = [r for r in rows if r["scenario"] in {"memory", "people", "dates", "shaving", "self"}
             and r["claim_result"]["status"] == "Known"]
    assert len(known) == 11 and all(r["sent_text"] for r in known)
    for scenario, direction in (("hypothesis_accept", 1), ("hypothesis_reject", -1)):
        first, second, third = (indexed[scenario, t] for t in (1, 2, 3))
        before, after = (r["hypothesis_rule_state"] for r in (first, second))
        assert first["sent_text"] and before["rule_id"] == after["rule_id"]
        assert (sum(after["weights"]) - sum(before["weights"])) * direction > 0
        assert bool(after["exceptions"]) == (direction < 0)
        assert response_decision(third)["confidence_level"] != "hypothesis"
    for scenario in ("stage48b_world_accept", "stage48b_world_reject"):
        first, second, third = (indexed[scenario, t] for t in (1, 2, 3))
        hypothesis = first["response_hypotheses"]
        assert hypothesis and all(h["positive_evidence"] == 0 for h in hypothesis)
        assert not first["confirmed_hypothesis_claims"]
        confirmation = json.loads(hypothesis[0]["roles_json"])["descriptor"].get("confirmation")
        if scenario.endswith("accept") and (
            not flexible_world_hypotheses or confirmation == "polar"
        ):
            assert second["confirmed_hypothesis_claims"]
            assert third["claim_result"]["status"] == "Known" and third["sent_text"]
        elif scenario.endswith("accept"):
            # Choosing a wh-question is permitted by 4.11. A subsequent bare
            # "yes" cannot establish its missing location/manner or its body.
            assert confirmation == "none"
            assert not second["confirmed_hypothesis_claims"]
            assert third["claim_result"]["status"] != "Known" and third["sent_text"]
        else:
            assert not second["confirmed_hypothesis_claims"]
            assert all(h["status"] == "rejected" for h in second["response_hypotheses"])
            assert third["claim_result"]["status"] != "Known"
    print(f"stage 4.8b verified: {len(rows)} turns, 11 facts, live repairs and human feedback",
          flush=True)


def verify_deliberation_coverage(rows, without_deliberation):
    def keys(sample):
        return {(r['scenario'], r['turn']) for r in sample}
    assert keys(rows) == keys(without_deliberation)
    def covered(sample):
        return sum(bool(r['sent_text']) for r in sample
                   if r['prompt'] != 'аузын жап котакбас')
    assert covered(rows) >= covered(without_deliberation), 'deliberation consumed delivered replies'


def verify_stage49_contract(rows, baseline=None, without_deliberation=None, *,
                            flexible_world_hypotheses=False):
    """Coverage is an acceptance condition: internal questions cannot replace replies."""
    verify_stage48b_contract(rows, flexible_world_hypotheses=flexible_world_hypotheses)
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    assert indexed['form_16', 1]['sent_text'], 'coarse praise must survive the reader check'
    for name in ('stage49_attribute_after_action', 'stage49_attribute_after_question',
                 'stage49_reason_after_unknown', 'stage49_causal_after_greeting'):
        chain = [r for r in rows if r['scenario'] == name]
        assert len(chain) >= 2 and all(r['sent_text'] for r in chain), name
        assert all(r['receipt'] for r in chain), name
    def covered(sample):
        return sum(bool(r['sent_text']) for r in sample
                   if r['prompt'] != 'аузын жап котакбас')
    if baseline is not None:
        old = {(r['scenario'], r['turn']): r for r in baseline}
        assert indexed.keys() == old.keys()
        facts = [r for r in baseline if r['scenario'] in {
            'memory', 'people', 'dates', 'shaving', 'self'
        } and r['claim_result']['status'] == 'Known']
        assert len(facts) == 11
        for before in facts:
            after = indexed[before['scenario'], before['turn']]
            # Dataclass tuples and their JSON array representation are the
            # same proof graph; compare the canonical serialized structure.
            assert json.dumps(after['claim_result'], sort_keys=True) == json.dumps(
                before['claim_result'], sort_keys=True)
            assert after['sent_text'] == before['sent_text']
        assert covered(rows) >= covered(baseline)
    if without_deliberation is not None:
        verify_deliberation_coverage(rows, without_deliberation)
    print('stage 4.9 verified: facts, delivered chains and no deliberation coverage loss',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-db", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harness", type=Path, default=Path(".codex_work/seed_harness.py"))
    parser.add_argument("--baseline", action="store_true", help="support code before stage 0")
    parser.add_argument("--audit-only", action="store_true", help="save failing baseline evidence")
    parser.add_argument("--scenario-prefix", default="", help="run selected independent scenarios")
    parser.add_argument("--stage48", action="store_true",
                        help="required 4.8 turns, all eleven facts and reaction learning")
    parser.add_argument("--stage48b", action="store_true",
                        help="4.8 turns plus live repair chains and past tense")
    parser.add_argument("--stage49", action="store_true",
                        help="4.8b plus receipt-backed contextual ellipsis chains")
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--without-deliberation", type=Path)
    parser.add_argument("--compare", type=Path, help="fail if choices differ from this JSON")
    parser.add_argument("--require-reactive-coverage", action="store_true",
                        help="also fail on missing learned reaction forms")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("seed_harness", args.harness)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    rows = asyncio.run(probe(harness, args.seed_db, args.work_dir, baseline=args.baseline,
                             scenario_prefix=args.scenario_prefix,
                             stage48_only=args.stage48 or args.stage48b or args.stage49,
                             stage48b=args.stage48b or args.stage49, stage49=args.stage49))
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.baseline and not args.audit_only:
        if args.stage49:
            verify_stage49_contract(
                rows, (json.loads(args.baseline_results.read_text())
                       if args.baseline_results else None),
                json.loads(args.without_deliberation.read_text())
                if args.without_deliberation else None,
            )
        elif args.stage48b:
            verify_stage48b_contract(rows)
        elif args.stage48:
            verify_stage48_contract(rows)
        else:
            verify_contract(rows)
        if args.require_reactive_coverage:
            verify_reactive_contract(rows, require_coverage=True)
    if args.compare:
        assert behavior(rows) == behavior(json.loads(args.compare.read_text(encoding="utf-8")))
        print(f"behavior identical: {len(rows)} turns", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()


def verify_stage411_contract(rows, baseline, without_deliberation=None):
    """The same factual proof may now have a neutral grammatical word order."""
    from debilgi.language.canonical_claims import ClaimQuery, answer_thought, argument_ref
    from debilgi.language.communication import surface_issues
    from debilgi.language.dialogue_obligations import thought_from_data
    from debilgi.language.semantic_graph import proposition_key

    verify_stage49_contract(rows, without_deliberation=without_deliberation,
                            flexible_world_hypotheses=True)
    indexed = {(r["scenario"], r["turn"]): r for r in rows}
    assert indexed.keys() == {(r["scenario"], r["turn"]) for r in baseline}
    facts = [r for r in baseline if r["scenario"] in {
        "memory", "people", "dates", "shaving", "self"
    } and r["claim_result"]["status"] == "Known"]
    assert len(facts) == 11
    for before in facts:
        after = indexed[before["scenario"], before["turn"]]
        assert after["claim_result"] == before["claim_result"]
        assert after["sent_text"]
        selected = thought_from_data(after["selected_meaning_plan"]["claim"])
        query = ClaimQuery(**after["claim_result"]["query"])
        allowed = []
        for raw in after["claim_result"]["claims"]:
            claim = SimpleNamespace(**{**raw, "thought": thought_from_data(raw["thought"]),
                                      "proof": SimpleNamespace(**raw["proof"])})
            allowed.append(answer_thought(claim, query))
        def meaning(thought):
            return (proposition_key(thought), thought.semantic_time, thought.scopes,
                    tuple((d.role, argument_ref(d.argument)) for d in thought.dependencies),
                    thought.evidence)
        assert meaning(selected) in [meaning(thought) for thought in allowed]
        assert not surface_issues(selected, after["sent_text"])
    assert not [(r["scenario"], r["turn"]) for r in baseline
                if r["sent_text"] and not indexed[r["scenario"], r["turn"]]["sent_text"]]
