"""Warm-start sparse response-structure weights from attributed feedback."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime

from debilgi.config import Settings
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.arbitration_repository import ArbitrationRepository
from debilgi.language.entities import render_message_content
from debilgi.language.structural_learning import structural_response_features
from debilgi.storage.database import Database


async def train(*, force: bool = False) -> tuple[int, int]:
    settings = Settings.from_env()
    database = Database(settings.database_path, settings.migrations_path)
    await database.initialize()
    try:
        existing = await database.fetchone(
            "SELECT COUNT(*) AS count FROM language_structural_feature_weights"
        )
        if existing is not None and int(existing["count"]) > 0 and not force:
            return 0, int(existing["count"])
        if force:
            await database.execute("DELETE FROM language_structural_feature_weights")
        repository = ArbitrationRepository(
            database,
            learning_enabled=True,
            learning_rate=settings.arbitration_learning_rate,
            minimum_feedback_observations=1,
            maximum_weight_delta=settings.arbitration_maximum_weight_delta,
        )
        feedback_rows = await database.fetchall(
            """SELECT feedback.*, decision.event_id, decision.guild_id
               FROM language_arbitration_feedback AS feedback
               JOIN language_arbitration_decisions AS decision
                 ON decision.decision_id = feedback.decision_id
               WHERE feedback.attribution_confidence >= 0.60
                 AND feedback.reward != 0.0
               ORDER BY feedback.observed_at"""
        )
        applied = 0
        for feedback in feedback_rows:
            event = await database.fetchone(
                "SELECT payload_json FROM events WHERE event_id = ?",
                (feedback["event_id"],),
            )
            if event is None:
                continue
            payload = json.loads(str(event["payload_json"]))
            raw_content = payload.get("content")
            if not isinstance(raw_content, str):
                continue
            rendered = render_message_content(raw_content, payload)
            prompt = DEFAULT_ADDRESSING.analyze(rendered).content_text.strip()
            candidates = await database.fetchall(
                """SELECT candidate_id, source, speech_act, text, created_at
                   FROM language_response_candidates
                   WHERE decision_id = ?""",
                (feedback["decision_id"],),
            )
            for candidate in candidates:
                features = structural_response_features(
                    prompt=prompt,
                    candidate_text=str(candidate["text"]),
                    source=str(candidate["source"]),
                    speech_act=str(candidate["speech_act"]),
                )
                await database.execute(
                    """INSERT OR REPLACE INTO language_candidate_structural_features (
                           candidate_id, guild_id, features_json, created_at
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        candidate["candidate_id"],
                        feedback["guild_id"],
                        json.dumps(features, ensure_ascii=False, sort_keys=True),
                        candidate["created_at"],
                    ),
                )
            await repository._apply_structural_feedback(  # noqa: SLF001
                guild_id=int(feedback["guild_id"]),
                decision_id=str(feedback["decision_id"]),
                reward=float(feedback["reward"]),
                attribution=float(feedback["attribution_confidence"]),
                observed_at=datetime.fromisoformat(str(feedback["observed_at"])),
            )
            applied += 1
        weights = await database.fetchone(
            "SELECT COUNT(*) AS count FROM language_structural_feature_weights"
        )
        return applied, int(weights["count"]) if weights is not None else 0
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    applied, weights = asyncio.run(train(force=arguments.force))
    print(f"feedback_applied={applied} structural_weights={weights}")


if __name__ == "__main__":
    main()
