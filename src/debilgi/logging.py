"""Logging configuration that never includes secrets."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DIAGNOSTIC_LOGGERS = {
    "debilgi.reply_failures": "reply-fallback.log",
    "debilgi.topic_continuity": "topic-continuity.log",
    "debilgi.identity_replies": "identity-replies.log",
    "debilgi.open_questions": "open-questions.log",
    "debilgi.short_speech_acts": "short-speech-acts.log",
    "debilgi.background": "background-learning.log",
    "debilgi.continuous_cognition": "continuous-cognition.log",
    "debilgi.context_gate": "context-gate.log",
    "debilgi.response_arbitration": "response-arbitration.log",
    "debilgi.relation_filter": "relation-filter.log",
    "debilgi.knowledge_selection": "knowledge-selection.log",
    "debilgi.morphology_disambiguation": "morphology-disambiguation.log",
    "debilgi.spelling_correction": "spelling-correction.log",
    "debilgi.structured_utterance": "structured-utterance.log",
    "debilgi.sense_resolution": "sense-resolution.log",
    "debilgi.meaning_plan": "meaning-plan.log",
    "debilgi.dialogue_state": "dialogue-state.log",
    "debilgi.contextual_question_reasoning": "contextual-question-reasoning.log",
    "debilgi.semantic_reply_plan": "semantic-reply-plan.log",
    "debilgi.response_trace": "response-trace.log",
    "debilgi.learning_quality": "learning-quality.log",
}


def configure_logging(level: str = "INFO", diagnostics_directory: Path | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    if diagnostics_directory is None:
        return
    diagnostics_directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for logger_name, filename in _DIAGNOSTIC_LOGGERS.items():
        diagnostic_logger = logging.getLogger(logger_name)
        expected = (diagnostics_directory / filename).resolve()
        if any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename).resolve() == expected
            for handler in diagnostic_logger.handlers
        ):
            continue
        handler = RotatingFileHandler(
            expected,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        diagnostic_logger.addHandler(handler)
        diagnostic_logger.setLevel(logging.INFO)
        diagnostic_logger.propagate = False
