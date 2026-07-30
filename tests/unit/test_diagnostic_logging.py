import logging
from pathlib import Path

from debilgi.logging import configure_logging


def test_separate_diagnostic_log_files_are_created(tmp_path: Path) -> None:
    configure_logging("INFO", tmp_path)
    names = {
        "debilgi.reply_failures": "reply-fallback.log",
        "debilgi.topic_continuity": "topic-continuity.log",
        "debilgi.identity_replies": "identity-replies.log",
        "debilgi.open_questions": "open-questions.log",
        "debilgi.short_speech_acts": "short-speech-acts.log",
        "debilgi.background": "background-learning.log",
        "debilgi.continuous_cognition": "continuous-cognition.log",
        "debilgi.relation_filter": "relation-filter.log",
        "debilgi.knowledge_selection": "knowledge-selection.log",
        "debilgi.morphology_disambiguation": "morphology-disambiguation.log",
        "debilgi.spelling_correction": "spelling-correction.log",
        "debilgi.structured_utterance": "structured-utterance.log",
        "debilgi.sense_resolution": "sense-resolution.log",
        "debilgi.meaning_plan": "meaning-plan.log",
        "debilgi.dialogue_state": "dialogue-state.log",
        "debilgi.response_trace": "response-trace.log",
    }
    try:
        for logger_name in names:
            logger = logging.getLogger(logger_name)
            logger.info("diagnostic-test")
            for handler in logger.handlers:
                handler.flush()
        for filename in names.values():
            assert "diagnostic-test" in (tmp_path / filename).read_text(encoding="utf-8")
    finally:
        for logger_name in names:
            logger = logging.getLogger(logger_name)
            for handler in tuple(logger.handlers):
                if Path(getattr(handler, "baseFilename", "")).parent == tmp_path:
                    logger.removeHandler(handler)
                    handler.close()
            logger.propagate = True
