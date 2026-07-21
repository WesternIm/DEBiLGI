CREATE TABLE language_open_questions (
    question_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    concept_key TEXT NOT NULL,
    prompt_event_id TEXT NOT NULL,
    question_message_id INTEGER NOT NULL,
    asked_at TEXT NOT NULL,
    turns_remaining INTEGER NOT NULL DEFAULT 2 CHECK (turns_remaining >= 0),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'learned', 'stale', 'topic_changed')),
    response_event_id TEXT,
    response_text TEXT,
    relevance_reason TEXT,
    resolved_at TEXT
);

CREATE UNIQUE INDEX idx_open_language_question_per_user
    ON language_open_questions (guild_id, channel_id, user_id)
    WHERE status = 'active';

CREATE INDEX idx_open_language_question_status
    ON language_open_questions (guild_id, status, asked_at DESC);

CREATE UNIQUE INDEX idx_open_language_question_response
    ON language_open_questions (response_event_id)
    WHERE response_event_id IS NOT NULL;
