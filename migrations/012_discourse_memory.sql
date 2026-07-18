CREATE TABLE language_discourse_turns (
    source_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    parent_message_id INTEGER,
    actor_id INTEGER,
    addressed_to_id INTEGER,
    utterance_type TEXT NOT NULL
        CHECK (utterance_type IN ('assertion', 'question', 'command')),
    question_status TEXT NOT NULL
        CHECK (question_status IN ('not_question', 'open', 'answered', 'expired')),
    answered_by_source_id TEXT,
    content TEXT NOT NULL,
    topics_json TEXT NOT NULL DEFAULT '[]',
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (guild_id, channel_id, message_id)
);

CREATE INDEX idx_discourse_channel_time
    ON language_discourse_turns (guild_id, channel_id, observed_at DESC);

CREATE INDEX idx_discourse_actor_time
    ON language_discourse_turns (guild_id, actor_id, observed_at DESC);

CREATE INDEX idx_discourse_addressee_time
    ON language_discourse_turns (guild_id, addressed_to_id, observed_at DESC);

CREATE INDEX idx_discourse_parent
    ON language_discourse_turns (guild_id, channel_id, parent_message_id);

CREATE INDEX idx_discourse_open_questions
    ON language_discourse_turns (guild_id, channel_id, question_status, observed_at DESC);
