-- An episode is a sent semantic hypothesis, never an archived answer template.
CREATE TABLE assertion_experiences (
    prompt_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    anchor_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    response_message_id INTEGER NOT NULL,
    learning_key TEXT NOT NULL,
    negated INTEGER NOT NULL CHECK (negated IN (0, 1)),
    sent_at TEXT NOT NULL,
    UNIQUE(guild_id, response_message_id)
);
CREATE INDEX assertion_experience_scope
    ON assertion_experiences(guild_id, channel_id, user_id, learning_key, sent_at);

CREATE TABLE assertion_feedback (
    prompt_event_id TEXT PRIMARY KEY REFERENCES assertion_experiences(prompt_event_id)
        ON DELETE CASCADE,
    feedback_event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    observed_at TEXT NOT NULL
);
