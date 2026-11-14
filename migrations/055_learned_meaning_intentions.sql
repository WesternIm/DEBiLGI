-- Prepared meanings are not experience until a corresponding message was sent.
CREATE TABLE language_meaning_intentions (
    intention_id TEXT PRIMARY KEY,
    parent_intent_id TEXT NOT NULL,
    prompt_event_id TEXT NOT NULL REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    context_key TEXT NOT NULL,
    behavior_key TEXT NOT NULL,
    operation TEXT NOT NULL,
    plan_key TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    expected_reward REAL NOT NULL,
    expected_outcome TEXT,
    sample_count INTEGER NOT NULL,
    selection_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'prepared',
    response_message_id INTEGER,
    sent_at TEXT,
    action_json TEXT,
    response_text TEXT,
    feedback_event_id TEXT REFERENCES events(event_id),
    outcome_type TEXT,
    reward REAL,
    observed_at TEXT,
    CHECK (status IN ('prepared', 'sent', 'observed', 'expired')),
    CHECK (status = 'prepared' OR response_message_id IS NOT NULL),
    CHECK (reward IS NULL OR (reward >= -1.0 AND reward <= 1.0))
);
CREATE UNIQUE INDEX language_meaning_intentions_response
    ON language_meaning_intentions(guild_id, response_message_id);
CREATE INDEX language_meaning_intentions_learning
    ON language_meaning_intentions(guild_id, channel_id, user_id, context_key, observed_at);

CREATE TABLE language_meaning_consequences (
    intention_id TEXT NOT NULL REFERENCES language_meaning_intentions(intention_id),
    event_id TEXT NOT NULL REFERENCES events(event_id),
    outcome_type TEXT NOT NULL,
    semantic_effect_json TEXT,
    external_value REAL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (intention_id, event_id)
);
