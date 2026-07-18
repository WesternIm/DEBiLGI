CREATE TABLE cognition_patterns (
    guild_id INTEGER NOT NULL,
    situation_json TEXT NOT NULL,
    outcome_type TEXT NOT NULL
        CHECK (outcome_type IN (
            'no_response', 'other_message', 'same_actor_message', 'reply', 'reaction'
        )),
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    weight REAL NOT NULL DEFAULT 0.2 CHECK (weight >= 0.0 AND weight <= 1.0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, situation_json, outcome_type)
);

CREATE INDEX idx_cognition_patterns_guild_updated
    ON cognition_patterns (guild_id, updated_at DESC);

CREATE TABLE cognition_expectations (
    expectation_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    actor_id INTEGER,
    trigger_event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
    trigger_message_id INTEGER,
    situation_json TEXT NOT NULL,
    distribution_json TEXT NOT NULL,
    predicted_outcome TEXT NOT NULL,
    predicted_probability REAL NOT NULL
        CHECK (predicted_probability >= 0.0 AND predicted_probability <= 1.0),
    evidence_support REAL NOT NULL DEFAULT 0.0 CHECK (evidence_support >= 0.0),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved')),
    actual_outcome TEXT,
    difference REAL CHECK (difference IS NULL OR (difference >= 0.0 AND difference <= 1.0)),
    resolved_by_event_id TEXT,
    resolved_at TEXT
);

CREATE INDEX idx_cognition_expectations_pending_expiry
    ON cognition_expectations (status, expires_at);

CREATE INDEX idx_cognition_expectations_pending_channel
    ON cognition_expectations (guild_id, channel_id, status, created_at DESC);

CREATE INDEX idx_cognition_expectations_trigger_message
    ON cognition_expectations (guild_id, channel_id, trigger_message_id, status);

CREATE TABLE cognition_observed_events (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

