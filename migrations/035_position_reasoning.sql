CREATE TABLE language_positions (
    guild_id INTEGER NOT NULL,
    position_key TEXT NOT NULL,
    target TEXT NOT NULL,
    quality TEXT NOT NULL,
    stance TEXT NOT NULL,
    score REAL NOT NULL CHECK (score BETWEEN -1.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_json TEXT NOT NULL DEFAULT '[]',
    observation_count INTEGER NOT NULL DEFAULT 1 CHECK (observation_count > 0),
    positive_feedback INTEGER NOT NULL DEFAULT 0 CHECK (positive_feedback >= 0),
    negative_feedback INTEGER NOT NULL DEFAULT 0 CHECK (negative_feedback >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, position_key)
);

CREATE INDEX idx_language_positions_recall
    ON language_positions (guild_id, target, quality, confidence DESC);

CREATE TABLE language_dialogue_intent_preferences (
    guild_id INTEGER NOT NULL,
    act_kind TEXT NOT NULL,
    response_obligation TEXT NOT NULL,
    move TEXT NOT NULL,
    expression_mode TEXT NOT NULL,
    utility REAL NOT NULL DEFAULT 0.0 CHECK (utility BETWEEN -1.0 AND 1.0),
    observation_count INTEGER NOT NULL DEFAULT 1 CHECK (observation_count > 0),
    positive_feedback INTEGER NOT NULL DEFAULT 0 CHECK (positive_feedback >= 0),
    negative_feedback INTEGER NOT NULL DEFAULT 0 CHECK (negative_feedback >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (
        guild_id, act_kind, response_obligation, move, expression_mode
    )
);

CREATE TABLE language_expressed_positions (
    response_message_id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    candidate_id TEXT,
    position_key TEXT NOT NULL,
    stance TEXT NOT NULL,
    score REAL NOT NULL CHECK (score BETWEEN -1.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    act_kind TEXT NOT NULL,
    response_obligation TEXT NOT NULL,
    move TEXT NOT NULL,
    expression_mode TEXT NOT NULL,
    expressed_at TEXT NOT NULL,
    feedback_reward REAL NOT NULL DEFAULT 0.0 CHECK (feedback_reward BETWEEN -1.0 AND 1.0)
);

CREATE INDEX idx_expressed_positions_lookup
    ON language_expressed_positions (guild_id, position_key, expressed_at DESC);

CREATE TABLE language_position_feedback (
    feedback_event_id TEXT PRIMARY KEY,
    response_message_id INTEGER NOT NULL,
    guild_id INTEGER NOT NULL,
    position_key TEXT NOT NULL,
    feedback_kind TEXT NOT NULL,
    reward REAL NOT NULL CHECK (reward BETWEEN -1.0 AND 1.0),
    attribution_confidence REAL NOT NULL CHECK (attribution_confidence BETWEEN 0.0 AND 1.0),
    observed_at TEXT NOT NULL
);
