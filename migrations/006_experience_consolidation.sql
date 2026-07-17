CREATE TABLE language_utterance_experiences (
    experience_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER,
    prompt_event_id TEXT NOT NULL UNIQUE,
    intent_id TEXT NOT NULL,
    response_message_id INTEGER NOT NULL UNIQUE,
    response_tokens_json TEXT NOT NULL,
    genome_signature TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK (candidate_count > 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'expired')),
    outcome_type TEXT,
    reward REAL CHECK (reward BETWEEN -1.0 AND 1.0),
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX idx_language_experiences_pending
    ON language_utterance_experiences (guild_id, status, channel_id, created_at DESC);

CREATE INDEX idx_language_experiences_user
    ON language_utterance_experiences (guild_id, user_id, created_at DESC);

CREATE TABLE language_experience_transitions (
    guild_id INTEGER NOT NULL,
    context_json TEXT NOT NULL,
    next_token TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0 CHECK (weight BETWEEN -4.0 AND 4.0),
    uses INTEGER NOT NULL DEFAULT 0 CHECK (uses >= 0),
    positive_outcomes INTEGER NOT NULL DEFAULT 0 CHECK (positive_outcomes >= 0),
    negative_outcomes INTEGER NOT NULL DEFAULT 0 CHECK (negative_outcomes >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, context_json, next_token)
);

CREATE INDEX idx_language_experience_transition_weight
    ON language_experience_transitions (guild_id, weight DESC, updated_at DESC);
