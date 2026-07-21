CREATE TABLE language_arbitration_weight_profiles (
    profile_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('surface')),
    weights_json TEXT NOT NULL,
    source_biases_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL,
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (guild_id, stage)
);

CREATE TABLE language_arbitration_decisions (
    decision_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    user_id INTEGER,
    response_message_id INTEGER UNIQUE,
    mode TEXT NOT NULL CHECK (mode IN ('shadow', 'active')),
    winner_candidate_id TEXT,
    minimum_score REAL NOT NULL,
    weight_profile_id TEXT,
    weight_version INTEGER NOT NULL DEFAULT 1,
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
    eligible_count INTEGER NOT NULL CHECK (eligible_count >= 0),
    used_fallback INTEGER NOT NULL DEFAULT 0 CHECK (used_fallback IN (0, 1)),
    selection_reason TEXT NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0.0 CHECK (latency_ms >= 0.0),
    legacy_text_hash TEXT,
    legacy_source TEXT,
    shadow_disagreed INTEGER NOT NULL DEFAULT 0 CHECK (shadow_disagreed IN (0, 1)),
    created_at TEXT NOT NULL,
    sent_at TEXT
);

CREATE INDEX idx_arbitration_decisions_guild
    ON language_arbitration_decisions (guild_id, created_at DESC);

CREATE INDEX idx_arbitration_decisions_response
    ON language_arbitration_decisions (guild_id, response_message_id);

CREATE TABLE language_response_candidates (
    candidate_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL REFERENCES language_arbitration_decisions(decision_id)
        ON DELETE CASCADE,
    source TEXT NOT NULL,
    renderer TEXT NOT NULL,
    speech_act TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    required_facts_json TEXT NOT NULL DEFAULT '[]',
    features_json TEXT NOT NULL,
    weighted_components_json TEXT NOT NULL,
    source_confidence REAL NOT NULL CHECK (source_confidence BETWEEN 0.0 AND 1.0),
    total_score REAL NOT NULL,
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
    selected INTEGER NOT NULL DEFAULT 0 CHECK (selected IN (0, 1)),
    generation_order INTEGER NOT NULL CHECK (generation_order >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX idx_response_candidates_decision
    ON language_response_candidates (decision_id, total_score DESC);

CREATE INDEX idx_response_candidates_source
    ON language_response_candidates (source, selected, created_at DESC);

CREATE TABLE language_arbitration_feedback (
    feedback_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    decision_id TEXT NOT NULL REFERENCES language_arbitration_decisions(decision_id)
        ON DELETE CASCADE,
    feedback_kind TEXT NOT NULL,
    reward REAL NOT NULL CHECK (reward BETWEEN -1.0 AND 1.0),
    attribution_confidence REAL NOT NULL CHECK (attribution_confidence BETWEEN 0.0 AND 1.0),
    old_version INTEGER,
    new_version INTEGER,
    weight_delta_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    applied_at TEXT
);

CREATE INDEX idx_arbitration_feedback_decision
    ON language_arbitration_feedback (decision_id, observed_at DESC);
