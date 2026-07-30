CREATE TABLE language_processing_outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    response_message_id INTEGER,
    intention_id TEXT,
    reply_plan_id TEXT,
    outcome_type TEXT NOT NULL,
    generated_text TEXT,
    failure_stage TEXT,
    failure_reason TEXT,
    was_intentional INTEGER NOT NULL CHECK (was_intentional IN (0, 1)),
    diagnostic_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (source_event_id, reply_plan_id)
);

CREATE INDEX idx_processing_outcomes_guild
    ON language_processing_outcomes (guild_id, created_at DESC);

CREATE INDEX idx_processing_outcomes_failure
    ON language_processing_outcomes (outcome_type, failure_stage, created_at DESC);

CREATE TABLE language_content_proposals (
    proposal_id TEXT NOT NULL,
    decision_id TEXT NOT NULL REFERENCES language_arbitration_decisions(decision_id)
        ON DELETE CASCADE,
    source TEXT NOT NULL,
    speech_act TEXT NOT NULL,
    response_goal TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    source_confidence REAL NOT NULL CHECK (source_confidence BETWEEN 0.0 AND 1.0),
    applicability REAL,
    content_score REAL NOT NULL,
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
    selected_for_surface INTEGER NOT NULL CHECK (selected_for_surface IN (0, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (proposal_id, decision_id)
);

CREATE INDEX idx_content_proposals_decision
    ON language_content_proposals (decision_id, content_score DESC);

CREATE TABLE language_surface_proposals (
    candidate_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    FOREIGN KEY (proposal_id, decision_id)
        REFERENCES language_content_proposals(proposal_id, decision_id)
        ON DELETE CASCADE
);

CREATE TABLE language_arbitration_profile_versions (
    guild_id INTEGER NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('content', 'surface')),
    version INTEGER NOT NULL CHECK (version > 0),
    profile_id TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    source_biases_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL,
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, stage, version)
);

CREATE TABLE language_arbitration_profile_pointers (
    guild_id INTEGER NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('content', 'surface')),
    active_version INTEGER NOT NULL CHECK (active_version > 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, stage),
    FOREIGN KEY (guild_id, stage, active_version)
        REFERENCES language_arbitration_profile_versions(guild_id, stage, version)
);
