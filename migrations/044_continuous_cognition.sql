CREATE TABLE continuous_cognition_work_items (
    work_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL UNIQUE
        REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    priority REAL NOT NULL DEFAULT 0.5
        CHECK (priority BETWEEN 0.0 AND 1.0),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    result_summary TEXT,
    last_error TEXT,
    next_attempt_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX idx_continuous_work_due
    ON continuous_cognition_work_items (
        status, next_attempt_at, priority DESC, created_at
    );

CREATE TABLE continuous_cognition_hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    signature TEXT NOT NULL,
    subject_key TEXT,
    predicate_key TEXT NOT NULL,
    object_key TEXT,
    roles_json TEXT NOT NULL DEFAULT '[]',
    epistemic_status TEXT NOT NULL
        CHECK (epistemic_status IN ('inference', 'hypothesis')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    positive_evidence INTEGER NOT NULL DEFAULT 0 CHECK (positive_evidence >= 0),
    negative_evidence INTEGER NOT NULL DEFAULT 0 CHECK (negative_evidence >= 0),
    status TEXT NOT NULL DEFAULT 'working'
        CHECK (status IN ('working', 'retained', 'rejected')),
    first_seen_at TEXT NOT NULL,
    last_reviewed_at TEXT NOT NULL,
    UNIQUE (guild_id, signature)
);

CREATE INDEX idx_continuous_hypotheses_recall
    ON continuous_cognition_hypotheses (
        guild_id, status, confidence DESC, last_reviewed_at DESC
    );

CREATE TABLE continuous_cognition_evidence (
    evidence_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL
        REFERENCES continuous_cognition_hypotheses(hypothesis_id) ON DELETE CASCADE,
    source_event_id TEXT NOT NULL
        REFERENCES events(event_id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('supports', 'contradicts')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    details_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL,
    UNIQUE (hypothesis_id, source_event_id, relation)
);

CREATE INDEX idx_continuous_evidence_hypothesis
    ON continuous_cognition_evidence (hypothesis_id, observed_at DESC);

CREATE TABLE continuous_cognition_gaps (
    gap_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    signature TEXT NOT NULL,
    question_kind TEXT NOT NULL,
    subject_key TEXT,
    predicate_key TEXT,
    answer_role TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'abandoned')),
    source_event_id TEXT NOT NULL
        REFERENCES events(event_id) ON DELETE CASCADE,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (guild_id, signature)
);

CREATE INDEX idx_continuous_gaps_open
    ON continuous_cognition_gaps (
        guild_id, status, occurrence_count DESC, last_seen_at DESC
    );

CREATE TABLE continuous_cognition_signals (
    signal_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL UNIQUE
        REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    signal_kind TEXT NOT NULL,
    route TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_continuous_signals_recent
    ON continuous_cognition_signals (guild_id, observed_at DESC);

CREATE TABLE continuous_cognition_pulses (
    pulse_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed')),
    work_claimed INTEGER NOT NULL DEFAULT 0 CHECK (work_claimed >= 0),
    hypotheses_reviewed INTEGER NOT NULL DEFAULT 0 CHECK (hypotheses_reviewed >= 0),
    gaps_recorded INTEGER NOT NULL DEFAULT 0 CHECK (gaps_recorded >= 0),
    signals_recorded INTEGER NOT NULL DEFAULT 0 CHECK (signals_recorded >= 0),
    error_type TEXT,
    error_message TEXT
);

CREATE INDEX idx_continuous_pulses_recent
    ON continuous_cognition_pulses (started_at DESC);

CREATE TABLE continuous_cognition_runtime (
    runtime_id INTEGER PRIMARY KEY CHECK (runtime_id = 1),
    status TEXT NOT NULL CHECK (status IN ('stopped', 'running', 'degraded')),
    started_at TEXT,
    last_heartbeat_at TEXT,
    stopped_at TEXT,
    total_pulses INTEGER NOT NULL DEFAULT 0 CHECK (total_pulses >= 0),
    total_work_processed INTEGER NOT NULL DEFAULT 0
        CHECK (total_work_processed >= 0),
    last_error TEXT
);

INSERT INTO continuous_cognition_runtime (runtime_id, status)
VALUES (1, 'stopped');
