ALTER TABLE language_unknown_concepts
    ADD COLUMN study_attempts INTEGER NOT NULL DEFAULT 0 CHECK (study_attempts >= 0);

ALTER TABLE language_unknown_concepts
    ADD COLUMN last_study_attempt_at TEXT;

ALTER TABLE language_unknown_concepts
    ADD COLUMN study_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (study_status IN ('pending', 'learned', 'exhausted'));

UPDATE language_unknown_concepts SET study_status = 'learned' WHERE resolved = 1;

CREATE INDEX idx_unknown_background_study
    ON language_unknown_concepts (
        resolved, study_status, study_attempts, curiosity_score DESC, last_seen_at DESC
    );

ALTER TABLE learning_graph_edges ADD COLUMN last_decay_at TEXT;
UPDATE learning_graph_edges SET last_decay_at = last_seen_at WHERE last_decay_at IS NULL;

CREATE TABLE background_cycle_runs (
    cycle_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    words_processed INTEGER NOT NULL DEFAULT 0,
    words_found INTEGER NOT NULL DEFAULT 0,
    words_missing INTEGER NOT NULL DEFAULT 0,
    decayed_edges INTEGER NOT NULL DEFAULT 0,
    memory_items_maintained INTEGER NOT NULL DEFAULT 0,
    conflicts_found INTEGER NOT NULL DEFAULT 0,
    identity_checks INTEGER NOT NULL DEFAULT 0,
    initiative_queued INTEGER NOT NULL DEFAULT 0,
    initiative_sent INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed')),
    error_type TEXT,
    error_message TEXT
);

CREATE INDEX idx_background_cycles_recent
    ON background_cycle_runs (started_at DESC);

CREATE TABLE background_graph_conflicts (
    conflict_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    source_node_id TEXT NOT NULL,
    relation_a TEXT NOT NULL,
    target_a_node_id TEXT NOT NULL,
    relation_b TEXT NOT NULL,
    target_b_node_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    first_detected_at TEXT NOT NULL,
    last_detected_at TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    status TEXT NOT NULL DEFAULT 'review'
        CHECK (status IN ('review', 'resolved', 'ignored')),
    UNIQUE (
        guild_id, source_node_id, relation_a, target_a_node_id,
        relation_b, target_b_node_id, reason
    )
);

CREATE INDEX idx_background_conflicts_review
    ON background_graph_conflicts (guild_id, status, last_detected_at DESC);

CREATE TABLE background_identity_checks (
    check_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES background_cycle_runs(cycle_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    response_snapshot TEXT NOT NULL,
    evidence_source TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_background_identity_recent
    ON background_identity_checks (guild_id, observed_at DESC);

CREATE TABLE background_initiative_queue (
    initiative_id TEXT PRIMARY KEY,
    cycle_id TEXT NOT NULL REFERENCES background_cycle_runs(cycle_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    concept_key TEXT NOT NULL,
    message_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sent', 'failed', 'suppressed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    queued_at TEXT NOT NULL,
    attempted_at TEXT,
    sent_at TEXT,
    response_message_id INTEGER,
    failure_reason TEXT
);

CREATE INDEX idx_background_initiative_pending
    ON background_initiative_queue (status, queued_at);

CREATE INDEX idx_background_initiative_cooldown
    ON background_initiative_queue (guild_id, channel_id, status, sent_at DESC);
