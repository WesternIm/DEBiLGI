CREATE TABLE language_semantic_study_queue (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL DEFAULT 0,
    concept_key TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 0.5 CHECK (priority BETWEEN 0.0 AND 1.0),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count >= 1),
    study_attempts INTEGER NOT NULL DEFAULT 0 CHECK (study_attempts >= 0),
    last_study_attempt_at TEXT,
    study_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (study_status IN ('pending', 'learned', 'exhausted')),
    source_event_ids_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (guild_id, channel_id, concept_key)
);

CREATE INDEX idx_semantic_study_pending
    ON language_semantic_study_queue (
        study_status, priority DESC, occurrence_count DESC, last_seen_at DESC
    );
