CREATE TABLE language_semantic_consultations (
    consultation_id TEXT PRIMARY KEY,
    event_id TEXT,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    input_hash TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('utterance', 'lexical')),
    subject_terms_json TEXT NOT NULL DEFAULT '[]',
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    local_confidence REAL NOT NULL CHECK (local_confidence BETWEEN 0.0 AND 1.0),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    result_json TEXT NOT NULL,
    result_confidence REAL NOT NULL CHECK (result_confidence BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'provisional'
        CHECK (status IN ('provisional', 'confirmed', 'rejected')),
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    UNIQUE (guild_id, input_hash)
);

CREATE INDEX idx_semantic_consultations_reuse
    ON language_semantic_consultations (
        guild_id, input_hash, status, expires_at DESC
    );

CREATE INDEX idx_semantic_consultations_audit
    ON language_semantic_consultations (
        status, result_confidence DESC, updated_at DESC
    );
