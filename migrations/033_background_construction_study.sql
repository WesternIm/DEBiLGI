CREATE TABLE language_construction_study_queue (
    guild_id INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    channel_id INTEGER,
    prompt_hash TEXT NOT NULL,
    study_attempts INTEGER NOT NULL DEFAULT 0 CHECK (study_attempts >= 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'learned', 'exhausted', 'rejected')),
    queued_at TEXT NOT NULL,
    last_attempt_at TEXT,
    PRIMARY KEY (guild_id, source_event_id)
);

CREATE INDEX idx_construction_study_pending
    ON language_construction_study_queue (
        status, study_attempts, queued_at
    );

CREATE INDEX idx_construction_study_prompt
    ON language_construction_study_queue (
        guild_id, prompt_hash, status
    );
