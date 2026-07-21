CREATE TABLE language_reference_lookups (
    normalized_form TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('found', 'missing', 'error')),
    source_url TEXT,
    part_of_speech TEXT,
    meanings_json TEXT NOT NULL DEFAULT '[]',
    checked_at TEXT NOT NULL
);

CREATE INDEX idx_reference_lookups_status
    ON language_reference_lookups (status, checked_at DESC);
