CREATE TABLE learning_quality_cases (
    case_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    category TEXT NOT NULL CHECK (category IN (
        'text_understanding', 'response_quality', 'pipeline_failure'
    )),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('feedback', 'pipeline')),
    source_id TEXT NOT NULL UNIQUE,
    prompt_event_id TEXT,
    response_message_id INTEGER,
    fingerprint TEXT NOT NULL,
    reason TEXT NOT NULL,
    severity REAL NOT NULL CHECK (severity BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN (
        'queued', 'monitoring', 'resolved', 'exhausted'
    )),
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX idx_learning_quality_cases_open
    ON learning_quality_cases (
        guild_id, status, severity DESC, updated_at DESC
    );

CREATE INDEX idx_learning_quality_cases_category
    ON learning_quality_cases (
        guild_id, category, fingerprint, created_at DESC
    );

CREATE TABLE learning_quality_actions (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL
        REFERENCES learning_quality_cases(case_id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    target_key TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('queued', 'applied', 'failed')),
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (case_id, action_type, target_key)
);

CREATE INDEX idx_learning_quality_actions_case
    ON learning_quality_actions (case_id, created_at);
