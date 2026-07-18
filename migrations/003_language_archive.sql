CREATE TABLE archive_import_jobs (
    job_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    requested_by INTEGER NOT NULL,
    after_at TEXT NOT NULL,
    before_at TEXT NOT NULL,
    max_messages INTEGER NOT NULL CHECK (max_messages > 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'paused', 'completed', 'failed', 'cancelled')),
    last_message_id INTEGER,
    messages_seen INTEGER NOT NULL DEFAULT 0 CHECK (messages_seen >= 0),
    messages_processed INTEGER NOT NULL DEFAULT 0 CHECK (messages_processed >= 0),
    delete_raw_after INTEGER NOT NULL DEFAULT 0 CHECK (delete_raw_after IN (0, 1)),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE INDEX idx_archive_jobs_guild_created
    ON archive_import_jobs (guild_id, created_at DESC);

CREATE INDEX idx_archive_jobs_status
    ON archive_import_jobs (status, updated_at);

CREATE TABLE archive_messages (
    message_id INTEGER PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    content TEXT,
    reply_to_message_id INTEGER,
    author_is_bot INTEGER NOT NULL CHECK (author_is_bot IN (0, 1)),
    raw_deleted INTEGER NOT NULL DEFAULT 0 CHECK (raw_deleted IN (0, 1)),
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_archive_messages_reply
    ON archive_messages (guild_id, reply_to_message_id);

CREATE TABLE archive_job_messages (
    job_id TEXT NOT NULL REFERENCES archive_import_jobs(job_id) ON DELETE CASCADE,
    message_id INTEGER NOT NULL REFERENCES archive_messages(message_id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, message_id)
);

CREATE INDEX idx_archive_job_messages_message
    ON archive_job_messages (message_id);

CREATE TABLE language_token_counts (
    guild_id INTEGER NOT NULL,
    token TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, token)
);

CREATE TABLE language_ngram_counts (
    guild_id INTEGER NOT NULL,
    n INTEGER NOT NULL CHECK (n IN (2, 3)),
    gram_json TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, n, gram_json)
);

CREATE TABLE language_reply_patterns (
    guild_id INTEGER NOT NULL,
    prompt_tail_json TEXT NOT NULL,
    response_head_json TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, prompt_tail_json, response_head_json)
);
