CREATE TABLE memory_pending_fragments (
    fragment_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    actor_id INTEGER,
    occurred_at TEXT NOT NULL,
    text TEXT NOT NULL,
    topic_terms_json TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL CHECK (importance BETWEEN 0.0 AND 1.0),
    eviction_reason TEXT NOT NULL CHECK (eviction_reason IN ('expired', 'capacity')),
    queued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_memory_pending_scope
    ON memory_pending_fragments (guild_id, channel_id, occurred_at);

CREATE TABLE memory_long_term_capsules (
    capsule_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    participant_ids_json TEXT NOT NULL DEFAULT '[]',
    topic_terms_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL,
    source_event_ids_json TEXT NOT NULL DEFAULT '[]',
    source_kind TEXT NOT NULL DEFAULT 'working_memory',
    importance REAL NOT NULL CHECK (importance BETWEEN 0.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    strength REAL NOT NULL DEFAULT 1.0 CHECK (strength BETWEEN 0.0 AND 1.0),
    decay_rate REAL NOT NULL DEFAULT 0.015 CHECK (decay_rate >= 0.0),
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_memory_capsules_recall
    ON memory_long_term_capsules (guild_id, strength DESC, importance DESC, updated_at DESC);

CREATE INDEX idx_memory_capsules_channel
    ON memory_long_term_capsules (guild_id, channel_id, updated_at DESC);
