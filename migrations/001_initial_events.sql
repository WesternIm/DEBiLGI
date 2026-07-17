CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    actor_id INTEGER,
    target_id INTEGER,
    event_type TEXT NOT NULL CHECK (length(event_type) > 0),
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source TEXT NOT NULL CHECK (length(source) > 0),
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_guild_occurred_at
    ON events (guild_id, occurred_at DESC);

CREATE INDEX idx_events_guild_type
    ON events (guild_id, event_type);
