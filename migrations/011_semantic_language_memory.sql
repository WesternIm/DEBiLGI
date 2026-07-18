CREATE TABLE language_semantic_frames (
    source_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('event', 'archive', 'foundation')),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    actor_id INTEGER,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    modifiers_json TEXT NOT NULL DEFAULT '[]',
    negated INTEGER NOT NULL DEFAULT 0 CHECK (negated IN (0, 1)),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, frame_index)
);

CREATE INDEX idx_language_semantic_scope
    ON language_semantic_frames (guild_id, observed_at DESC);

CREATE INDEX idx_language_semantic_actor
    ON language_semantic_frames (guild_id, actor);

CREATE INDEX idx_language_semantic_action
    ON language_semantic_frames (guild_id, action);

CREATE INDEX idx_language_semantic_target
    ON language_semantic_frames (guild_id, target);
