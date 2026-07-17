CREATE TABLE social_reactions (
    reaction_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    actor_id INTEGER,
    agreement REAL NOT NULL CHECK (agreement BETWEEN 0.0 AND 1.0),
    resistance REAL NOT NULL CHECK (resistance BETWEEN 0.0 AND 1.0),
    approval REAL NOT NULL CHECK (approval BETWEEN 0.0 AND 1.0),
    curiosity REAL NOT NULL CHECK (curiosity BETWEEN 0.0 AND 1.0),
    aversion REAL NOT NULL CHECK (aversion BETWEEN 0.0 AND 1.0),
    trust REAL NOT NULL CHECK (trust BETWEEN 0.0 AND 1.0),
    arousal REAL NOT NULL CHECK (arousal BETWEEN 0.0 AND 1.0),
    primary_component TEXT NOT NULL,
    secondary_component TEXT,
    blend_id TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    signals_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX idx_social_reactions_guild_created
    ON social_reactions (guild_id, created_at DESC);

CREATE INDEX idx_social_reactions_actor_created
    ON social_reactions (guild_id, actor_id, created_at DESC);

CREATE INDEX idx_social_reactions_blend
    ON social_reactions (guild_id, blend_id, created_at DESC);

CREATE TABLE social_relation_states (
    guild_id INTEGER NOT NULL,
    actor_id INTEGER NOT NULL,
    agreement REAL NOT NULL CHECK (agreement BETWEEN 0.0 AND 1.0),
    resistance REAL NOT NULL CHECK (resistance BETWEEN 0.0 AND 1.0),
    approval REAL NOT NULL CHECK (approval BETWEEN 0.0 AND 1.0),
    curiosity REAL NOT NULL CHECK (curiosity BETWEEN 0.0 AND 1.0),
    aversion REAL NOT NULL CHECK (aversion BETWEEN 0.0 AND 1.0),
    trust REAL NOT NULL CHECK (trust BETWEEN 0.0 AND 1.0),
    arousal REAL NOT NULL CHECK (arousal BETWEEN 0.0 AND 1.0),
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, actor_id)
);
