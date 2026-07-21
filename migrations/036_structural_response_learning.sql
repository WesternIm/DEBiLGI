CREATE TABLE language_structural_feature_weights (
    guild_id INTEGER NOT NULL,
    feature_key TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0 CHECK (weight BETWEEN -0.75 AND 0.75),
    observation_count INTEGER NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
    positive_feedback INTEGER NOT NULL DEFAULT 0 CHECK (positive_feedback >= 0),
    negative_feedback INTEGER NOT NULL DEFAULT 0 CHECK (negative_feedback >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, feature_key)
);

CREATE INDEX idx_structural_feature_weights_recall
    ON language_structural_feature_weights (
        guild_id, observation_count DESC, ABS(weight) DESC
    );

CREATE TABLE language_candidate_structural_features (
    candidate_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    features_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_candidate_structural_features_guild
    ON language_candidate_structural_features (guild_id, created_at DESC);
