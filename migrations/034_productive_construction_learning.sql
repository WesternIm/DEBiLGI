CREATE TABLE language_productive_constructions (
    guild_id INTEGER NOT NULL,
    construction_id TEXT NOT NULL,
    clause_kind TEXT NOT NULL,
    predicate_class TEXT NOT NULL DEFAULT 'any',
    roles_json TEXT NOT NULL,
    required_roles_json TEXT NOT NULL DEFAULT '[]',
    optional_roles_json TEXT NOT NULL DEFAULT '[]',
    focus_role TEXT,
    negation_mode TEXT NOT NULL DEFAULT 'any',
    observation_count INTEGER NOT NULL DEFAULT 1 CHECK (observation_count > 0),
    positive_feedback INTEGER NOT NULL DEFAULT 0 CHECK (positive_feedback >= 0),
    negative_feedback INTEGER NOT NULL DEFAULT 0 CHECK (negative_feedback >= 0),
    utility REAL NOT NULL DEFAULT 0.0 CHECK (utility BETWEEN -1.0 AND 1.0),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, construction_id)
);

CREATE INDEX idx_productive_constructions_selection
    ON language_productive_constructions (
        guild_id, clause_kind, predicate_class, utility DESC,
        confidence DESC, observation_count DESC
    );

CREATE TABLE language_candidate_constructions (
    candidate_id TEXT PRIMARY KEY REFERENCES language_response_candidates(candidate_id)
        ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    construction_id TEXT NOT NULL
);

CREATE INDEX idx_candidate_constructions_pattern
    ON language_candidate_constructions (guild_id, construction_id);
