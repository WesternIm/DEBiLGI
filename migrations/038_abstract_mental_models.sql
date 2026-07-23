CREATE TABLE language_mental_models (
    model_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    source_event_id TEXT,
    parent_model_id TEXT REFERENCES language_mental_models(model_id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (
        mode IN (
            'grounded',
            'hypothetical',
            'counterfactual',
            'metaphorical',
            'fictional'
        )
    ),
    concepts_json TEXT NOT NULL DEFAULT '[]',
    assumptions_json TEXT NOT NULL DEFAULT '[]',
    question_focus TEXT,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'working' CHECK (
        status IN ('working', 'retained', 'draft', 'superseded')
    ),
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    UNIQUE (guild_id, source_event_id)
);

CREATE INDEX idx_mental_models_context
    ON language_mental_models (
        guild_id,
        channel_id,
        status,
        last_used_at DESC
    );

CREATE TABLE language_mental_relations (
    model_id TEXT NOT NULL REFERENCES language_mental_models(model_id) ON DELETE CASCADE,
    subject_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    object_key TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    inferred INTEGER NOT NULL DEFAULT 0 CHECK (inferred IN (0, 1)),
    evidence_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (model_id, subject_key, relation, object_key)
);

CREATE INDEX idx_mental_relations_lookup
    ON language_mental_relations (
        relation,
        subject_key,
        object_key,
        confidence DESC
    );
