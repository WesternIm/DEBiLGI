CREATE TABLE language_reference_concepts (
    guild_id INTEGER NOT NULL,
    normalized_form TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count > 0),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, normalized_form, concept_key, source)
);

CREATE INDEX idx_reference_concepts_word
    ON language_reference_concepts (guild_id, normalized_form, confidence DESC);

CREATE TABLE language_usage_patterns (
    guild_id INTEGER NOT NULL,
    normalized_form TEXT NOT NULL,
    lemma TEXT NOT NULL,
    preferred_case TEXT,
    preferred_number TEXT,
    preferred_role TEXT,
    neighbor_lemmas_json TEXT NOT NULL DEFAULT '[]',
    pattern_status TEXT NOT NULL CHECK (pattern_status IN ('stable', 'uncertain')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    example_count INTEGER NOT NULL DEFAULT 0 CHECK (example_count >= 0),
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, normalized_form, source)
);

CREATE INDEX idx_usage_patterns_lemma
    ON language_usage_patterns (guild_id, lemma, confidence DESC);
