CREATE TABLE language_dictionary_senses (
    sense_id TEXT PRIMARY KEY,
    lemma TEXT NOT NULL REFERENCES language_dictionary_entries(lemma) ON DELETE CASCADE,
    part_of_speech TEXT,
    definition TEXT NOT NULL,
    context_terms_json TEXT NOT NULL DEFAULT '[]',
    examples_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dictionary_senses_lemma
    ON language_dictionary_senses (lemma, confidence DESC);

CREATE TABLE language_synonym_links (
    guild_id INTEGER NOT NULL DEFAULT 0,
    lemma TEXT NOT NULL,
    synonym_lemma TEXT NOT NULL,
    sense_id TEXT NOT NULL DEFAULT '',
    style_label TEXT NOT NULL DEFAULT 'neutral',
    weight REAL NOT NULL DEFAULT 0.8 CHECK (weight BETWEEN 0.0 AND 1.0),
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, lemma, synonym_lemma, sense_id)
);

CREATE INDEX idx_synonym_reverse
    ON language_synonym_links (guild_id, synonym_lemma, weight DESC);

CREATE TABLE language_homonym_groups (
    group_key TEXT NOT NULL,
    lemma TEXT NOT NULL,
    sense_id TEXT NOT NULL REFERENCES language_dictionary_senses(sense_id) ON DELETE CASCADE,
    discriminator_terms_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (group_key, sense_id)
);

CREATE INDEX idx_homonym_lemma ON language_homonym_groups (lemma, group_key);

CREATE TABLE language_morphology_observations (
    guild_id INTEGER NOT NULL,
    surface_form TEXT NOT NULL,
    lemma TEXT NOT NULL,
    part_of_speech TEXT,
    grammatical_features_json TEXT NOT NULL DEFAULT '{}',
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, surface_form, lemma, grammatical_features_json)
);

CREATE INDEX idx_morphology_lemma
    ON language_morphology_observations (guild_id, lemma, occurrence_count DESC);

CREATE TABLE language_grammar_patterns (
    guild_id INTEGER NOT NULL,
    pattern_key TEXT NOT NULL,
    roles_json TEXT NOT NULL,
    cases_json TEXT NOT NULL DEFAULT '[]',
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, pattern_key)
);

CREATE INDEX idx_grammar_patterns_confidence
    ON language_grammar_patterns (guild_id, confidence DESC, occurrence_count DESC);
