CREATE TABLE knowledge_concepts (
    concept_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    language TEXT NOT NULL,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    definition TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    wikidata_id TEXT,
    revision_id INTEGER,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    fetched_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    use_count INTEGER NOT NULL DEFAULT 0 CHECK (use_count >= 0),
    UNIQUE (source, language, external_id)
);

CREATE INDEX idx_knowledge_concepts_title
    ON knowledge_concepts (source, language, normalized_title);

CREATE TABLE knowledge_aliases (
    alias_key TEXT NOT NULL,
    alias_text TEXT NOT NULL,
    concept_id TEXT NOT NULL REFERENCES knowledge_concepts(concept_id) ON DELETE CASCADE,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    PRIMARY KEY (alias_key, concept_id)
);

CREATE INDEX idx_knowledge_aliases_lookup ON knowledge_aliases (alias_key, confidence DESC);

CREATE TABLE knowledge_queries (
    query_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    source TEXT NOT NULL,
    language TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_key TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('explicit', 'uncertainty')),
    uncertainty REAL NOT NULL CHECK (uncertainty BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'cached', 'no_result', 'failed')),
    concept_id TEXT REFERENCES knowledge_concepts(concept_id),
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (source, event_id)
);

CREATE INDEX idx_knowledge_queries_guild
    ON knowledge_queries (guild_id, created_at DESC);

CREATE UNIQUE INDEX idx_knowledge_queries_one_pending_term
    ON knowledge_queries (source, language, query_key)
    WHERE status = 'pending';

CREATE TABLE knowledge_relations (
    relation_id TEXT PRIMARY KEY,
    subject_concept_id TEXT NOT NULL
        REFERENCES knowledge_concepts(concept_id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_title TEXT NOT NULL,
    normalized_object_title TEXT NOT NULL,
    object_concept_id TEXT REFERENCES knowledge_concepts(concept_id),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    source_query_id TEXT REFERENCES knowledge_queries(query_id),
    created_at TEXT NOT NULL,
    UNIQUE (subject_concept_id, predicate, normalized_object_title)
);

CREATE INDEX idx_knowledge_relations_subject
    ON knowledge_relations (subject_concept_id, predicate);

CREATE TABLE knowledge_acquisition_states (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    evidence_balance REAL NOT NULL DEFAULT 1.0 CHECK (evidence_balance BETWEEN 0.0 AND 1.0),
    observed_human_events INTEGER NOT NULL DEFAULT 0 CHECK (observed_human_events >= 0),
    queries_started INTEGER NOT NULL DEFAULT 0 CHECK (queries_started >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id)
);
