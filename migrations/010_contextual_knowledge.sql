ALTER TABLE knowledge_queries ADD COLUMN actor_id INTEGER;

CREATE TABLE knowledge_event_concepts (
    event_id TEXT NOT NULL,
    concept_id TEXT NOT NULL REFERENCES knowledge_concepts(concept_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    actor_id INTEGER,
    source_query_id TEXT REFERENCES knowledge_queries(query_id),
    explicit INTEGER NOT NULL DEFAULT 0 CHECK (explicit IN (0, 1)),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (event_id, concept_id)
);

CREATE INDEX idx_knowledge_event_concepts_scope
    ON knowledge_event_concepts (guild_id, channel_id, actor_id, observed_at DESC);

CREATE TABLE knowledge_context_associations (
    scope_type TEXT NOT NULL CHECK (scope_type IN ('guild', 'channel', 'user')),
    scope_id INTEGER NOT NULL,
    concept_id TEXT NOT NULL REFERENCES knowledge_concepts(concept_id) ON DELETE CASCADE,
    mention_count INTEGER NOT NULL DEFAULT 0 CHECK (mention_count >= 0),
    explicit_query_count INTEGER NOT NULL DEFAULT 0 CHECK (explicit_query_count >= 0),
    relevance REAL NOT NULL DEFAULT 0.5 CHECK (relevance BETWEEN 0.0 AND 1.0),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL,
    PRIMARY KEY (scope_type, scope_id, concept_id)
);

CREATE INDEX idx_knowledge_context_relevance
    ON knowledge_context_associations (scope_type, scope_id, relevance DESC, mention_count DESC);
