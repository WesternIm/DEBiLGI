CREATE TABLE learning_instruction_examples (
    example_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    user_id INTEGER,
    source_event_id TEXT NOT NULL,
    response_message_id INTEGER NOT NULL,
    instruction_text TEXT NOT NULL,
    input_text TEXT NOT NULL DEFAULT '',
    answer_text TEXT NOT NULL,
    reasoning_json TEXT NOT NULL DEFAULT '{}',
    topic_terms_json TEXT NOT NULL DEFAULT '[]',
    prompt_hash TEXT NOT NULL,
    answer_hash TEXT NOT NULL,
    quality_score REAL NOT NULL DEFAULT 0.55 CHECK (quality_score BETWEEN 0.0 AND 1.0),
    feedback_score REAL NOT NULL DEFAULT 0.0 CHECK (feedback_score BETWEEN -1.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'accepted', 'rejected')),
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (guild_id, source_event_id),
    UNIQUE (guild_id, prompt_hash, answer_hash)
);

CREATE INDEX idx_learning_examples_quality
    ON learning_instruction_examples (guild_id, status, quality_score DESC, updated_at DESC);

CREATE UNIQUE INDEX idx_learning_examples_response
    ON learning_instruction_examples (guild_id, response_message_id);

CREATE TABLE learning_dataset_feedback (
    feedback_event_id TEXT PRIMARY KEY,
    example_id TEXT NOT NULL REFERENCES learning_instruction_examples(example_id)
        ON DELETE CASCADE,
    feedback_kind TEXT NOT NULL,
    score REAL NOT NULL CHECK (score BETWEEN -1.0 AND 1.0),
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_learning_feedback_example
    ON learning_dataset_feedback (example_id, observed_at DESC);

CREATE TABLE learning_graph_nodes (
    node_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    node_type TEXT NOT NULL,
    node_key TEXT NOT NULL,
    label TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (guild_id, node_type, node_key)
);

CREATE INDEX idx_learning_graph_nodes_lookup
    ON learning_graph_nodes (guild_id, node_type, occurrence_count DESC);

CREATE TABLE learning_graph_edges (
    edge_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    source_node_id TEXT NOT NULL REFERENCES learning_graph_nodes(node_id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    target_node_id TEXT NOT NULL REFERENCES learning_graph_nodes(node_id) ON DELETE CASCADE,
    properties_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count > 0),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (guild_id, source_node_id, relation, target_node_id)
);

CREATE INDEX idx_learning_graph_edges_source
    ON learning_graph_edges (guild_id, source_node_id, relation, confidence DESC);

CREATE INDEX idx_learning_graph_edges_target
    ON learning_graph_edges (guild_id, target_node_id, relation, confidence DESC);
