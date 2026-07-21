CREATE TABLE learning_example_sources (
    guild_id INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    example_id TEXT NOT NULL REFERENCES learning_instruction_examples(example_id)
        ON DELETE CASCADE,
    response_message_id INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, source_event_id),
    UNIQUE (guild_id, response_message_id)
);

CREATE INDEX idx_learning_example_sources_example
    ON learning_example_sources (example_id, observed_at DESC);

INSERT OR IGNORE INTO learning_example_sources (
    guild_id, source_event_id, example_id, response_message_id, observed_at
)
SELECT guild_id, source_event_id, example_id, response_message_id, created_at
FROM learning_instruction_examples;
