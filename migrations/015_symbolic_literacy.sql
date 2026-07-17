CREATE TABLE cognition_symbol_observations (
    event_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    actor_id INTEGER,
    letters INTEGER NOT NULL CHECK (letters >= 0),
    digits INTEGER NOT NULL CHECK (digits >= 0),
    punctuation INTEGER NOT NULL CHECK (punctuation >= 0),
    symbols INTEGER NOT NULL CHECK (symbols >= 0),
    whitespace INTEGER NOT NULL CHECK (whitespace >= 0),
    numbers_json TEXT NOT NULL,
    scripts_json TEXT NOT NULL,
    query_kinds_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_cognition_symbol_observations_guild
    ON cognition_symbol_observations (guild_id, observed_at DESC);

CREATE TABLE cognition_literacy_states (
    guild_id INTEGER PRIMARY KEY,
    messages_observed INTEGER NOT NULL DEFAULT 0 CHECK (messages_observed >= 0),
    letters_observed INTEGER NOT NULL DEFAULT 0 CHECK (letters_observed >= 0),
    digits_observed INTEGER NOT NULL DEFAULT 0 CHECK (digits_observed >= 0),
    punctuation_observed INTEGER NOT NULL DEFAULT 0 CHECK (punctuation_observed >= 0),
    symbols_observed INTEGER NOT NULL DEFAULT 0 CHECK (symbols_observed >= 0),
    arithmetic_queries INTEGER NOT NULL DEFAULT 0 CHECK (arithmetic_queries >= 0),
    alphabet_queries INTEGER NOT NULL DEFAULT 0 CHECK (alphabet_queries >= 0),
    counting_queries INTEGER NOT NULL DEFAULT 0 CHECK (counting_queries >= 0),
    comparison_queries INTEGER NOT NULL DEFAULT 0 CHECK (comparison_queries >= 0),
    symbol_queries INTEGER NOT NULL DEFAULT 0 CHECK (symbol_queries >= 0),
    updated_at TEXT NOT NULL
);
