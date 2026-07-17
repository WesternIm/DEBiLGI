ALTER TABLE language_utterance_experiences
    ADD COLUMN expected_reward REAL NOT NULL DEFAULT 0.0
        CHECK (expected_reward BETWEEN -1.0 AND 1.0);

CREATE TABLE cognition_reflections (
    reflection_id TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL UNIQUE,
    guild_id INTEGER NOT NULL,
    user_id INTEGER,
    outcome_type TEXT NOT NULL,
    expected_reward REAL NOT NULL CHECK (expected_reward BETWEEN -1.0 AND 1.0),
    actual_reward REAL NOT NULL CHECK (actual_reward BETWEEN -1.0 AND 1.0),
    prediction_error REAL NOT NULL CHECK (prediction_error BETWEEN -2.0 AND 2.0),
    surprise REAL NOT NULL CHECK (surprise BETWEEN 0.0 AND 2.0),
    lesson_kind TEXT NOT NULL CHECK (lesson_kind IN (
        'better_than_expected',
        'worse_than_expected',
        'expectation_confirmed',
        'revise_strategy'
    )),
    strategies_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_cognition_reflections_guild
    ON cognition_reflections (guild_id, created_at DESC);

CREATE INDEX idx_cognition_reflections_user
    ON cognition_reflections (guild_id, user_id, created_at DESC);

CREATE TABLE cognition_strategy_beliefs (
    guild_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    mean_reward REAL NOT NULL DEFAULT 0.0 CHECK (mean_reward BETWEEN -1.0 AND 1.0),
    mean_surprise REAL NOT NULL DEFAULT 0.0 CHECK (mean_surprise BETWEEN 0.0 AND 2.0),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, strategy)
);

CREATE INDEX idx_cognition_strategy_beliefs_reward
    ON cognition_strategy_beliefs (guild_id, mean_reward DESC, confidence DESC);
