ALTER TABLE language_utterance_experiences
    ADD COLUMN initiative INTEGER NOT NULL DEFAULT 0 CHECK (initiative IN (0, 1));

ALTER TABLE language_utterance_experiences
    ADD COLUMN goal_id TEXT;

CREATE TABLE cognition_self_models (
    guild_id INTEGER PRIMARY KEY,
    utterances_observed INTEGER NOT NULL DEFAULT 0 CHECK (utterances_observed >= 0),
    positive_outcomes INTEGER NOT NULL DEFAULT 0 CHECK (positive_outcomes >= 0),
    negative_outcomes INTEGER NOT NULL DEFAULT 0 CHECK (negative_outcomes >= 0),
    average_reward REAL NOT NULL DEFAULT 0.0 CHECK (average_reward BETWEEN -1.0 AND 1.0),
    initiative_attempts INTEGER NOT NULL DEFAULT 0 CHECK (initiative_attempts >= 0),
    initiative_successes INTEGER NOT NULL DEFAULT 0 CHECK (initiative_successes >= 0),
    efficacy REAL NOT NULL DEFAULT 0.5 CHECK (efficacy BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL
);

CREATE TABLE cognition_autonomy_goals (
    goal_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    target_user_id INTEGER,
    trigger_event_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    response_message_id INTEGER NOT NULL UNIQUE,
    drive REAL NOT NULL CHECK (drive BETWEEN 0.0 AND 1.0),
    reasoning_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'answered', 'continued', 'abandoned')),
    events_waited INTEGER NOT NULL DEFAULT 0 CHECK (events_waited >= 0),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    outcome_event_id TEXT
);

CREATE UNIQUE INDEX idx_cognition_autonomy_one_open_channel
    ON cognition_autonomy_goals (guild_id, channel_id)
    WHERE status = 'open';

CREATE INDEX idx_cognition_autonomy_goals_recent
    ON cognition_autonomy_goals (guild_id, created_at DESC);

CREATE TABLE cognition_initiative_states (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    evidence_balance REAL NOT NULL DEFAULT 1.0 CHECK (evidence_balance BETWEEN 0.0 AND 1.0),
    observed_human_events INTEGER NOT NULL DEFAULT 0 CHECK (observed_human_events >= 0),
    initiatives_started INTEGER NOT NULL DEFAULT 0 CHECK (initiatives_started >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id)
);
