-- A dialogue task is an unresolved effect in the conversation, not a reply string.
-- Attempts remain separate from tasks so a sent message cannot prove its own success.
CREATE TABLE dialogue_tasks (
    task_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    parent_task_id TEXT REFERENCES dialogue_tasks(task_id),
    signature TEXT NOT NULL,
    kind TEXT NOT NULL,
    goal TEXT NOT NULL,
    objective_json TEXT NOT NULL,
    priority REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    recurrence_count INTEGER NOT NULL DEFAULT 1,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    CHECK (priority >= 0.0 AND priority <= 1.0),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (status IN ('open', 'attempted', 'completed', 'superseded'))
);
CREATE INDEX dialogue_tasks_scope_status
    ON dialogue_tasks(guild_id, channel_id, user_id, status, updated_at);
CREATE INDEX dialogue_tasks_signature
    ON dialogue_tasks(guild_id, channel_id, user_id, signature, status);

CREATE TABLE dialogue_task_events (
    task_id TEXT NOT NULL REFERENCES dialogue_tasks(task_id),
    event_id TEXT NOT NULL REFERENCES events(event_id),
    relation TEXT NOT NULL,
    observation_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (task_id, event_id, relation)
);

CREATE TABLE dialogue_task_attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES dialogue_tasks(task_id),
    prompt_event_id TEXT NOT NULL REFERENCES events(event_id),
    response_message_id INTEGER NOT NULL,
    response_text TEXT NOT NULL,
    predicted_progress REAL NOT NULL,
    assessment TEXT NOT NULL DEFAULT 'pending',
    outcome_event_id TEXT REFERENCES events(event_id),
    created_at TEXT NOT NULL,
    assessed_at TEXT,
    CHECK (predicted_progress >= 0.0 AND predicted_progress <= 1.0),
    CHECK (assessment IN ('pending', 'completed', 'failed', 'continued', 'unassessed'))
);
CREATE UNIQUE INDEX dialogue_task_attempts_response
    ON dialogue_task_attempts(task_id, response_message_id);
CREATE INDEX dialogue_task_attempts_pending
    ON dialogue_task_attempts(task_id, assessment, created_at);

CREATE TABLE dialogue_situations (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    active_task_id TEXT REFERENCES dialogue_tasks(task_id),
    revision INTEGER NOT NULL,
    situation_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id, user_id)
);
