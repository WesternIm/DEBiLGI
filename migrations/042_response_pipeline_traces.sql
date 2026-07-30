CREATE TABLE language_response_traces (
    trace_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL UNIQUE
        REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    user_id INTEGER,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    selected_route TEXT,
    final_text TEXT,
    response_message_id INTEGER,
    arbitration_decision_id TEXT,
    failure_stage TEXT,
    failure_reason TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX idx_response_traces_recent
    ON language_response_traces (guild_id, started_at DESC);

CREATE INDEX idx_response_traces_status
    ON language_response_traces (status, started_at DESC);

CREATE TABLE language_response_trace_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL
        REFERENCES language_response_traces(trace_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    stage TEXT NOT NULL,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (trace_id, sequence)
);

CREATE INDEX idx_response_trace_steps_trace
    ON language_response_trace_steps (trace_id, sequence);

CREATE INDEX idx_response_trace_steps_stage
    ON language_response_trace_steps (stage, status, created_at DESC);
