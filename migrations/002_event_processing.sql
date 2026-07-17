CREATE TABLE event_processing (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'processed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error TEXT,
    queued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX idx_event_processing_status
    ON event_processing (status, queued_at);

CREATE TRIGGER create_event_processing_state
AFTER INSERT ON events
BEGIN
    INSERT INTO event_processing (event_id) VALUES (NEW.event_id);
END;

INSERT INTO event_processing (event_id)
SELECT event_id FROM events
WHERE event_id NOT IN (SELECT event_id FROM event_processing);

