CREATE TABLE reply_delivery_jobs (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'prepared', 'sending', 'uncertain',
        'sent', 'completed', 'no_reply', 'failed'
    )),
    prepared_json TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    response_message_id INTEGER,
    sent_at TEXT,
    next_attempt_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_reply_delivery_ready ON reply_delivery_jobs(status, next_attempt_at);

-- Only new gateway messages create delivery work. Historical events are not backfilled.
CREATE TRIGGER create_reply_delivery_job AFTER INSERT ON events
WHEN NEW.source = 'discord'
 AND NEW.event_type IN ('MESSAGE_CREATED', 'MESSAGE_REPLIED')
 AND NEW.channel_id IS NOT NULL
 AND json_extract(NEW.payload_json, '$.author_is_bot') = 0
BEGIN
    INSERT OR IGNORE INTO reply_delivery_jobs(event_id) VALUES (NEW.event_id);
END;
