CREATE TABLE language_meaning_records (
    meaning_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    record_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX language_meaning_records_scope
    ON language_meaning_records(guild_id, channel_id, speaker_id, observed_at);
