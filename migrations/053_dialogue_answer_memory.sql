CREATE TABLE language_dialogue_reports (
    answer_event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    question_event_id TEXT NOT NULL REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    answer_text TEXT NOT NULL,
    thought_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX language_dialogue_reports_scope
    ON language_dialogue_reports(guild_id, channel_id, speaker_id, observed_at);
