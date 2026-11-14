CREATE TABLE language_proposition_reports (
    report_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_proposition_reports_scope
    ON language_proposition_reports (guild_id, channel_id, speaker_id, observed_at);

CREATE INDEX idx_proposition_reports_event
    ON language_proposition_reports (source_event_id);

CREATE TABLE language_proposition_terms (
    report_id TEXT NOT NULL REFERENCES language_proposition_reports(report_id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    PRIMARY KEY (term, report_id)
);
