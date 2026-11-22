-- Descriptive language evidence is deliberately separate from rewarded agent actions.
CREATE TABLE language_observed_exchanges (
    answer_event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    prompt_event_id TEXT NOT NULL REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    prompt_act TEXT NOT NULL,
    response_act TEXT NOT NULL,
    prompt_features_json TEXT NOT NULL,
    response_forms_json TEXT NOT NULL,
    topic_links_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE INDEX observed_exchanges_selection
    ON language_observed_exchanges(guild_id, prompt_act, observed_at);
