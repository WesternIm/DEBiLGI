CREATE TABLE language_generation_observations (
    source_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE TABLE language_generation_ngrams (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    gram_json TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, channel_id, gram_json)
);
