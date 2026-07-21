CREATE TABLE language_short_speech_acts (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER,
    act_kind TEXT NOT NULL
        CHECK (act_kind IN ('confirmation', 'rejection', 'continue', 'reaction', 'ambiguous')),
    matched_pattern TEXT,
    recognized INTEGER NOT NULL CHECK (recognized IN (0, 1)),
    coreference_to_bot INTEGER NOT NULL DEFAULT 0 CHECK (coreference_to_bot IN (0, 1)),
    target_message_id INTEGER,
    target_source_id TEXT,
    target_content TEXT,
    interpretation TEXT NOT NULL,
    word_count INTEGER NOT NULL CHECK (word_count >= 0),
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_short_speech_acts_channel
    ON language_short_speech_acts (guild_id, channel_id, observed_at DESC);

CREATE INDEX idx_short_speech_acts_target
    ON language_short_speech_acts (guild_id, target_message_id, act_kind);
