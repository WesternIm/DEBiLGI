-- Descriptive hypotheses never supply rewards to language_meaning_intentions.
CREATE TABLE language_interlocutor_examples (
    example_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    prompt_event_id TEXT NOT NULL REFERENCES events(event_id),
    response_event_id TEXT REFERENCES events(event_id),
    intention_id TEXT REFERENCES language_meaning_intentions(intention_id),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('observed', 'live')),
    features_json TEXT NOT NULL,
    signature_json TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'unassessed',
    continuation_event_id TEXT REFERENCES events(event_id),
    observed_at TEXT NOT NULL
);
CREATE INDEX interlocutor_examples_scope
    ON language_interlocutor_examples(guild_id, observed_at);
CREATE INDEX interlocutor_examples_response
    ON language_interlocutor_examples(response_event_id);

CREATE TABLE language_clarification_constructions (
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    construction_key TEXT NOT NULL,
    guild_id INTEGER NOT NULL,
    slots_json TEXT NOT NULL,
    focus_mode TEXT NOT NULL CHECK (focus_mode IN ('nominal', 'mention')),
    observed_at TEXT NOT NULL,
    PRIMARY KEY (source_event_id, construction_key)
);
CREATE INDEX clarification_constructions_scope
    ON language_clarification_constructions(guild_id, observed_at);
