CREATE TABLE language_generation_relation_observations (
    source_event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE
);

CREATE TABLE language_generation_reader_ngrams (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    gram_json TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, channel_id, gram_json)
);

CREATE TABLE language_generation_word_relations (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    governor TEXT NOT NULL,
    dependent TEXT NOT NULL,
    relation TEXT NOT NULL,
    dependent_pos TEXT NOT NULL,
    grammatical_case TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    PRIMARY KEY (guild_id, channel_id, governor, dependent, relation,
                 dependent_pos, grammatical_case)
);
