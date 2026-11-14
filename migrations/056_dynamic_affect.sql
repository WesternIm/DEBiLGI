CREATE TABLE affect_moods (
    guild_id INTEGER PRIMARY KEY,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    dominance REAL NOT NULL,
    novelty REAL NOT NULL,
    affiliation REAL NOT NULL,
    observations INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE affect_emotions (
    guild_id INTEGER NOT NULL,
    actor_id INTEGER NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    dominance REAL NOT NULL,
    novelty REAL NOT NULL,
    affiliation REAL NOT NULL,
    observations INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, actor_id)
);

CREATE TABLE affect_observations (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    actor_id INTEGER NOT NULL,
    stimulus_json TEXT NOT NULL,
    mood_json TEXT NOT NULL,
    emotion_json TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
