CREATE TABLE language_utterance_understandings (
    event_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    content_terms_json TEXT NOT NULL DEFAULT '[]',
    known_terms_json TEXT NOT NULL DEFAULT '[]',
    unknown_terms_json TEXT NOT NULL DEFAULT '[]',
    relations_json TEXT NOT NULL DEFAULT '[]',
    context_links_json TEXT NOT NULL DEFAULT '[]',
    comprehension_score REAL NOT NULL CHECK (comprehension_score BETWEEN 0.0 AND 1.0),
    sufficient INTEGER NOT NULL CHECK (sufficient IN (0, 1)),
    observed_at TEXT NOT NULL
);

CREATE INDEX idx_utterance_understanding_unknowns
    ON language_utterance_understandings (guild_id, sufficient, observed_at DESC);

UPDATE language_open_questions
SET status = 'stale', relevance_reason = 'sentence_first_upgrade',
    resolved_at = CURRENT_TIMESTAMP
WHERE status = 'active';
