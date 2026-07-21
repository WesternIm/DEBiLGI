CREATE TABLE language_semantic_lessons (
    lesson_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    lesson_kind TEXT NOT NULL CHECK (
        lesson_kind IN (
            'lexical', 'entity', 'relation', 'ambiguity',
            'morphology', 'syntax', 'grammar', 'dialogue', 'style'
        )
    ),
    lesson_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 0 CHECK (evidence_count >= 0),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'active', 'rejected')),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    activated_at TEXT,
    UNIQUE (guild_id, lesson_kind, lesson_key)
);

CREATE INDEX idx_semantic_lessons_lookup
    ON language_semantic_lessons (
        guild_id, status, lesson_kind, confidence DESC, evidence_count DESC
    );

CREATE TABLE language_semantic_lesson_evidence (
    lesson_id TEXT NOT NULL REFERENCES language_semantic_lessons(lesson_id)
        ON DELETE CASCADE,
    input_hash TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (lesson_id, input_hash)
);

CREATE TABLE language_dialogue_act_patterns (
    guild_id INTEGER NOT NULL,
    prompt_act TEXT NOT NULL,
    response_act TEXT NOT NULL,
    register_label TEXT NOT NULL DEFAULT 'neutral',
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count > 0),
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, prompt_act, response_act, register_label, source)
);

CREATE INDEX idx_dialogue_act_patterns_prompt
    ON language_dialogue_act_patterns (
        guild_id, prompt_act, confidence DESC, evidence_count DESC
    );
