CREATE TABLE language_character_transitions (
    guild_id INTEGER NOT NULL,
    order_size INTEGER NOT NULL CHECK (order_size BETWEEN 1 AND 16),
    prefix TEXT NOT NULL,
    next_character TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1 CHECK (count > 0),
    PRIMARY KEY (guild_id, order_size, prefix, next_character)
);

CREATE INDEX idx_character_transitions_prefix
    ON language_character_transitions (guild_id, order_size, prefix);

CREATE TABLE language_dictionary_entries (
    lemma TEXT PRIMARY KEY,
    part_of_speech TEXT,
    meanings_json TEXT NOT NULL DEFAULT '[]',
    style_labels_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE language_dictionary_forms (
    surface_form TEXT NOT NULL,
    lemma TEXT NOT NULL REFERENCES language_dictionary_entries(lemma) ON DELETE CASCADE,
    grammatical_features_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (surface_form, lemma)
);

CREATE INDEX idx_dictionary_forms_lemma ON language_dictionary_forms (lemma);

CREATE TABLE language_personal_interpretations (
    guild_id INTEGER NOT NULL,
    normalized_form TEXT NOT NULL,
    lemma TEXT,
    interpretation TEXT,
    associations_json TEXT NOT NULL DEFAULT '[]',
    supporting_event_ids_json TEXT NOT NULL DEFAULT '[]',
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    confidence REAL NOT NULL DEFAULT 0.1 CHECK (confidence BETWEEN 0.0 AND 1.0),
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, normalized_form)
);

CREATE INDEX idx_personal_interpretations_lemma
    ON language_personal_interpretations (guild_id, lemma, confidence DESC);

CREATE TABLE language_server_slang (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL DEFAULT 0,
    normalized_form TEXT NOT NULL,
    meaning TEXT NOT NULL,
    lemma TEXT,
    style_labels_json TEXT NOT NULL DEFAULT '["server_slang"]',
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, channel_id, normalized_form)
);

CREATE INDEX idx_server_slang_lookup
    ON language_server_slang (guild_id, normalized_form, channel_id, confidence DESC);

CREATE TABLE language_neologism_candidates (
    candidate_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER,
    word TEXT NOT NULL,
    normalized_form TEXT NOT NULL,
    created_at TEXT NOT NULL,
    creator_context TEXT NOT NULL,
    source_components_json TEXT NOT NULL DEFAULT '[]',
    intended_meaning TEXT NOT NULL,
    example_usage TEXT NOT NULL,
    dictionary_match TEXT,
    similar_words_json TEXT NOT NULL DEFAULT '[]',
    usage_count INTEGER NOT NULL DEFAULT 0,
    human_repeat_count INTEGER NOT NULL DEFAULT 0,
    positive_reaction_count INTEGER NOT NULL DEFAULT 0,
    negative_reaction_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.1 CHECK (confidence BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'experimental'
        CHECK (status IN ('experimental', 'understood', 'personal_slang',
                          'server_slang', 'rejected', 'expired')),
    expires_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE (guild_id, normalized_form)
);

CREATE INDEX idx_neologism_status
    ON language_neologism_candidates (guild_id, status, created_at DESC);

CREATE TABLE language_neologism_cooldowns (
    scope_key TEXT PRIMARY KEY,
    last_response_number INTEGER NOT NULL CHECK (last_response_number >= 0),
    updated_at TEXT NOT NULL
);

CREATE TABLE language_rejected_word_forms (
    guild_id INTEGER NOT NULL,
    normalized_form TEXT NOT NULL,
    reason TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY (guild_id, normalized_form)
);

CREATE TABLE language_unknown_concepts (
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL DEFAULT 0,
    concept_key TEXT NOT NULL,
    unknown_kind TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    importance_score REAL NOT NULL DEFAULT 0.0,
    novelty_score REAL NOT NULL DEFAULT 1.0,
    contradiction_score REAL NOT NULL DEFAULT 0.0,
    conversation_relevance REAL NOT NULL DEFAULT 0.0,
    curiosity_score REAL NOT NULL DEFAULT 0.0,
    last_question_at TEXT,
    question_count INTEGER NOT NULL DEFAULT 0,
    resolved INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0, 1)),
    PRIMARY KEY (guild_id, channel_id, concept_key)
);

CREATE INDEX idx_unknown_curiosity
    ON language_unknown_concepts (guild_id, resolved, curiosity_score DESC, last_seen_at DESC);

CREATE TABLE language_conversation_sessions (
    conversation_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    main_topic TEXT,
    topic_history_json TEXT NOT NULL DEFAULT '[]',
    current_summary TEXT NOT NULL DEFAULT '',
    accepted_conclusions_json TEXT NOT NULL DEFAULT '[]',
    active_unknowns_json TEXT NOT NULL DEFAULT '[]',
    unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
    recent_message_ids_json TEXT NOT NULL DEFAULT '[]',
    last_response_plan_id TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closing')),
    UNIQUE (guild_id, channel_id)
);

CREATE INDEX idx_conversation_sessions_idle
    ON language_conversation_sessions (status, last_activity_at);

CREATE TABLE language_conversation_participants (
    conversation_id TEXT NOT NULL REFERENCES language_conversation_sessions(conversation_id)
        ON DELETE CASCADE,
    participant_id INTEGER NOT NULL,
    position_summary TEXT NOT NULL DEFAULT '',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_id)
);

CREATE TABLE language_conversation_episodes (
    episode_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    participants_json TEXT NOT NULL DEFAULT '[]',
    main_topic TEXT,
    summary TEXT NOT NULL,
    development_path_json TEXT NOT NULL DEFAULT '[]',
    conclusions_json TEXT NOT NULL DEFAULT '[]',
    new_concepts_json TEXT NOT NULL DEFAULT '[]',
    unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
    important_message_ids_json TEXT NOT NULL DEFAULT '[]',
    importance_score REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX idx_conversation_episodes_topic
    ON language_conversation_episodes (guild_id, channel_id, ended_at DESC);

CREATE TABLE language_response_plans (
    plan_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    source_event_id TEXT NOT NULL,
    author_id INTEGER,
    topic_ids_json TEXT NOT NULL DEFAULT '[]',
    message_intent TEXT NOT NULL,
    response_goal TEXT NOT NULL,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    tone TEXT NOT NULL,
    unknowns_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'used', 'failed', 'discarded')),
    validation_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX idx_response_plans_source
    ON language_response_plans (source_event_id, created_at DESC);
