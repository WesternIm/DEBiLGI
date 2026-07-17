ALTER TABLE cognition_expectations
    ADD COLUMN source TEXT NOT NULL DEFAULT 'message_observation';

ALTER TABLE cognition_expectations
    ADD COLUMN observation_window_seconds REAL NOT NULL DEFAULT 120.0
        CHECK (observation_window_seconds > 0.0);

ALTER TABLE cognition_expectations
    ADD COLUMN allowed_outcomes_json TEXT NOT NULL
        DEFAULT '["no_response","other_message","same_actor_message","reply","reaction"]';

ALTER TABLE events ADD COLUMN raw_expires_at TEXT;
ALTER TABLE events ADD COLUMN raw_redacted_at TEXT;

CREATE INDEX idx_events_raw_retention
    ON events (raw_expires_at)
    WHERE raw_redacted_at IS NULL AND raw_expires_at IS NOT NULL;

CREATE TABLE cognition_difference_vectors (
    expectation_id TEXT PRIMARY KEY
        REFERENCES cognition_expectations(expectation_id) ON DELETE CASCADE,
    guild_id INTEGER NOT NULL,
    scalar_d REAL NOT NULL CHECK (scalar_d >= 0.0 AND scalar_d <= 1.0),
    outcome_mismatch REAL NOT NULL CHECK (outcome_mismatch BETWEEN 0.0 AND 1.0),
    participant_mismatch REAL NOT NULL CHECK (participant_mismatch BETWEEN 0.0 AND 1.0),
    timing_mismatch REAL NOT NULL CHECK (timing_mismatch BETWEEN 0.0 AND 1.0),
    intensity_mismatch REAL NOT NULL CHECK (intensity_mismatch BETWEEN 0.0 AND 1.0),
    duration_mismatch REAL NOT NULL CHECK (duration_mismatch BETWEEN 0.0 AND 1.0),
    unexpectedness REAL NOT NULL CHECK (unexpectedness BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL
);

CREATE INDEX idx_cognition_differences_guild_created
    ON cognition_difference_vectors (guild_id, created_at DESC);

CREATE TABLE cognition_episodes (
    episode_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    participants_json TEXT NOT NULL DEFAULT '[]',
    key_event_types_json TEXT NOT NULL DEFAULT '[]',
    observed_outcome TEXT,
    unexpectedness REAL NOT NULL DEFAULT 0.0 CHECK (unexpectedness BETWEEN 0.0 AND 1.0),
    importance REAL NOT NULL DEFAULT 0.0 CHECK (importance BETWEEN 0.0 AND 1.0),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    signature TEXT NOT NULL,
    consolidated_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cognition_episodes_open_channel
    ON cognition_episodes (guild_id, channel_id, status, ended_at DESC);

CREATE INDEX idx_cognition_episodes_consolidation
    ON cognition_episodes (status, consolidated_at, ended_at);

CREATE INDEX idx_cognition_episodes_signature
    ON cognition_episodes (guild_id, signature, ended_at DESC);

CREATE TABLE cognition_episode_events (
    episode_id TEXT NOT NULL REFERENCES cognition_episodes(episode_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (episode_id, event_id),
    UNIQUE (episode_id, position)
);

CREATE INDEX idx_cognition_episode_events_event
    ON cognition_episode_events (event_id);

CREATE TABLE cognition_semantic_memories (
    semantic_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('server', 'channel', 'user')),
    scope_id INTEGER NOT NULL,
    feature TEXT NOT NULL,
    estimate REAL NOT NULL CHECK (estimate BETWEEN 0.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    confirmations INTEGER NOT NULL DEFAULT 0 CHECK (confirmations >= 0),
    contradictions INTEGER NOT NULL DEFAULT 0 CHECK (contradictions >= 0),
    updated_at TEXT NOT NULL,
    decay_rate REAL NOT NULL DEFAULT 0.01 CHECK (decay_rate >= 0.0),
    provenance_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE (guild_id, scope_type, scope_id, feature)
);

CREATE INDEX idx_cognition_semantic_candidates
    ON cognition_semantic_memories (guild_id, scope_type, scope_id, confidence DESC, updated_at DESC);

CREATE TABLE cognition_subconscious_weights (
    guild_id INTEGER NOT NULL,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('server', 'channel', 'user')),
    scope_id INTEGER NOT NULL,
    association_key TEXT NOT NULL,
    value REAL NOT NULL CHECK (value BETWEEN -1.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    decay_rate REAL NOT NULL DEFAULT 0.02 CHECK (decay_rate >= 0.0),
    updated_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (guild_id, scope_type, scope_id, association_key)
);

CREATE TABLE cognition_entity_models (
    guild_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('server', 'channel', 'user', 'self')),
    entity_id INTEGER NOT NULL,
    familiarity REAL NOT NULL DEFAULT 0.0 CHECK (familiarity BETWEEN 0.0 AND 1.0),
    interaction_intensity REAL NOT NULL DEFAULT 0.0 CHECK (interaction_intensity BETWEEN 0.0 AND 1.0),
    predictability REAL NOT NULL DEFAULT 0.5 CHECK (predictability BETWEEN 0.0 AND 1.0),
    reply_probability REAL NOT NULL DEFAULT 0.5 CHECK (reply_probability BETWEEN 0.0 AND 1.0),
    average_reply_delay REAL,
    pattern_stability REAL NOT NULL DEFAULT 0.5 CHECK (pattern_stability BETWEEN 0.0 AND 1.0),
    expectation_reliability REAL NOT NULL DEFAULT 0.5 CHECK (expectation_reliability BETWEEN 0.0 AND 1.0),
    unfinished_interactions INTEGER NOT NULL DEFAULT 0 CHECK (unfinished_interactions >= 0),
    bot_action_response REAL NOT NULL DEFAULT 0.5 CHECK (bot_action_response BETWEEN 0.0 AND 1.0),
    observations INTEGER NOT NULL DEFAULT 0 CHECK (observations >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, entity_type, entity_id)
);

CREATE INDEX idx_cognition_entity_models_guild_type
    ON cognition_entity_models (guild_id, entity_type, updated_at DESC);

CREATE TABLE cognition_internal_states (
    guild_id INTEGER NOT NULL,
    subject_id INTEGER NOT NULL,
    unexpectedness REAL NOT NULL CHECK (unexpectedness BETWEEN 0.0 AND 1.0),
    uncertainty REAL NOT NULL CHECK (uncertainty BETWEEN 0.0 AND 1.0),
    activation REAL NOT NULL CHECK (activation BETWEEN 0.0 AND 1.0),
    tension REAL NOT NULL CHECK (tension BETWEEN 0.0 AND 1.0),
    interest REAL NOT NULL CHECK (interest BETWEEN 0.0 AND 1.0),
    familiarity REAL NOT NULL CHECK (familiarity BETWEEN 0.0 AND 1.0),
    expected_control REAL NOT NULL CHECK (expected_control BETWEEN 0.0 AND 1.0),
    goal_alignment REAL NOT NULL CHECK (goal_alignment BETWEEN 0.0 AND 1.0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (guild_id, subject_id)
);

CREATE TABLE cognition_assessments (
    assessment_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    object_type TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    dimensions_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    supporting_episode_ids_json TEXT NOT NULL DEFAULT '[]',
    contradicting_episode_ids_json TEXT NOT NULL DEFAULT '[]',
    alternative_explanations_json TEXT NOT NULL DEFAULT '[]',
    data_gap REAL NOT NULL CHECK (data_gap BETWEEN 0.0 AND 1.0),
    next_review_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_cognition_assessments_object
    ON cognition_assessments (guild_id, object_type, object_id, created_at DESC);

CREATE TABLE cognition_action_intents (
    intent_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN (
        'stay_silent', 'add_reaction', 'short_reply', 'ask_question',
        'continue_topic', 'save_observation', 'check_hypothesis_later',
        'reward_brains', 'remove_brains', 'give_item'
    )),
    target_id INTEGER,
    channel_id INTEGER,
    urgency REAL NOT NULL CHECK (urgency BETWEEN 0.0 AND 1.0),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    expected_effect_json TEXT NOT NULL,
    reasoning_trace_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    decision_context_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    shadow_only INTEGER NOT NULL DEFAULT 1 CHECK (shadow_only = 1),
    executed_at TEXT CHECK (executed_at IS NULL),
    actual_outcome TEXT,
    evaluated_at TEXT
);

CREATE INDEX idx_cognition_action_intents_recent
    ON cognition_action_intents (guild_id, created_at DESC);

CREATE INDEX idx_cognition_action_intents_target
    ON cognition_action_intents (guild_id, target_id, created_at DESC);

CREATE TABLE cognition_core_observations (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'processed', 'failed')),
    processing_ms REAL NOT NULL DEFAULT 0.0 CHECK (processing_ms >= 0.0),
    last_error TEXT,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
