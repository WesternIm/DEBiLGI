-- Extend the existing actor, gap queue and revisable hypothesis store.
ALTER TABLE continuous_cognition_runtime ADD COLUMN sleep_state TEXT NOT NULL DEFAULT 'awake'
    CHECK (sleep_state IN ('awake','sleeping'));
ALTER TABLE continuous_cognition_runtime ADD COLUMN last_activity_at TEXT;
ALTER TABLE continuous_cognition_runtime ADD COLUMN cycle_started_at TEXT;
ALTER TABLE continuous_cognition_runtime ADD COLUMN cycle_ends_at TEXT;
ALTER TABLE continuous_cognition_runtime ADD COLUMN pending_activity INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_runtime ADD COLUMN wake_requested INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_runtime ADD COLUMN sleep_cycles INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_runtime ADD COLUMN experiment_units INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_runtime ADD COLUMN maintenance_at TEXT;
ALTER TABLE continuous_cognition_gaps ADD COLUMN replayed_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_gaps ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE continuous_cognition_hypotheses ADD COLUMN sleep_gap_id TEXT
    REFERENCES continuous_cognition_gaps(gap_id);
ALTER TABLE continuous_cognition_hypotheses ADD COLUMN sleep_revision INTEGER NOT NULL DEFAULT 0;
CREATE INDEX continuous_sleep_candidates ON continuous_cognition_hypotheses
    (guild_id, predicate_key, status) WHERE sleep_gap_id IS NOT NULL;
CREATE INDEX continuous_sleep_queue ON continuous_cognition_gaps
    (status, occurrence_count DESC, last_seen_at DESC) WHERE signature LIKE 'sleep:%';
ALTER TABLE language_response_traces ADD COLUMN sleep_captured INTEGER NOT NULL DEFAULT 0;
-- Versioned hydration must not scan obsolete JSON projections on a cold start.
CREATE INDEX analysis_parser_version ON language_exchange_analysis(parser_version);
