-- Sleep gaps are identified by the shared meaning key, and return to open gaps.
ALTER TABLE continuous_cognition_gaps ADD COLUMN meaning_scope TEXT;
ALTER TABLE continuous_cognition_gaps ADD COLUMN visited_occurrences INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_gaps ADD COLUMN visit_cycle INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_gaps ADD COLUMN exhausted_at TEXT;
UPDATE continuous_cognition_gaps SET visited_occurrences=replayed_count
    WHERE signature LIKE 'sleep:%';
ALTER TABLE continuous_cognition_runtime ADD COLUMN sleep_units INTEGER NOT NULL DEFAULT 0;
ALTER TABLE continuous_cognition_runtime ADD COLUMN sleep_restarts INTEGER NOT NULL DEFAULT 0;
CREATE INDEX continuous_sleep_scope ON continuous_cognition_gaps
    (guild_id, meaning_scope) WHERE signature LIKE 'sleep:%';
