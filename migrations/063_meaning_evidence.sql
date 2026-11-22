-- Keep provenance; index independent human evidence without scanning events.
ALTER TABLE meaning_evidence ADD COLUMN meaning_key TEXT;
ALTER TABLE meaning_evidence ADD COLUMN guild_id INTEGER;
ALTER TABLE meaning_evidence ADD COLUMN actor_id INTEGER;
ALTER TABLE meaning_evidence ADD COLUMN observed_at TEXT;
ALTER TABLE meaning_evidence ADD COLUMN polarity INTEGER;
CREATE INDEX meaning_evidence_support
    ON meaning_evidence(guild_id, meaning_key, observed_at, actor_id, polarity);
CREATE INDEX meaning_evidence_source_history
    ON meaning_evidence(guild_id, actor_id, meaning_key);
CREATE INDEX meaning_claims_world
    ON meaning_claims(guild_id, source_kind, predicate, subject_ref, source_time);
CREATE INDEX language_word_relation_dependents ON language_generation_word_relations
    (guild_id, dependent, governor, relation, grammatical_case, dependent_pos);
CREATE INDEX language_word_relation_governors ON language_generation_word_relations
    (guild_id, governor, relation, grammatical_case, dependent_pos, dependent);

-- Historical SQL stays checksum stable. No runtime readers or writers remain.
-- SQLite releases pages to its freelist; deliberately no VACUUM.
DROP TABLE IF EXISTS language_generation_ngrams;
DROP TABLE IF EXISTS language_generation_reader_ngrams;
DROP TABLE IF EXISTS language_generation_observations;
DROP TABLE IF EXISTS language_character_transitions;
DROP TABLE IF EXISTS language_experience_transitions;
