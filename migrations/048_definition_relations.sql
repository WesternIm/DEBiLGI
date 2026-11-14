CREATE TABLE knowledge_definition_relations (
    concept_id TEXT NOT NULL REFERENCES knowledge_concepts(concept_id) ON DELETE CASCADE,
    sense_key TEXT NOT NULL,
    predicate TEXT NOT NULL CHECK (predicate IN ('is_a', 'part_of', 'used_for', 'consists_of')),
    target TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    PRIMARY KEY (concept_id, sense_key, predicate, target)
);
