-- Existing reports/meaning records remain source evidence, not a second fact reader.
-- Role/ref indexes below replace the obsolete lemma-overlap index.
DROP TABLE language_proposition_terms;
CREATE TABLE meaning_claims (
    claim_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    guild_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    speaker_id INTEGER NOT NULL,
    subject_ref TEXT NOT NULL,
    predicate TEXT NOT NULL,
    polarity INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    modality_json TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN
        ('self_report', 'other_report', 'agent_observed', 'corpus_form')),
    source_time TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    resolved INTEGER NOT NULL,
    parser_version TEXT NOT NULL,
    thought_json TEXT NOT NULL,
    correction_target TEXT,
    is_correction INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX meaning_claims_lookup
    ON meaning_claims(guild_id, channel_id, speaker_id, predicate, source_time);
CREATE TABLE meaning_arguments (
    claim_id TEXT NOT NULL REFERENCES meaning_claims(claim_id),
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    ref TEXT NOT NULL,
    argument_json TEXT NOT NULL,
    PRIMARY KEY (claim_id, position)
);
CREATE INDEX meaning_arguments_lookup ON meaning_arguments(role, ref, claim_id);
CREATE TABLE meaning_evidence (
    claim_id TEXT NOT NULL REFERENCES meaning_claims(claim_id),
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    source_key TEXT NOT NULL,
    PRIMARY KEY (claim_id, source_event_id, source_key)
);
CREATE TABLE meaning_supersedes (
    claim_id TEXT NOT NULL REFERENCES meaning_claims(claim_id),
    previous_claim_id TEXT NOT NULL REFERENCES meaning_claims(claim_id),
    PRIMARY KEY (claim_id, previous_claim_id),
    CHECK (claim_id != previous_claim_id)
);
CREATE TABLE meaning_claim_projections (
    source_event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    schema_version INTEGER NOT NULL
);
