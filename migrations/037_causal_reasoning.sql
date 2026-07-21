CREATE TABLE language_causal_cues (
    cue TEXT NOT NULL,
    cue_kind TEXT NOT NULL CHECK (cue_kind IN ('marker', 'predicate')),
    relation TEXT NOT NULL CHECK (
        relation IN ('causes', 'enables', 'prevents', 'inhibits', 'precedes')
    ),
    orientation TEXT NOT NULL CHECK (
        orientation IN (
            'cause_before_effect', 'effect_before_cause',
            'condition_then_effect', 'subject_then_object'
        )
    ),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count > 0),
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cue, cue_kind, relation, orientation)
);

CREATE TABLE language_causal_propositions (
    proposition_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    proposition_kind TEXT NOT NULL CHECK (proposition_kind IN ('clause', 'nominal')),
    surface TEXT NOT NULL,
    subject_lemma TEXT,
    predicate_lemma TEXT,
    object_lemma TEXT,
    terms_json TEXT NOT NULL,
    negated INTEGER NOT NULL DEFAULT 0 CHECK (negated IN (0, 1)),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 1 CHECK (evidence_count > 0),
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_causal_propositions_terms
    ON language_causal_propositions (guild_id, predicate_lemma, confidence DESC);

CREATE TABLE language_causal_links (
    link_id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    cause_proposition_id TEXT NOT NULL
        REFERENCES language_causal_propositions(proposition_id) ON DELETE CASCADE,
    effect_proposition_id TEXT NOT NULL
        REFERENCES language_causal_propositions(proposition_id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (
        relation IN ('causes', 'enables', 'prevents', 'inhibits', 'precedes')
    ),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    positive_evidence INTEGER NOT NULL DEFAULT 0 CHECK (positive_evidence >= 0),
    negative_evidence INTEGER NOT NULL DEFAULT 0 CHECK (negative_evidence >= 0),
    evidence_surface TEXT NOT NULL,
    source TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (guild_id, cause_proposition_id, effect_proposition_id, relation)
);

CREATE INDEX idx_causal_links_forward
    ON language_causal_links (
        guild_id, cause_proposition_id, confidence DESC
    );

CREATE INDEX idx_causal_links_reverse
    ON language_causal_links (
        guild_id, effect_proposition_id, confidence DESC
    );

CREATE TABLE language_causal_evidence (
    evidence_id TEXT PRIMARY KEY,
    link_id TEXT NOT NULL REFERENCES language_causal_links(link_id) ON DELETE CASCADE,
    source_event_id TEXT NOT NULL,
    supports_link INTEGER NOT NULL CHECK (supports_link IN (0, 1)),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    surface TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    UNIQUE (link_id, source_event_id)
);

INSERT INTO language_causal_cues (
    cue, cue_kind, relation, orientation, confidence, source
) VALUES
    ('потому что', 'marker', 'causes', 'effect_before_cause', 0.94, 'grammar_prior'),
    ('так как', 'marker', 'causes', 'effect_before_cause', 0.90, 'grammar_prior'),
    ('поскольку', 'marker', 'causes', 'effect_before_cause', 0.90, 'grammar_prior'),
    ('поэтому', 'marker', 'causes', 'cause_before_effect', 0.94, 'grammar_prior'),
    ('следовательно', 'marker', 'causes', 'cause_before_effect', 0.88, 'grammar_prior'),
    ('если', 'marker', 'enables', 'condition_then_effect', 0.86, 'grammar_prior'),
    ('затем', 'marker', 'precedes', 'cause_before_effect', 0.78, 'grammar_prior'),
    ('потом', 'marker', 'precedes', 'cause_before_effect', 0.74, 'grammar_prior'),
    ('вызвать', 'predicate', 'causes', 'subject_then_object', 0.90, 'semantic_prior'),
    ('вызывать', 'predicate', 'causes', 'subject_then_object', 0.90, 'semantic_prior'),
    ('привести', 'predicate', 'causes', 'subject_then_object', 0.88, 'semantic_prior'),
    ('приводить', 'predicate', 'causes', 'subject_then_object', 0.88, 'semantic_prior'),
    ('породить', 'predicate', 'causes', 'subject_then_object', 0.86, 'semantic_prior'),
    ('порождать', 'predicate', 'causes', 'subject_then_object', 0.86, 'semantic_prior'),
    ('спровоцировать', 'predicate', 'causes', 'subject_then_object', 0.86, 'semantic_prior'),
    ('провоцировать', 'predicate', 'causes', 'subject_then_object', 0.86, 'semantic_prior'),
    ('позволить', 'predicate', 'enables', 'subject_then_object', 0.82, 'semantic_prior'),
    ('позволять', 'predicate', 'enables', 'subject_then_object', 0.82, 'semantic_prior'),
    ('способствовать', 'predicate', 'enables', 'subject_then_object', 0.82, 'semantic_prior'),
    ('предотвратить', 'predicate', 'prevents', 'subject_then_object', 0.90, 'semantic_prior'),
    ('предотвращать', 'predicate', 'prevents', 'subject_then_object', 0.90, 'semantic_prior'),
    ('помешать', 'predicate', 'prevents', 'subject_then_object', 0.82, 'semantic_prior'),
    ('мешать', 'predicate', 'inhibits', 'subject_then_object', 0.78, 'semantic_prior'),
    ('ослабить', 'predicate', 'inhibits', 'subject_then_object', 0.78, 'semantic_prior'),
    ('ослаблять', 'predicate', 'inhibits', 'subject_then_object', 0.78, 'semantic_prior');
