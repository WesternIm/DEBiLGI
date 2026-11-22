-- Disposable parser projections, invalidated by parser/configuration fingerprint.
CREATE TABLE language_exchange_analysis (
    text_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (parser_version, kind, text_hash)
);
ALTER TABLE meaning_claims ADD COLUMN scope TEXT NOT NULL DEFAULT 'private';
ALTER TABLE meaning_claims ADD COLUMN time_start TEXT;
ALTER TABLE meaning_claims ADD COLUMN time_end TEXT;
ALTER TABLE meaning_evidence ADD COLUMN voter INTEGER NOT NULL DEFAULT 0;
ALTER TABLE meaning_evidence ADD COLUMN scope TEXT NOT NULL DEFAULT 'private';
ALTER TABLE meaning_evidence ADD COLUMN time_start TEXT;
ALTER TABLE meaning_evidence ADD COLUMN time_end TEXT;
UPDATE meaning_claims SET time_start=json_extract(thought_json,'$.semantic_time.start'),
    time_end=json_extract(thought_json,'$.semantic_time.end');
CREATE INDEX meaning_claims_interval ON meaning_claims
    (guild_id,source_kind,polarity,scope,time_start,time_end);
CREATE INDEX meaning_evidence_voters ON meaning_evidence
    (guild_id,meaning_key,voter,scope,time_start,time_end,observed_at,source_event_id);
CREATE INDEX meaning_evidence_human_history ON meaning_evidence
    (guild_id,voter,scope,actor_id,meaning_key,observed_at,source_event_id);
CREATE TABLE meaning_positions (
    guild_id INTEGER NOT NULL,
    meaning_key TEXT NOT NULL,
    actor_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    interval_key TEXT NOT NULL,
    polarity INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    PRIMARY KEY (guild_id,meaning_key,scope,interval_key,actor_id)
);
CREATE TABLE meaning_support (
    guild_id INTEGER NOT NULL,
    meaning_key TEXT NOT NULL,
    scope TEXT NOT NULL,
    interval_key TEXT NOT NULL,
    support_json TEXT NOT NULL,
    PRIMARY KEY (guild_id,meaning_key,scope,interval_key)
);
CREATE TABLE meaning_evidence_revision (
    guild_id INTEGER PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0
);
CREATE TRIGGER meaning_evidence_insert_revision AFTER INSERT ON meaning_evidence BEGIN
    INSERT INTO meaning_evidence_revision VALUES (NEW.guild_id,1)
    ON CONFLICT(guild_id) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER meaning_evidence_update_revision AFTER UPDATE ON meaning_evidence BEGIN
    INSERT INTO meaning_evidence_revision VALUES (NEW.guild_id,1)
    ON CONFLICT(guild_id) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER meaning_evidence_delete_revision AFTER DELETE ON meaning_evidence BEGIN
    INSERT INTO meaning_evidence_revision VALUES (OLD.guild_id,1)
    ON CONFLICT(guild_id) DO UPDATE SET revision=revision+1;
END;
-- Preserve the old event count as provenance; response evidence now counts humans.
ALTER TABLE continuous_cognition_hypotheses ADD COLUMN legacy_positive_events INTEGER;
UPDATE continuous_cognition_hypotheses SET legacy_positive_events=positive_evidence
    WHERE signature LIKE 'response:%';
UPDATE continuous_cognition_hypotheses AS h SET positive_evidence=(
    SELECT COUNT(*) FROM (
        SELECT e.actor_id FROM continuous_cognition_evidence x JOIN events e
        ON e.event_id=x.source_event_id
        WHERE x.hypothesis_id=h.hypothesis_id AND e.actor_id IS NOT NULL
          AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0
          AND instr(e.source,'corpus')=0 AND e.source NOT LIKE 'seed:%'
        GROUP BY e.actor_id
        HAVING (SELECT y.relation FROM continuous_cognition_evidence y JOIN events z
                ON z.event_id=y.source_event_id
                WHERE y.hypothesis_id=h.hypothesis_id AND z.actor_id=e.actor_id
                  AND COALESCE(json_extract(z.payload_json,'$.author_is_bot'),0)=0
                  AND instr(z.source,'corpus')=0 AND z.source NOT LIKE 'seed:%'
                ORDER BY y.observed_at DESC,y.source_event_id DESC LIMIT 1)='supports'
    )
) WHERE signature LIKE 'response:%';
CREATE TABLE language_causal_positions (
    link_id TEXT NOT NULL REFERENCES language_causal_links(link_id),
    actor_id INTEGER NOT NULL,
    polarity INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    PRIMARY KEY (link_id,actor_id)
);
INSERT INTO language_causal_positions
SELECT link_id,actor_id,polarity,observed_at,source_event_id FROM (
    SELECT x.link_id,e.actor_id,CASE x.supports_link WHEN 1 THEN 1 ELSE -1 END polarity,
        x.observed_at,x.source_event_id,
        ROW_NUMBER() OVER (PARTITION BY x.link_id,e.actor_id
            ORDER BY x.observed_at DESC,x.source_event_id DESC) AS n
    FROM language_causal_evidence x JOIN events e ON e.event_id=x.source_event_id
    WHERE e.actor_id IS NOT NULL AND COALESCE(json_extract(e.payload_json,
        '$.author_is_bot'),0)=0 AND instr(e.source,'corpus')=0 AND e.source NOT LIKE 'seed:%'
) WHERE n=1;
-- Keep materialized provenance current even for replay and explicit revocation.
CREATE TRIGGER meaning_claim_metadata AFTER INSERT ON meaning_claims BEGIN
    UPDATE meaning_claims SET
      scope=CASE WHEN EXISTS(SELECT 1 FROM json_each(NEW.thought_json,'$.dependencies') d
          WHERE json_extract(d.value,'$.role')='subject')
        AND NOT EXISTS(SELECT 1 FROM json_each(NEW.thought_json,'$.dependencies') d,
          json_tree(d.value,'$.argument') w WHERE
          (w.key='participant_ref' AND w.value IS NOT NULL AND w.value!='') OR
          (w.key='lemma' AND w.value IN ('я','мы','ты','вы','мой','наш','твой','ваш')))
        THEN 'world' ELSE 'private' END,
      time_start=json_extract(NEW.thought_json,'$.semantic_time.start'),
      time_end=json_extract(NEW.thought_json,'$.semantic_time.end')
    WHERE claim_id=NEW.claim_id;
END;
CREATE TRIGGER meaning_evidence_metadata AFTER INSERT ON meaning_evidence BEGIN
    UPDATE meaning_evidence SET
      voter=EXISTS(SELECT 1 FROM events e JOIN meaning_claims c ON c.claim_id=NEW.claim_id
        WHERE e.event_id=NEW.source_event_id AND e.guild_id=NEW.guild_id
          AND NEW.actor_id IS NOT NULL AND c.resolved=1
          AND c.source_kind IN ('self_report','other_report')
          AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0
          AND instr(e.source,'corpus')=0 AND e.source NOT LIKE 'seed:%'),
      scope=(SELECT scope FROM meaning_claims WHERE claim_id=NEW.claim_id),
      time_start=(SELECT time_start FROM meaning_claims WHERE claim_id=NEW.claim_id),
      time_end=(SELECT time_end FROM meaning_claims WHERE claim_id=NEW.claim_id)
    WHERE claim_id=NEW.claim_id AND source_event_id=NEW.source_event_id
      AND source_key=NEW.source_key;
    DELETE FROM meaning_support WHERE guild_id=NEW.guild_id AND meaning_key=NEW.meaning_key;
END;
CREATE TRIGGER meaning_evidence_dirty AFTER UPDATE ON meaning_evidence BEGIN
    DELETE FROM meaning_support WHERE guild_id=NEW.guild_id AND meaning_key=NEW.meaning_key;
    DELETE FROM meaning_support WHERE guild_id=OLD.guild_id AND meaning_key=OLD.meaning_key;
END;
CREATE TRIGGER meaning_evidence_removed AFTER DELETE ON meaning_evidence BEGIN
    DELETE FROM meaning_support WHERE guild_id=OLD.guild_id AND meaning_key=OLD.meaning_key;
    DELETE FROM meaning_positions WHERE guild_id=OLD.guild_id AND meaning_key=OLD.meaning_key;
END;
CREATE TRIGGER meaning_claim_resolution AFTER UPDATE OF resolved ON meaning_claims BEGIN
    UPDATE meaning_evidence SET voter=CASE WHEN NEW.resolved=0 THEN 0 ELSE EXISTS(
      SELECT 1 FROM events e WHERE e.event_id=meaning_evidence.source_event_id
      AND e.guild_id=meaning_evidence.guild_id AND meaning_evidence.actor_id IS NOT NULL
      AND NEW.source_kind IN ('self_report','other_report')
      AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0
      AND instr(e.source,'corpus')=0 AND e.source NOT LIKE 'seed:%') END
    WHERE claim_id=NEW.claim_id;
END;
CREATE INDEX meaning_evidence_event ON meaning_evidence(source_event_id,voter,meaning_key);
UPDATE language_causal_links SET
    positive_evidence=(SELECT COUNT(*) FROM language_causal_positions p
        WHERE p.link_id=language_causal_links.link_id AND p.polarity=1),
    negative_evidence=(SELECT COUNT(*) FROM language_causal_positions p
        WHERE p.link_id=language_causal_links.link_id AND p.polarity=-1);
