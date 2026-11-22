-- Operations project into the same canonical graph as human reports.
CREATE TABLE agent_activities (
    activity_id TEXT PRIMARY KEY,
    agent_ref TEXT NOT NULL,
    source_event_id TEXT NOT NULL REFERENCES events(event_id),
    source_key TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'active', 'completed', 'failed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    receipt_id INTEGER,
    claim_id TEXT NOT NULL REFERENCES meaning_claims(claim_id),
    UNIQUE (agent_ref, source_key)
);
CREATE INDEX agent_activity_source ON agent_activities(source_event_id);

-- Only exact repeated, explicitly attributed own outcomes qualify. This is
-- an estimate about an action in a relationship, not a human-like taste.
CREATE VIEW agent_preferences AS
SELECT i.guild_id, i.channel_id, i.user_id, i.behavior_key, i.operation,
       COUNT(DISTINCT a.activity_id) AS support,
       AVG(c.external_value) AS estimate,
       MAX(c.observed_at) AS updated_at,
       GROUP_CONCAT(DISTINCT c.event_id) AS evidence_ids,
       MIN(i.intention_id) AS intention_id
FROM language_meaning_intentions i
JOIN agent_activities a ON a.receipt_id = i.response_message_id
    AND a.source_event_id = i.prompt_event_id AND a.operation = 'send'
JOIN language_meaning_consequences c ON c.intention_id = i.intention_id
JOIN events e ON e.event_id = c.event_id
WHERE i.status = 'observed' AND a.status = 'completed'
    AND c.external_value IS NOT NULL AND c.external_value != 0
    AND i.outcome_type = c.outcome_type AND i.feedback_event_id = c.event_id
    AND e.actor_id = i.user_id AND e.guild_id = i.guild_id AND e.channel_id = i.channel_id
    AND e.source NOT LIKE '%corpus%' AND e.source NOT LIKE 'seed:%'
GROUP BY i.guild_id, i.channel_id, i.user_id, i.behavior_key, i.operation
HAVING COUNT(DISTINCT a.activity_id) >= 2;

-- Preserve the source of locally observed form learning without recovering it
-- from lesson summaries or promoting corpus actors to runtime experience.
ALTER TABLE language_semantic_lesson_evidence
    ADD COLUMN source_event_id TEXT REFERENCES events(event_id);
