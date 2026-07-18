INSERT INTO cognition_self_models (
    guild_id,
    utterances_observed,
    positive_outcomes,
    negative_outcomes,
    average_reward,
    initiative_attempts,
    initiative_successes,
    efficacy,
    updated_at
)
SELECT
    guild_id,
    COUNT(*),
    SUM(CASE WHEN reward > 0.0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN reward < 0.0 THEN 1 ELSE 0 END),
    AVG(reward),
    SUM(CASE WHEN initiative = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN initiative = 1 AND reward > 0.0 THEN 1 ELSE 0 END),
    MAX(0.0, MIN(1.0, (AVG(reward) + 1.0) / 2.0)),
    MAX(resolved_at)
FROM language_utterance_experiences
WHERE reward IS NOT NULL
GROUP BY guild_id
ON CONFLICT (guild_id) DO NOTHING;
