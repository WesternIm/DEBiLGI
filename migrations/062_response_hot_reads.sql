-- Existing projections, indexed in the order used during response preparation.
CREATE INDEX observed_exchanges_recent
    ON language_observed_exchanges(guild_id, observed_at DESC, answer_event_id);
CREATE INDEX meaning_intentions_sent_scope
    ON language_meaning_intentions(guild_id, channel_id, user_id, sent_at DESC)
    WHERE response_message_id IS NOT NULL;
CREATE INDEX response_trace_behavior
    ON language_response_trace_steps(trace_id, stage, component, sequence DESC);
-- Evaluated when an event is stored, rather than parsing every payload per turn.
CREATE INDEX events_human_answers
    ON events(event_id, guild_id)
    WHERE COALESCE(json_extract(payload_json, '$.author_is_bot'), 0) = 0;
CREATE INDEX discourse_recent
    ON language_discourse_turns(guild_id, observed_at DESC, source_id);
