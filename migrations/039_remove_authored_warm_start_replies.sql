-- Authored warm-start exchanges were useful during initial bootstrapping but
-- must not remain available as exact prompt-to-reply answers.

UPDATE learning_instruction_examples
SET status = 'rejected',
    quality_score = 0.0,
    updated_at = CURRENT_TIMESTAMP
WHERE source_event_id LIKE 'curriculum:warm-start:%';

DELETE FROM archive_messages
WHERE message_id BETWEEN 8700000000000000000 AND 8700000000000000999
   OR message_id BETWEEN 8710000000000000000 AND 8710000000000000999;
