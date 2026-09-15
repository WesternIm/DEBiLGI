ALTER TABLE language_personal_interpretations
    ADD COLUMN explicitly_taught INTEGER NOT NULL DEFAULT 0 CHECK (explicitly_taught IN (0, 1));

UPDATE language_personal_interpretations AS p SET explicitly_taught = 1
WHERE EXISTS (
    SELECT 1 FROM language_open_questions q
    WHERE q.guild_id = p.guild_id AND q.concept_key = p.normalized_form
      AND q.status = 'learned' AND q.response_text = p.interpretation
);
