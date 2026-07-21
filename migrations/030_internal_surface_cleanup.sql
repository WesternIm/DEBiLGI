CREATE TEMP TABLE migration_internal_surface_markers (
    marker TEXT PRIMARY KEY
);

INSERT INTO migration_internal_surface_markers (marker) VALUES
    ('команды узнай названного объекта'),
    ('значения и памяти. но не хватает фактов'),
    ('в памяти пока не хватает фактов'),
    ('связь пока неоднозначна'),
    ('уточни, что именно с этим произошло'),
    ('по реплике вижу тему'),
    ('продолжаю прежнюю тему'),
    ('это продолжение темы');

UPDATE learning_instruction_examples
SET status = 'rejected', quality_score = 0.0, updated_at = CURRENT_TIMESTAMP
WHERE EXISTS (
    SELECT 1 FROM migration_internal_surface_markers
    WHERE instr(learning_instruction_examples.answer_text, marker) > 0
);

DELETE FROM learning_graph_nodes
WHERE EXISTS (
    SELECT 1 FROM migration_internal_surface_markers
    WHERE instr(learning_graph_nodes.label, marker) > 0
);

DELETE FROM memory_long_term_capsules
WHERE EXISTS (
    SELECT 1 FROM migration_internal_surface_markers
    WHERE instr(memory_long_term_capsules.summary, marker) > 0
);

UPDATE language_conversation_sessions
SET current_summary = ''
WHERE EXISTS (
    SELECT 1 FROM migration_internal_surface_markers
    WHERE instr(language_conversation_sessions.current_summary, marker) > 0
);

DROP TABLE migration_internal_surface_markers;
