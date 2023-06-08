--- Create INDEX for each foreign key constraint

CREATE INDEX relations_head_id_fkey ON relations (head_id);
CREATE INDEX relations_tail_id_fkey ON relations (tail_id);

CREATE INDEX sentence_entities_sentence_id_fkey ON sentence_entities (sentence_id);
CREATE INDEX sentence_entities_entity_id_fkey ON sentence_entities (entity_id);

CREATE INDEX sentence_relations_sentence_id_fkey ON sentence_relations (sentence_id);
CREATE INDEX sentence_relations_relation_id_fkey ON sentence_relations (relation_id);
