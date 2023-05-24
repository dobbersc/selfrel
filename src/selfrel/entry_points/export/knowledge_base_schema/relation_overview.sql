DROP TABLE IF EXISTS relation_overview;
CREATE TABLE relation_overview
(
    sentence_relation_id INTEGER PRIMARY KEY,
    sentence_id          INTEGER NOT NULL,
    relation_id          INTEGER NOT NULL,
    head_id              INTEGER NOT NULL,
    tail_id              INTEGER NOT NULL,
    sentence_text        TEXT    NOT NULL,
    head_text            TEXT    NOT NULL,
    tail_text            TEXT    NOT NULL,
    head_label           TEXT    NOT NULL,
    tail_label           TEXT    NOT NULL,
    label                TEXT    NOT NULL,
    confidence           REAL    NOT NULL,
    occurrence           INTEGER NOT NULL,
    FOREIGN KEY (sentence_relation_id) REFERENCES sentence_relations (sentence_relation_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences (sentence_id),
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id),
    FOREIGN KEY (head_id) REFERENCES entities (entity_id),
    FOREIGN KEY (tail_id) REFERENCES entities (entity_id)
);


INSERT INTO relation_overview
SELECT sentence_relation_id,
       sentence_id,
       relation_id,
       relation.head_id,
       relation.tail_id,
       sentence.text                           sentence_text,
       head.text                               head_text,
       tail.text                               tail_text,
       head.label                              head_label,
       tail.label                              tail_label,
       relation.label,
       sr.confidence,
       count() OVER (PARTITION BY relation_id) occurrence
FROM sentence_relations sr
         JOIN sentences sentence USING (sentence_id)
         JOIN relations relation USING (relation_id)
         JOIN entities head ON head.entity_id = relation.head_id
         JOIN entities tail ON tail.entity_id = relation.tail_id;


-- Create INDEX for each foreign key
CREATE INDEX relation_overview_sentence_relation_id_fkey ON relation_overview (sentence_relation_id);
CREATE INDEX relation_overview_sentence_id_fkey ON relation_overview (sentence_id);
CREATE INDEX relation_overview_relation_id_fkey ON relation_overview (relation_id);
CREATE INDEX relation_overview_head_id_fkey ON relation_overview (head_id);
CREATE INDEX relation_overview_tail_id_fkey ON relation_overview (tail_id);


-- Create INDEX for entity properties
CREATE INDEX relation_overview_head_text_idx ON relation_overview (head_text);
CREATE INDEX relation_overview_tail_text_idx ON relation_overview (tail_text);
CREATE INDEX relation_overview_head_label_idx ON relation_overview (head_label);
CREATE INDEX relation_overview_tail_label_idx ON relation_overview (tail_label);


-- Create INDEX for relation properties
CREATE INDEX relation_overview_label_idx ON relation_overview (label);
CREATE INDEX relation_overview_occurrence_idx ON relation_overview (occurrence);
