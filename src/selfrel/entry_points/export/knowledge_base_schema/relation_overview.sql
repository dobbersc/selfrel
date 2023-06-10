DROP TABLE IF EXISTS relation_overview;
CREATE TABLE relation_overview
(
    sentence_relation_id                 INTEGER PRIMARY KEY,
    sentence_id                          INTEGER NOT NULL,
    relation_id                          INTEGER NOT NULL,
    head_id                              INTEGER NOT NULL,
    tail_id                              INTEGER NOT NULL,
    sentence_text                        TEXT    NOT NULL,
    in_between_text                      TEXT    NOT NULL,
    head_start_position                  INTEGER NOT NULL,
    tail_start_position                  INTEGER NOT NULL,
    head_text                            TEXT    NOT NULL,
    tail_text                            TEXT    NOT NULL,
    head_label                           TEXT    NOT NULL,
    tail_label                           TEXT    NOT NULL,
    head_confidence                      REAL    NOT NULL,
    tail_confidence                      REAL    NOT NULL,
    label                                TEXT    NOT NULL,
    confidence                           REAL    NOT NULL,
    occurrence                           INTEGER NOT NULL,
    occurrence_distinct_sentences        INTEGER NOT NULL,
    occurrence_distinct_in_between_texts INTEGER NOT NULL,
    entropy                              REAL    NOT NULL,
    entropy_distinct_sentences           REAL    NOT NULL,
    entropy_distinct_in_between_texts    REAL    NOT NULL,
    FOREIGN KEY (sentence_relation_id) REFERENCES sentence_relations (sentence_relation_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences (sentence_id),
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id),
    FOREIGN KEY (head_id) REFERENCES entities (entity_id),
    FOREIGN KEY (tail_id) REFERENCES entities (entity_id)
);


INSERT INTO relation_overview
SELECT sentence_relation_id,
       sentence.sentence_id,
       relation_id,
       relation.head_id,
       relation.tail_id,
       sentence.text      sentence_text,
       in_between_text,
       sr.head_start_position,
       sr.tail_start_position,
       head.text          head_text,
       tail.text          tail_text,
       head.label         head_label,
       tail.label         tail_label,
       se_head.confidence head_confidence,
       se_tail.confidence tail_confidence,
       relation.label,
       sr.confidence,
       occurrence,
       occurrence_distinct_sentences,
       occurrence_distinct_in_between_texts,
       entropy,
       entropy_distinct_sentences,
       entropy_distinct_in_between_texts
FROM sentence_relations sr
         JOIN in_between_texts USING (sentence_relation_id)
         JOIN sentences sentence USING (sentence_id)
         JOIN relations relation USING (relation_id)
         JOIN relation_metrics USING (relation_id)
         JOIN entities head ON head.entity_id = relation.head_id
         JOIN entities tail ON tail.entity_id = relation.tail_id
         JOIN sentence_entities se_head
              ON se_head.sentence_id = sentence.sentence_id
                  AND se_head.entity_id = head.entity_id
                  AND se_head.start_position = sr.head_start_position
         JOIN sentence_entities se_tail
              ON se_tail.sentence_id = sentence.sentence_id
                  AND se_tail.entity_id = tail.entity_id
                  AND se_tail.start_position = sr.tail_start_position;


-- Create INDEX on each foreign key
CREATE INDEX relation_overview_sentence_relation_id_fkey ON relation_overview (sentence_relation_id);
CREATE INDEX relation_overview_sentence_id_fkey ON relation_overview (sentence_id);
CREATE INDEX relation_overview_relation_id_fkey ON relation_overview (relation_id);
CREATE INDEX relation_overview_head_id_fkey ON relation_overview (head_id);
CREATE INDEX relation_overview_tail_id_fkey ON relation_overview (tail_id);

-- Create INDEX on sentence text
CREATE INDEX relation_overview_sentence_text_idx ON relation_overview (sentence_text);

-- Create INDEX on entity properties
CREATE INDEX relation_overview_head_start_position_idx ON relation_overview (head_start_position);
CREATE INDEX relation_overview_tail_start_position_idx ON relation_overview (tail_start_position);
CREATE INDEX relation_overview_head_text_idx ON relation_overview (head_text);
CREATE INDEX relation_overview_tail_text_idx ON relation_overview (tail_text);
CREATE INDEX relation_overview_head_label_idx ON relation_overview (head_label);
CREATE INDEX relation_overview_tail_label_idx ON relation_overview (tail_label);
CREATE INDEX relation_overview_head_confidence_idx ON relation_overview (head_confidence);
CREATE INDEX relation_overview_tail_confidence_idx ON relation_overview (tail_confidence);

-- Create INDEX on relation properties
CREATE INDEX relation_overview_label_idx ON relation_overview (label);
CREATE INDEX relation_overview_confidence_idx ON relation_overview (confidence);
CREATE INDEX relation_overview_in_between_text_idx ON relation_overview (in_between_text);

-- Create INDEX on relation metrics
CREATE INDEX relation_overview_occurrence_idx ON relation_overview (occurrence);
CREATE INDEX relation_overview_occurrence_distinct_sentences_idx ON relation_overview (occurrence_distinct_sentences);
CREATE INDEX relation_overview_occurrence_distinct_in_between_texts_idx ON relation_overview (occurrence_distinct_in_between_texts);
CREATE INDEX relation_overview_entropy_idx ON relation_overview (entropy);
CREATE INDEX relation_overview_entropy_distinct_sentences_idx ON relation_overview (entropy_distinct_sentences);
CREATE INDEX relation_overview_entropy_distinct_in_between_texts_idx ON relation_overview (entropy_distinct_in_between_texts);
