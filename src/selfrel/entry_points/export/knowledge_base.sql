DROP TABLE IF EXISTS sentences;
CREATE TABLE sentences
(
    sentence_id INTEGER PRIMARY KEY,
    text        TEXT NOT NULL
);


DROP TABLE IF EXISTS entities;
CREATE TABLE entities
(
    entity_id INTEGER PRIMARY KEY,
    text      TEXT NOT NULL,
    label     TEXT NOT NULL,
    UNIQUE (text, label)
);


DROP TABLE IF EXISTS relations;
CREATE TABLE relations
(
    relation_id INTEGER PRIMARY KEY,
    head_id     INTEGER NOT NULL,
    tail_id     INTEGER NOT NULL,
    label       TEXT    NOT NULL,
    UNIQUE (head_id, tail_id, label),
    FOREIGN KEY (head_id) REFERENCES entities (entity_id),
    FOREIGN KEY (tail_id) REFERENCES entities (entity_id)
);


DROP TABLE IF EXISTS sentence_entities;
CREATE TABLE sentence_entities
(
    sentence_entity_id INTEGER PRIMARY KEY,
    sentence_id        INTEGER NOT NULL,
    entity_id          INTEGER NOT NULL,
    confidence         REAL    NOT NULL,
    FOREIGN KEY (sentence_id) REFERENCES sentences (sentence_id),
    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
);


DROP TABLE IF EXISTS sentence_relations;
CREATE TABLE sentence_relations
(
    sentence_relation_id INTEGER PRIMARY KEY,
    sentence_id          INTEGER NOT NULL,
    relation_id          INTEGER NOT NULL,
    confidence           REAL    NOT NULL,
    FOREIGN KEY (sentence_id) REFERENCES sentences (sentence_id),
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id)
);


DROP VIEW IF EXISTS relation_overview;
CREATE VIEW relation_overview AS
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
