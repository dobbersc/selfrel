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
)
