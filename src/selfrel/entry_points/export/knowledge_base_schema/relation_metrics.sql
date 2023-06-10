-- Compute relation occurrence
DROP TABLE IF EXISTS relation_occurrence;
CREATE TEMPORARY TABLE relation_occurrence
(
    relation_id               INTEGER PRIMARY KEY,
    occurrence                INTEGER NOT NULL,
    distinct_sentences        INTEGER NOT NULL,
    distinct_in_between_texts INTEGER NOT NULL
);

INSERT INTO relation_occurrence
SELECT relation_id,
       count()                         occurrence,
       count(DISTINCT sentences.text)  distinct_sentences,
       count(DISTINCT in_between_text) distinct_in_between_texts
FROM sentence_relations
         JOIN sentences USING (sentence_id)
         JOIN in_between_texts USING (sentence_relation_id)
GROUP BY relation_id;


-- Compute relation entropy
DROP TABLE IF EXISTS relation_entropy;
CREATE TEMPORARY TABLE relation_entropy
(
    relation_id               INTEGER PRIMARY KEY,
    entropy                   REAL NOT NULL,
    distinct_sentences        REAL NOT NULL,
    distinct_in_between_texts REAL NOT NULL
);

INSERT INTO relation_entropy
SELECT relation_id,
       -sum(p * log(2, p)) OVER (PARTITION BY head_id, tail_id)         entropy,
       -sum(p_s * log(2, p_s)) OVER (PARTITION BY head_id, tail_id)     distinct_sentences,
       -sum(p_ibt * log(2, p_ibt)) OVER (PARTITION BY head_id, tail_id) distinct_in_between_texts
FROM (SELECT relation_id,
             head_id,
             tail_id,
             cast(relation_occurrence.occurrence AS REAL) /
             sum(relation_occurrence.occurrence) OVER (PARTITION BY head_id, tail_id)                p,
             cast(relation_occurrence.distinct_sentences AS REAL) /
             sum(relation_occurrence.distinct_sentences) OVER (PARTITION BY head_id, tail_id)        p_s,
             cast(relation_occurrence.distinct_in_between_texts AS REAL) /
             sum(relation_occurrence.distinct_in_between_texts) OVER (PARTITION BY head_id, tail_id) p_ibt
      FROM relations
               JOIN relation_occurrence USING (relation_id));


--- Assemble temporary tables in relation metrics table
DROP TABLE IF EXISTS relation_metrics;
CREATE TABLE relation_metrics
(
    relation_id                          INTEGER PRIMARY KEY,
    occurrence                           INTEGER NOT NULL,
    occurrence_distinct_sentences        INTEGER NOT NULL,
    occurrence_distinct_in_between_texts INTEGER NOT NULL,
    entropy                              REAL    NOT NULL,
    entropy_distinct_sentences           REAL    NOT NULL,
    entropy_distinct_in_between_texts    REAL    NOT NULL,
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id)
);

INSERT INTO relation_metrics
SELECT relation_id,
       relation_occurrence.occurrence,
       relation_occurrence.distinct_sentences,
       relation_occurrence.distinct_in_between_texts,
       relation_entropy.entropy,
       relation_entropy.distinct_sentences,
       relation_entropy.distinct_in_between_texts
FROM relation_occurrence
         JOIN relation_entropy USING (relation_id);

-- Create INDEX on foreign key
CREATE INDEX relation_metrics_relation_id_fkey ON relation_metrics (relation_id);

-- Create INDEX on each metric
CREATE INDEX relation_metrics_occurrence_idx ON relation_metrics (occurrence);
CREATE INDEX relation_metrics_occurrence_distinct_sentences_idx ON relation_metrics (occurrence_distinct_sentences);
CREATE INDEX relation_metrics_occurrence_distinct_in_between_texts_idx ON relation_metrics (occurrence_distinct_in_between_texts);
CREATE INDEX relation_metrics_entropy_idx ON relation_metrics (entropy);
CREATE INDEX relation_metrics_entropy_distinct_sentences_idx ON relation_metrics (entropy_distinct_sentences);
CREATE INDEX relation_metrics_entropy_distinct_in_between_texts_idx ON relation_metrics (entropy_distinct_in_between_texts);
