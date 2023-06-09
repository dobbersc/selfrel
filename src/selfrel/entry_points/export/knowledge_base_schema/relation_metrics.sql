-- Compute (distinct) relation occurrence
DROP TABLE IF EXISTS relation_occurrence;
CREATE TEMPORARY TABLE relation_occurrence
(
    relation_id         INTEGER PRIMARY KEY,
    occurrence          INTEGER NOT NULL,
    distinct_occurrence INTEGER NOT NULL
);

INSERT INTO relation_occurrence
SELECT relation_id, count() occurrence, count(DISTINCT sentences.text)
FROM sentence_relations
         JOIN sentences USING (sentence_id)
GROUP BY relation_id;


-- Compute relation entropy
DROP TABLE IF EXISTS relation_entropy;
CREATE TEMPORARY TABLE relation_entropy
(
    relation_id      INTEGER PRIMARY KEY,
    entropy          REAL    NOT NULL,
    distinct_entropy INTEGER NOT NULL
);

INSERT INTO relation_entropy
SELECT relation_id,
       -sum(p * log(2, p)) OVER (PARTITION BY head_id, tail_id)   entropy,
       -sum(dp * log(2, dp)) OVER (PARTITION BY head_id, tail_id) distinct_entropy
FROM (SELECT relation_id,
             head_id,
             tail_id,
             cast(occurrence AS REAL) /
             sum(occurrence) OVER (PARTITION BY head_id, tail_id)          p,
             cast(distinct_occurrence AS REAL) /
             sum(distinct_occurrence) OVER (PARTITION BY head_id, tail_id) dp
      FROM relations
               JOIN relation_occurrence USING (relation_id));


--- Assemble temporary tables in relation metrics table
DROP TABLE IF EXISTS relation_metrics;
CREATE TABLE relation_metrics
(
    relation_id         INTEGER PRIMARY KEY,
    occurrence          INTEGER NOT NULL,
    distinct_occurrence INTEGER NOT NULL,
    entropy             REAL    NOT NULL,
    distinct_entropy    REAL    NOT NULL,
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id)
);

INSERT INTO relation_metrics
SELECT relation_id, occurrence, distinct_occurrence, entropy, distinct_entropy
FROM relation_occurrence
         JOIN relation_entropy USING (relation_id);

-- Create INDEX on foreign key
CREATE INDEX relation_metrics_relation_id_fkey ON relation_metrics (relation_id);

-- Create INDEX on each metric
CREATE INDEX relation_metrics_occurrence_idx ON relation_metrics (occurrence);
CREATE INDEX relation_metrics_distinct_occurrence_idx ON relation_metrics (distinct_occurrence);
CREATE INDEX relation_metrics_entropy_idx ON relation_metrics (entropy);
CREATE INDEX relation_metrics_distinct_entropy_idx ON relation_metrics (distinct_entropy);
