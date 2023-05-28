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


-- Insert relation ids and metric placeholders
INSERT INTO relation_metrics
SELECT relation_id, 0, 0, 0.0, 0.0
FROM relations;


-- Compute relation occurrence
UPDATE relation_metrics
SET occurrence = result.occurrence
FROM (SELECT relation_id, count() occurrence FROM sentence_relations GROUP BY relation_id) result
WHERE relation_metrics.relation_id = result.relation_id;


-- Compute distinct relation occurrence
UPDATE relation_metrics
SET distinct_occurrence = result.distinct_occurrence
FROM (SELECT relation_id, count(DISTINCT sentences.text) distinct_occurrence
      FROM sentence_relations
               JOIN sentences USING (sentence_id)
      GROUP BY relation_id) result
WHERE relation_metrics.relation_id = result.relation_id;


-- Compute relation entropy
WITH relation_occurrence AS
         (SELECT relation_id, count() occurrence, count(DISTINCT sentence_id) distinct_occurrence
          FROM sentence_relations
                   JOIN sentences USING (sentence_id)
          GROUP BY relation_id),

     relation_entropy AS
         (SELECT relation_id,
                 -sum(p * log(2, p)) OVER (PARTITION BY head_id, tail_id)   entropy,
                 -sum(dp * log(2, dp)) OVER (PARTITION BY head_id, tail_id) distinct_entropy
          FROM (SELECT relation_id,
                       head_id,
                       tail_id,
                       cast(occurrence AS REAL) /
                       sum(occurrence) OVER (PARTITION BY head_id, tail_id)          p,
                       cast(distinct_occurrence AS REAL) /
                       sum(distinct_occurrence) OVER (PARTITION BY head_id, tail_id) dp
                FROM relation_occurrence
                         JOIN relations USING (relation_id)))

UPDATE relation_metrics
SET entropy          = relation_entropy.entropy,
    distinct_entropy = relation_entropy.distinct_entropy
FROM relation_entropy
WHERE relation_metrics.relation_id = relation_entropy.relation_id;


-- Create INDEX on foreign key
CREATE INDEX relation_metrics_relation_id_fkey ON relation_metrics (relation_id);

-- Create INDEX on each metric
CREATE INDEX relation_metrics_occurrence_idx ON relation_metrics (occurrence);
CREATE INDEX relation_metrics_distinct_occurrence_idx ON relation_metrics (distinct_occurrence);
CREATE INDEX relation_metrics_entropy_idx ON relation_metrics (entropy);
CREATE INDEX relation_metrics_distinct_entropy_idx ON relation_metrics (distinct_entropy);
