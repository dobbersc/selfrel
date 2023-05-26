DROP TABLE IF EXISTS relation_metrics;
CREATE TABLE relation_metrics
(
    relation_id         INTEGER PRIMARY KEY,
    occurrence          INTEGER NOT NULL,
    distinct_occurrence INTEGER NOT NULL,
    FOREIGN KEY (relation_id) REFERENCES relations (relation_id)
);


-- Insert relation ids and metric placeholders
INSERT INTO relation_metrics
SELECT relation_id, 0, 0
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


-- Create INDEX on foreign key
CREATE INDEX relation_metrics_relation_id_fkey ON relation_metrics (relation_id);

-- Create INDEX on each metric
CREATE INDEX relation_metrics_occurrence_idx ON relation_metrics (occurrence);
CREATE INDEX relation_metrics_distinct_occurrence_idx ON relation_metrics (distinct_occurrence);
