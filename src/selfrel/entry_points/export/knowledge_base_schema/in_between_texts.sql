DROP TABLE IF EXISTS in_between_texts;
CREATE TABLE in_between_texts
(
    sentence_relation_id INTEGER PRIMARY KEY,
    in_between_text      TEXT NOT NULL
);


INSERT INTO in_between_texts
SELECT sentence_relation_id,
       CASE
           WHEN head_end_position < tail_start_position THEN
               trim(substr(sentence_text, head_end_position, tail_start_position - head_end_position))
           WHEN tail_end_position < head_start_position THEN
               trim(substr(sentence_text, tail_end_position, head_start_position - tail_end_position))
           ELSE
               -- The head and tail are overlapping
               ''
           END in_between_text
FROM (SELECT sentence_relation_id,
             sentences.text                           sentence_text,
             head_start_position,
             tail_start_position,
             -- The head and tail end position are exclusive
             head_start_position + length(heads.text) head_end_position,
             tail_start_position + length(tails.text) tail_end_position
      FROM sentence_relations sr
               JOIN relations USING (relation_id)
               JOIN entities heads ON relations.head_id = heads.entity_id
               JOIN entities tails ON relations.tail_id = tails.entity_id
               JOIN sentences USING (sentence_id));


CREATE INDEX in_between_texts_in_between_text_idx ON in_between_texts (in_between_text);
