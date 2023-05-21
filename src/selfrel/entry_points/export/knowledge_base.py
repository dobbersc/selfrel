import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import importlib_resources
from flair.data import Label, Sentence
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset

if TYPE_CHECKING:
    from importlib_resources.abc import Traversable


def get_entity_id(cursor: sqlite3.Cursor, text: str, label: str) -> int:
    entity_id: int = cursor.execute(
        "SELECT entity_id FROM entities WHERE text = ? AND label = ?",
        (text, label),
    ).fetchone()[0]
    return entity_id


def get_relation_id(cursor: sqlite3.Cursor, head_id: int, tail_id: int, label: str) -> int:
    relation_id: int = cursor.execute(
        "SELECT relation_id FROM relations WHERE head_id = ? AND tail_id = ? AND label = ?",
        (head_id, tail_id, label),
    ).fetchone()[0]
    return relation_id


def export_knowledge_base(
    dataset: Union[str, Path, Iterable[Sentence]],
    out: Union[str, Path] = Path("knowledge-base.db"),
    entity_label_type: str = "ner",
    relation_label_type: str = "relation",
) -> None:
    """See `selfrel export knowledge-base --help`."""
    dataset = CoNLLUPlusDataset(dataset) if isinstance(dataset, (str, Path)) else dataset

    connection: sqlite3.Connection = sqlite3.connect(out)
    with connection:
        cursor: sqlite3.Cursor = connection.cursor()

        ddm: Traversable = importlib_resources.files("selfrel.entry_points.export") / "knowledge_base.sql"
        cursor.executescript(ddm.read_text(encoding="utf-8"))

        for sentence in tqdm(dataset, desc="Exporting to Knowledge Base"):
            # Insert sentence
            cursor.execute("INSERT INTO sentences(text) VALUES(?)", (sentence.to_original_text(),))
            sentence_id: Optional[int] = cursor.lastrowid
            assert sentence is not None

            # Insert entity annotations
            for entity in sentence.get_spans(entity_label_type):
                entity_label: Label = entity.get_label(entity_label_type)

                cursor.execute(
                    "INSERT INTO entities(text, label) VALUES(?, ?) ON CONFLICT DO NOTHING",
                    (entity.text, entity_label.value),
                )
                entity_id: int = get_entity_id(cursor, text=entity.text, label=entity_label.value)

                cursor.execute(
                    "INSERT INTO sentence_entities(sentence_id, entity_id, confidence) VALUES(?, ?, ?)",
                    (sentence_id, entity_id, entity_label.score),
                )

            # Insert relation annotations
            for relation in sentence.get_relations(relation_label_type):
                relation_label: Label = relation.get_label(relation_label_type)

                head_id: int = get_entity_id(
                    cursor, text=relation.first.text, label=relation.first.get_label(entity_label_type).value
                )
                tail_id: int = get_entity_id(
                    cursor, text=relation.second.text, label=relation.second.get_label(entity_label_type).value
                )

                cursor.execute(
                    "INSERT INTO relations(head_id, tail_id, label) VALUES(?, ?, ?) ON CONFLICT DO NOTHING",
                    (head_id, tail_id, relation_label.value),
                )
                relation_id: int = get_relation_id(cursor, head_id=head_id, tail_id=tail_id, label=relation_label.value)

                cursor.execute(
                    "INSERT INTO sentence_relations(sentence_id, relation_id, confidence) VALUES(?, ?, ?)",
                    (sentence_id, relation_id, relation.get_label(relation_label_type).score),
                )

    connection.close()
