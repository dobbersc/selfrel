import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import importlib_resources
from flair.data import Label, Sentence
from importlib_resources.abc import Traversable
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.utils.sqlite3 import register_log

__all__ = [
    "export_knowledge_base",
    "create_knowledge_base",
    "update_in_between_texts",
    "update_relation_metrics",
    "update_relation_overview",
]

knowledge_base_sql: Traversable = importlib_resources.files("selfrel.entry_points.export.knowledge_base_schema")


def _get_entity_id(cursor: sqlite3.Cursor, text: str, label: str) -> int:
    entity_id: int = cursor.execute(
        "SELECT entity_id FROM entities WHERE text = ? AND label = ?",
        (text, label),
    ).fetchone()[0]
    return entity_id


def _get_relation_id(cursor: sqlite3.Cursor, head_id: int, tail_id: int, label: str) -> int:
    relation_id: int = cursor.execute(
        "SELECT relation_id FROM relations WHERE head_id = ? AND tail_id = ? AND label = ?",
        (head_id, tail_id, label),
    ).fetchone()[0]
    return relation_id


def create_knowledge_base(
    cursor: sqlite3.Cursor, sentences: Iterable[Sentence], entity_label_type: str, relation_label_type: str
) -> None:
    tables: Traversable = knowledge_base_sql / "tables.sql"
    cursor.executescript(tables.read_text(encoding="utf-8"))

    for sentence in sentences:
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
            entity_id: int = _get_entity_id(cursor, text=entity.text, label=entity_label.value)

            cursor.execute(
                "INSERT INTO sentence_entities(sentence_id, entity_id, start_position, confidence) "
                "VALUES(?, ?, ?, ?)",
                # In SQLite, strings are indexed starting with 1.
                (sentence_id, entity_id, entity.start_position + 1, entity_label.score),
            )

        # Insert relation annotations
        for relation in sentence.get_relations(relation_label_type):
            relation_label: Label = relation.get_label(relation_label_type)

            head_id: int = _get_entity_id(
                cursor, text=relation.first.text, label=relation.first.get_label(entity_label_type).value
            )
            tail_id: int = _get_entity_id(
                cursor, text=relation.second.text, label=relation.second.get_label(entity_label_type).value
            )

            cursor.execute(
                "INSERT INTO relations(head_id, tail_id, label) VALUES(?, ?, ?) ON CONFLICT DO NOTHING",
                (head_id, tail_id, relation_label.value),
            )
            relation_id: int = _get_relation_id(cursor, head_id=head_id, tail_id=tail_id, label=relation_label.value)

            cursor.execute(
                "INSERT INTO sentence_relations(sentence_id, relation_id, "
                "                               head_start_position, tail_start_position, "
                "                               confidence) "
                "VALUES(?, ?, ?, ?, ?)",
                # In SQLite, strings are indexed starting with 1.
                (
                    sentence_id,
                    relation_id,
                    relation.first.start_position + 1,
                    relation.second.start_position + 1,
                    relation.get_label(relation_label_type).score,
                ),
            )

    # Create indices after inserting
    indices: Traversable = knowledge_base_sql / "indices.sql"
    cursor.executescript(indices.read_text(encoding="utf-8"))


def update_in_between_texts(cursor: sqlite3.Cursor) -> None:
    sql: Traversable = knowledge_base_sql / "in_between_texts.sql"
    cursor.executescript(sql.read_text(encoding="utf-8"))


def update_relation_metrics(cursor: sqlite3.Cursor) -> None:
    sql: Traversable = knowledge_base_sql / "relation_metrics.sql"
    cursor.executescript(sql.read_text(encoding="utf-8"))


def update_relation_overview(cursor: sqlite3.Cursor) -> None:
    sql: Traversable = knowledge_base_sql / "relation_overview.sql"
    cursor.executescript(sql.read_text(encoding="utf-8"))


def export_knowledge_base(
    dataset: Union[str, Path, Iterable[Sentence]],
    out: Union[str, Path] = Path("knowledge-base.db"),
    entity_label_type: str = "ner",
    relation_label_type: str = "relation",
    create_relation_overview: bool = True,
) -> None:
    """See `selfrel export knowledge-base --help`."""
    dataset = CoNLLUPlusDataset(dataset, persist=False) if isinstance(dataset, (str, Path)) else dataset

    connection: sqlite3.Connection = sqlite3.connect(out)
    cursor: sqlite3.Cursor = connection.cursor()

    # Register "log" function if not available, e.g. for Python 3.9 on Windows
    register_log(connection)

    print("Building knowledge base...")
    create_knowledge_base(
        cursor,
        sentences=tqdm(dataset, desc=f"Exporting to {str(out)!r}"),
        entity_label_type=entity_label_type,
        relation_label_type=relation_label_type,
    )

    if create_relation_overview:
        print("Building 'in_between_texts' table...")
        update_in_between_texts(cursor)

        print("Building 'relation_metrics' table...")
        update_relation_metrics(cursor)

        print("Building 'relation_overview' table...")
        update_relation_overview(cursor)

    connection.commit()
    connection.close()
    print("Done.")
