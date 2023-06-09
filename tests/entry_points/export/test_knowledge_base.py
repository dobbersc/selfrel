import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from flair.data import Sentence

from selfrel.entry_points.export.knowledge_base import export_knowledge_base


@contextlib.contextmanager
def get_cursor(database_path: Path) -> Iterator[sqlite3.Cursor]:
    connection: sqlite3.Connection = sqlite3.connect(database_path)
    cursor: sqlite3.Cursor = connection.cursor()

    yield cursor

    cursor.close()
    connection.close()


@pytest.fixture()
def knowledge_base(sentences_with_relation_annotations: list[Sentence], tmp_path: Path) -> Iterator[sqlite3.Cursor]:
    """The knowledge base used for the basic table tests."""
    database_path: Path = tmp_path / "knowledge-base.db"
    export_knowledge_base(dataset=sentences_with_relation_annotations, out=database_path)
    with get_cursor(database_path) as cursor:
        yield cursor


@pytest.fixture()
def entropy_knowledge_base(entropy_sentences: list[Sentence], tmp_path: Path) -> Iterator[sqlite3.Cursor]:
    """The knowledge base used for the entropy relation calculation test."""
    database_path: Path = tmp_path / "knowledge-base.db"
    export_knowledge_base(entropy_sentences, out=database_path)
    with get_cursor(database_path) as cursor:
        yield cursor


class TestTables:
    def test_sentences(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute("SELECT * FROM sentences ORDER BY sentence_id").fetchall() == [
            (1, "Berlin is the capital of Germany."),
            (2, "Berlin is the capital of Germany."),
            (3, "Albert Einstein was born in Ulm, Germany."),
            (4, "Ulm, located in Germany, is the birthplace of Albert Einstein."),
            (5, "Amazon was founded by Jeff Bezos."),
            (6, "This is a sentence."),
        ]

    def test_entities(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute("SELECT * FROM entities ORDER BY entity_id").fetchall() == [
            (1, "Berlin", "LOC"),
            (2, "Germany", "LOC"),
            (3, "Albert Einstein", "PER"),
            (4, "Ulm", "LOC"),
            (5, "Amazon", "ORG"),
            (6, "Jeff Bezos", "PER"),
        ]

    def test_relations(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute("SELECT * FROM relations ORDER BY relation_id").fetchall() == [
            (1, 1, 2, "capital_of"),  # Berlin -> Germany
            (2, 3, 4, "born_in"),  # Albert Einstein -> Ulm
            (3, 4, 2, "located_in"),  # Ulm -> Germany
            (4, 5, 6, "founded_by"),  # Amazon -> Jeff Bezos
        ]

    def test_sentence_entities(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute("SELECT * FROM sentence_entities ORDER BY sentence_entity_id").fetchall() == [
            # Sentence 1: "Berlin is the capital of Germany."
            (1, 1, 1, 1, 1.0),  # Berlin
            (2, 1, 2, 26, 1.0),  # Germany,
            # Sentence 2: "Berlin is the capital of Germany."
            (3, 2, 1, 1, 1.0),  # Berlin
            (4, 2, 2, 26, 1.0),  # Germany
            # Sentence 3: "Albert Einstein was born in Ulm, Germany."
            (5, 3, 3, 1, 1.0),  # Albert Einstein
            (6, 3, 4, 29, 1.0),  # Ulm
            (7, 3, 2, 34, 1.0),  # Germany
            # Sentence 4: "Ulm, located in Germany, is the birthplace of Albert Einstein."
            (8, 4, 4, 1, 1.0),  # Ulm
            (9, 4, 2, 17, 1.0),  # Germany
            (10, 4, 3, 47, 1.0),  # Albert Einstein
            # Sentence 5: "Amazon was founded by Jeff Bezos."
            (11, 5, 5, 1, 1.0),  # Amazon
            (12, 5, 6, 23, 1.0),  # Jeff Bezos
        ]

    def test_sentence_relations(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute("SELECT * FROM sentence_relations ORDER BY sentence_relation_id").fetchall() == [
            # Sentence 1: "Berlin is the capital of Germany."
            (1, 1, 1, 1, 26, 1.0),  # Berlin -> Germany
            # Sentence 2: "Berlin is the capital of Germany."
            (2, 2, 1, 1, 26, 1.0),  # Berlin -> Germany
            # Sentence 3: "Albert Einstein was born in Ulm, Germany."
            (3, 3, 2, 1, 29, 1.0),  # Albert Einstein -> Ulm
            (4, 3, 3, 29, 34, 1.0),  # Ulm -> Germany
            # Sentence 4: "Ulm, located in Germany, is the birthplace of Albert Einstein."
            (5, 4, 2, 47, 1, 1.0),  # Albert Einstein -> Ulm
            (6, 4, 3, 1, 17, 1.0),  # Ulm -> Germany
            # Sentence 5: "Amazon was founded by Jeff Bezos."
            (7, 5, 4, 1, 23, 1.0),  # Amazon -> Jeff Bezos
        ]


class TestRelationMetrics:
    def test_occurrence(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute(
            "SELECT relation_id, occurrence FROM relation_metrics ORDER BY relation_id"
        ).fetchall() == [
            (1, 2),  # Berlin          ---capital_of--> Germany
            (2, 2),  # Albert Einstein ---born_in-----> Ulm
            (3, 2),  # Ulm             ---located_in--> Germany
            (4, 1),  # Amazon          ---founded_by--> Jeff
        ]

    def test_distinct_occurrence(self, knowledge_base: sqlite3.Cursor) -> None:
        assert knowledge_base.execute(
            "SELECT relation_id, distinct_occurrence FROM relation_metrics ORDER BY relation_id"
        ).fetchall() == [
            (1, 1),  # Berlin          ---capital_of--> Germany
            (2, 2),  # Albert Einstein ---born_in-----> Ulm
            (3, 2),  # Ulm             ---located_in--> Germany
            (4, 1),  # Amazon          ---founded_by--> Jeff
        ]

    def test_entropy(self, entropy_knowledge_base: sqlite3.Cursor) -> None:
        assert entropy_knowledge_base.execute(
            "SELECT relation_id, round(entropy, 2) FROM relation_metrics ORDER BY relation_id"
        ).fetchall() == [
            (1, 0.72),
            (2, 0.72),
            (3, 0.97),
            (4, 0.97),
        ]


def test_in_between_texts(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM in_between_texts ORDER BY sentence_relation_id").fetchall() == [
        # Sentence 1: "Berlin is the capital of Germany."
        (1, "is the capital of"),  # Berlin -> Germany
        # Sentence 2: "Berlin is the capital of Germany."
        (2, "is the capital of"),  # Berlin -> Germany
        # Sentence 3: "Albert Einstein was born in Ulm, Germany."
        (3, "was born in"),  # Albert Einstein -> Ulm
        (4, ","),  # Ulm -> Germany
        # Sentence 4: "Ulm, located in Germany, is the birthplace of Albert Einstein."
        (5, ", located in Germany, is the birthplace of"),  # Albert Einstein -> Ulm
        (6, ", located in"),  # Ulm -> Germany
        # Sentence 5: "Amazon was founded by Jeff Bezos."
        (7, "was founded by"),  # Amazon -> Jeff Bezos
    ]


def test_relation_overview(knowledge_base: sqlite3.Cursor, resources_dir: Path) -> None:
    result: list[tuple[Any, ...]] = knowledge_base.execute(
        "SELECT * FROM relation_overview ORDER BY sentence_relation_id"
    ).fetchall()

    expected: list[tuple[Any, ...]] = list(
        pd.read_csv(resources_dir / "knowledge_base" / "relation-overview.tsv", sep="\t").itertuples(index=False)
    )

    assert result == expected
