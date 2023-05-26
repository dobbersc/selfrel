import sqlite3
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from _pytest.tmpdir import TempPathFactory
from flair.data import Relation, Sentence

from selfrel.entry_points.export.knowledge_base import export_knowledge_base

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def dataset() -> list[Sentence]:
    sentences: list[Sentence] = [
        Sentence("Berlin is the capital of Germany."),
        Sentence("Berlin is the capital of Germany."),
        Sentence("Albert Einstein was born in Ulm, Germany."),
        Sentence("Ulm, located in Germany, is the birthplace of Albert Einstein."),
        Sentence("Amazon was founded by Jeff Bezos."),
        Sentence("This is a sentence."),
    ]
    sentence: Sentence

    # Annotate "Berlin is the capital of Germany."
    for sentence in sentences[:2]:
        sentence[:1].add_label("ner", value="LOC")
        sentence[5:6].add_label("ner", value="LOC")
        Relation(first=sentence[:1], second=sentence[5:6]).add_label("relation", value="capital_of")

    # Annotate "Albert Einstein was born in Ulm, Germany."
    sentence = sentences[2]
    sentence[:2].add_label("ner", value="PER")
    sentence[5:6].add_label("ner", value="LOC")
    sentence[7:8].add_label("ner", value="LOC")
    Relation(first=sentence[:2], second=sentence[5:6]).add_label("relation", value="born_in")
    Relation(first=sentence[5:6], second=sentence[7:8]).add_label("relation", value="located_in")

    # Annotate "Ulm, located in Germany, is the birthplace of Albert Einstein."
    sentence = sentences[3]
    sentence[:1].add_label("ner", value="LOC")
    sentence[4:5].add_label("ner", value="LOC")
    sentence[10:12].add_label("ner", value="PER")
    Relation(first=sentence[10:12], second=sentence[:1]).add_label("relation", value="born_in")
    Relation(first=sentence[:1], second=sentence[4:5]).add_label("relation", value="located_in")

    # Annotate "Amazon was founded by Jeff Bezos."
    sentence = sentences[4]
    sentence[:1].add_label("ner", value="ORG")
    sentence[4:6].add_label("ner", value="PER")
    Relation(first=sentence[:1], second=sentence[4:6]).add_label("relation", value="founded_by")

    return sentences


@pytest.fixture(scope="module")
def knowledge_base(dataset: list[Sentence], tmp_path_factory: TempPathFactory) -> Iterator[sqlite3.Cursor]:
    database_path: Path = tmp_path_factory.getbasetemp() / "knowledge-base.db"
    export_knowledge_base(dataset, out=database_path)

    connection: sqlite3.Connection = sqlite3.connect(database_path)
    cursor: sqlite3.Cursor = connection.cursor()

    yield cursor

    cursor.close()
    connection.close()


def test_sentences(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM sentences ORDER BY sentence_id").fetchall() == [
        (1, "Berlin is the capital of Germany."),
        (2, "Berlin is the capital of Germany."),
        (3, "Albert Einstein was born in Ulm, Germany."),
        (4, "Ulm, located in Germany, is the birthplace of Albert Einstein."),
        (5, "Amazon was founded by Jeff Bezos."),
        (6, "This is a sentence."),
    ]


def test_entities(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM entities ORDER BY entity_id").fetchall() == [
        (1, "Berlin", "LOC"),
        (2, "Germany", "LOC"),
        (3, "Albert Einstein", "PER"),
        (4, "Ulm", "LOC"),
        (5, "Amazon", "ORG"),
        (6, "Jeff Bezos", "PER"),
    ]


def test_relations(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM relations ORDER BY relation_id").fetchall() == [
        (1, 1, 2, "capital_of"),  # Berlin -> Germany
        (2, 3, 4, "born_in"),  # Albert Einstein -> Ulm
        (3, 4, 2, "located_in"),  # Ulm -> Germany
        (4, 5, 6, "founded_by"),  # Amazon -> Jeff Bezos
    ]


def test_sentence_entities(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM sentence_entities ORDER BY sentence_entity_id").fetchall() == [
        # Sentence 1: "Berlin is the capital of Germany."
        (1, 1, 1, 1.0),  # Berlin
        (2, 1, 2, 1.0),  # Germany,
        # Sentence 2: "Berlin is the capital of Germany."
        (3, 2, 1, 1.0),  # Berlin
        (4, 2, 2, 1.0),  # Germany
        # Sentence 3: "Albert Einstein was born in Ulm, Germany."
        (5, 3, 3, 1.0),  # Albert Einstein
        (6, 3, 4, 1.0),  # Ulm
        (7, 3, 2, 1.0),  # Germany
        # Sentence 4: "Ulm, located in Germany, is the birthplace of Albert Einstein."
        (8, 4, 4, 1.0),  # Ulm
        (9, 4, 2, 1.0),  # Germany
        (10, 4, 3, 1.0),  # Albert Einstein
        # Sentence 5: "Amazon was founded by Jeff Bezos."
        (11, 5, 5, 1.0),  # Amazon
        (12, 5, 6, 1.0),  # Jeff Bezos
    ]


def test_sentence_relations(knowledge_base: sqlite3.Cursor) -> None:
    assert knowledge_base.execute("SELECT * FROM sentence_relations ORDER BY sentence_relation_id").fetchall() == [
        # Sentence 1: "Berlin is the capital of Germany."
        (1, 1, 1, 1.0),  # Berlin -> Germany
        # Sentence 2: "Berlin is the capital of Germany."
        (2, 2, 1, 1.0),  # Berlin -> Germany
        # Sentence 3: "Albert Einstein was born in Ulm, Germany."
        (3, 3, 2, 1.0),  # Albert Einstein -> Ulm
        (4, 3, 3, 1.0),  # Ulm -> Germany
        # Sentence 4: "Ulm, located in Germany, is the birthplace of Albert Einstein."
        (5, 4, 2, 1.0),  # Albert Einstein -> Ulm
        (6, 4, 3, 1.0),  # Ulm -> Germany
        # Sentence 5: "Amazon was founded by Jeff Bezos."
        (7, 5, 4, 1.0),  # Amazon -> Jeff Bezos
    ]


def test_relation_metrics(knowledge_base: sqlite3.Cursor) -> None:
    # | relation_id | occurrence | distinct_occurrence |
    assert knowledge_base.execute("SELECT * FROM relation_metrics ORDER BY relation_id").fetchall() == [
        (1, 2, 1),  # Berlin          ---capital_of--> Germany
        (2, 2, 2),  # Albert Einstein ---born_in-----> Ulm
        (3, 2, 2),  # Ulm             ---located_in--> Germany
        (4, 1, 1),  # Amazon          ---founded_by--> Jeff
    ]


def test_relation_overview(knowledge_base: sqlite3.Cursor) -> None:
    # | sentence_relation_id | sentence_id | relation_id | head_id | tail_id |
    # | sentence_text |
    # | head_text | tail_text | head_label | tail_label | label | confidence |
    # fmt: off
    assert knowledge_base.execute("SELECT * FROM relation_overview ORDER BY sentence_relation_id").fetchall() == [
        (
            1, 1, 1, 1, 2,
            "Berlin is the capital of Germany.",
            "Berlin", "Germany", "LOC", "LOC", "capital_of", 1.0,
        ),
        (
            2, 2, 1, 1, 2,
            "Berlin is the capital of Germany.",
            "Berlin", "Germany", "LOC", "LOC", "capital_of", 1.0,
        ),
        (
            3, 3, 2, 3, 4,
            "Albert Einstein was born in Ulm, Germany.",
            "Albert Einstein", "Ulm", "PER", "LOC", "born_in", 1.0,
        ),
        (
            4, 3, 3, 4, 2,
            "Albert Einstein was born in Ulm, Germany.",
            "Ulm", "Germany", "LOC", "LOC", "located_in", 1.0,
        ),
        (
            5, 4, 2, 3, 4,
            "Ulm, located in Germany, is the birthplace of Albert Einstein.",
            "Albert Einstein", "Ulm", "PER", "LOC", "born_in", 1.0,
        ),
        (
            6, 4, 3, 4, 2,
            "Ulm, located in Germany, is the birthplace of Albert Einstein.",
            "Ulm", "Germany", "LOC", "LOC", "located_in", 1.0,
        ),
        (
            7, 5, 4, 5, 6,
            "Amazon was founded by Jeff Bezos.",
            "Amazon", "Jeff Bezos", "ORG", "PER", "founded_by", 1.0,
        ),
    ]
    # fmt: on
