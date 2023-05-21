import sqlite3
from pathlib import Path

import pytest
from flair.data import Relation, Sentence

from selfrel.entry_points.export.knowledge_base import export_knowledge_base


@pytest.fixture()
def dataset() -> list[Sentence]:
    sentences: list[Sentence] = [
        Sentence("Berlin is the capital of Germany."),
        Sentence("Albert Einstein was born in Ulm, Germany."),
        Sentence("Ulm, located in Germany, is the birthplace of Albert Einstein."),
        Sentence("This is a sentence."),
    ]

    # Annotate "Berlin is the capital of Germany."
    sentence: Sentence = sentences[0]
    sentence[:1].add_label("ner", value="LOC")
    sentence[5:6].add_label("ner", value="LOC")
    Relation(first=sentence[:1], second=sentence[5:6]).add_label("relation", value="capital_of")

    # Annotate "Albert Einstein was born in Ulm, Germany."
    sentence = sentences[1]
    sentence[:2].add_label("ner", value="PER")
    sentence[5:6].add_label("ner", value="LOC")
    sentence[7:8].add_label("ner", value="LOC")
    Relation(first=sentence[:2], second=sentence[5:6]).add_label("relation", value="born_in")
    Relation(first=sentence[5:6], second=sentence[7:8]).add_label("relation", value="located_in")

    # Annotate "Ulm, located in Germany, is the birthplace of Albert Einstein."
    sentence = sentences[2]
    sentence[:1].add_label("ner", value="LOC")
    sentence[4:5].add_label("ner", value="LOC")
    sentence[10:12].add_label("ner", value="PER")
    Relation(first=sentence[10:12], second=sentence[:1]).add_label("relation", value="born_in")
    Relation(first=sentence[:1], second=sentence[4:5]).add_label("relation", value="located_in")

    return sentences


def test_export_knowledge_base(dataset: list[Sentence], tmp_path: Path) -> None:
    export_knowledge_base(dataset, out=tmp_path / "knowledge-base.db")

    connection: sqlite3.Connection = sqlite3.connect(tmp_path / "knowledge-base.db")
    cursor: sqlite3.Cursor = connection.cursor()

    # Test contents for "sentences" table
    assert cursor.execute("SELECT * FROM sentences ORDER BY sentence_id").fetchall() == [
        (1, "Berlin is the capital of Germany."),
        (2, "Albert Einstein was born in Ulm, Germany."),
        (3, "Ulm, located in Germany, is the birthplace of Albert Einstein."),
        (4, "This is a sentence."),
    ]

    # Test contents for "entities" table
    assert cursor.execute("SELECT * FROM entities ORDER BY entity_id").fetchall() == [
        (1, "Berlin", "LOC"),
        (2, "Germany", "LOC"),
        (3, "Albert Einstein", "PER"),
        (4, "Ulm", "LOC"),
    ]

    # Test contents for "relations" table
    assert cursor.execute("SELECT * FROM relations ORDER BY relation_id").fetchall() == [
        (1, 1, 2, "capital_of"),  # Berlin -> Germany
        (2, 3, 4, "born_in"),  # Albert Einstein -> Ulm
        (3, 4, 2, "located_in"),  # Ulm -> Germany
    ]

    # Test contents for "sentence_entities" table
    assert cursor.execute("SELECT * FROM sentence_entities ORDER BY sentence_entity_id").fetchall() == [
        # Sentence 1
        (1, 1, 1, 1.0),  # Berlin
        (2, 1, 2, 1.0),  # Germany
        # Sentence 2
        (3, 2, 3, 1.0),  # Albert Einstein
        (4, 2, 4, 1.0),  # Ulm
        (5, 2, 2, 1.0),  # Germany
        # Sentence 3
        (6, 3, 4, 1.0),  # Ulm
        (7, 3, 2, 1.0),  # Germany
        (8, 3, 3, 1.0),  # Albert Einstein
    ]

    # Test contents for "sentence_relations" table
    assert cursor.execute("SELECT * FROM sentence_relations ORDER BY sentence_relation_id").fetchall() == [
        # Sentence 1
        (1, 1, 1, 1.0),  # Berlin -> Germany
        # Sentence 2
        (2, 2, 2, 1.0),  # Albert Einstein -> Ulm
        (3, 2, 3, 1.0),  # Ulm -> Germany
        # Sentence 3
        (4, 3, 2, 1.0),  # Albert Einstein -> Ulm
        (5, 3, 3, 1.0),  # Ulm -> Germany
    ]

    connection.close()