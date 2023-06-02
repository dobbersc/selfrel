import pandas as pd
import pytest
from flair.data import Relation, Sentence

from selfrel.selection_strategies import (
    PredictionConfidence,
    SelectionStrategy,
    TotalOccurrence,
    build_relation_overview,
)


def test_build_relation_overview(sentences_with_relation_annotations: list[Sentence]) -> None:
    sentences: list[Sentence] = sentences_with_relation_annotations
    result: pd.DataFrame = build_relation_overview(sentences, entity_label_type="ner", relation_label_type="relation")

    index: pd.MultiIndex = pd.MultiIndex.from_arrays(
        ((0, 1, 2, 2, 3, 3, 4), (0, 0, 0, 1, 0, 1, 0)),
        names=("sentence_index", "relation_index"),
    )
    expected: pd.DataFrame = pd.DataFrame(
        {
            "sentence_text": pd.Series(
                (
                    "Berlin is the capital of Germany.",
                    "Berlin is the capital of Germany.",
                    "Albert Einstein was born in Ulm, Germany.",
                    "Albert Einstein was born in Ulm, Germany.",
                    "Ulm, located in Germany, is the birthplace of Albert Einstein.",
                    "Ulm, located in Germany, is the birthplace of Albert Einstein.",
                    "Amazon was founded by Jeff Bezos.",
                ),
                index=index,
                dtype="string",
            ),
            "head_text": pd.Series(
                ("Berlin", "Berlin", "Albert Einstein", "Ulm", "Albert Einstein", "Ulm", "Amazon"),
                index=index,
                dtype="string",
            ),
            "tail_text": pd.Series(
                ("Germany", "Germany", "Ulm", "Germany", "Ulm", "Germany", "Jeff Bezos"), index=index, dtype="string"
            ),
            "head_label": pd.Series(("LOC", "LOC", "PER", "LOC", "PER", "LOC", "ORG"), index=index, dtype="category"),
            "tail_label": pd.Series(("LOC", "LOC", "LOC", "LOC", "LOC", "LOC", "PER"), index=index, dtype="category"),
            "label": pd.Series(
                ("capital_of", "capital_of", "born_in", "located_in", "born_in", "located_in", "founded_by"),
                index=index,
                dtype="category",
            ),
            "confidence": (1.0,) * 7,
        },
        index=index,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_prediction_confidence(prediction_confidence_sentences: list[Sentence]) -> None:
    sentences: list[Sentence] = prediction_confidence_sentences
    selection_strategy: SelectionStrategy = PredictionConfidence(min_confidence=0.8, top_k=2)

    selected_sentences: list[Sentence] = list(
        selection_strategy.select_relations(sentences, entity_label_type="ner", relation_label_type="relation")
    )
    assert len(selected_sentences) == 1

    selected_sentence: Sentence = selected_sentences[0]
    assert selected_sentence is not sentences[0]
    assert selected_sentence.to_original_text() == "Berlin located in Germany and Hamburg located in Germany."
    assert [(span.text, span.get_label("ner").value) for span in selected_sentence.get_spans("ner")] == [
        ("Berlin", "LOC"),
        ("Germany", "LOC"),
        ("Hamburg", "LOC"),
        ("Germany", "LOC"),
    ]
    assert [
        (
            relation.first.text,
            relation.second.text,
            relation.get_label("relation").value,
            relation.get_label("relation").score,
        )
        for relation in selected_sentence.get_relations("relation")
    ] == [("Berlin", "Germany", "located_in", 0.9), ("Hamburg", "Germany", "located_in", 0.9)]


@pytest.fixture()
def total_occurrence_sentences() -> list[Sentence]:
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


def test_total_occurrence(total_occurrence_sentences: list[Sentence]) -> None:
    sentences: list[Sentence] = total_occurrence_sentences
    selection_strategy: SelectionStrategy = TotalOccurrence(min_occurrence=2)

    selected_sentences: list[Sentence] = list(selection_strategy.select_relations(sentences, label_type="relation"))
    assert len(selected_sentences) == 2

    assert selected_sentences[0] is not sentences[1]
    assert selected_sentences[0].to_original_text() == "Albert Einstein was born in Ulm, Germany."
    assert selected_sentences[1] is not sentences[2]
    assert selected_sentences[1].to_original_text() == "Ulm, located in Germany, is the birthplace of Albert Einstein."

    for selected_sentence in selected_sentences:
        assert {(span.text, span.get_label("ner").value) for span in selected_sentence.get_spans("ner")} == {
            ("Albert Einstein", "PER"),
            ("Ulm", "LOC"),
            ("Germany", "LOC"),
        }
        assert {
            (relation.first.text, relation.second.text, relation.get_label("relation").value)
            for relation in selected_sentence.get_relations("relation")
        } == {("Albert Einstein", "Ulm", "born_in"), ("Ulm", "Germany", "located_in")}
