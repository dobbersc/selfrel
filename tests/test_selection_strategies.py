import pandas as pd
import pytest
from flair.data import Sentence

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
    """
    Tests the PredictionConfidence selection strategy
    including the underlying DFSelectionStrategy abstract class.
    """
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


# For all further selection strategies inheriting from DFSelectionStrategy,
# we test the `compute_score` and `select_rows` functions.


@pytest.fixture()
def total_occurrence_relation_overview(sentences_with_relation_annotations: list[Sentence]) -> pd.DataFrame:
    sentences: list[Sentence] = sentences_with_relation_annotations
    return build_relation_overview(sentences, entity_label_type="ner", relation_label_type="relation")


class TestTotalOccurrence:
    def test_distinct(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        relation_overview: pd.DataFrame = total_occurrence_relation_overview
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=2, distinct=True)

        scored_relation_overview: pd.DataFrame = selection_strategy.compute_score(relation_overview)
        expected_score: pd.Series[int] = pd.Series(
            (1, 1, 2, 2, 2, 2, 1), index=scored_relation_overview.index, name="occurrence"
        )
        pd.testing.assert_series_equal(scored_relation_overview["occurrence"], expected_score)

        selected_relation_overview: pd.DataFrame = selection_strategy.select_rows(scored_relation_overview)
        expected_index: pd.MultiIndex = pd.MultiIndex.from_arrays(
            ((2, 2, 3, 3), (0, 1, 0, 1)),
            names=("sentence_index", "relation_index"),
        )
        pd.testing.assert_index_equal(selected_relation_overview.index, expected_index)

    def test_not_distinct(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        relation_overview: pd.DataFrame = total_occurrence_relation_overview
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=2, distinct=False)

        scored_relation_overview: pd.DataFrame = selection_strategy.compute_score(relation_overview)
        expected_score: pd.Series[int] = pd.Series(
            (2, 2, 2, 2, 2, 2, 1), index=scored_relation_overview.index, name="occurrence"
        )
        pd.testing.assert_series_equal(scored_relation_overview["occurrence"], expected_score)

        selected_relation_overview: pd.DataFrame = selection_strategy.select_rows(scored_relation_overview)
        expected_index: pd.MultiIndex = pd.MultiIndex.from_arrays(
            ((0, 1, 2, 2, 3, 3), (0, 0, 0, 1, 0, 1)),
            names=("sentence_index", "relation_index"),
        )
        pd.testing.assert_index_equal(selected_relation_overview.index, expected_index)

    def test_top_k(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        relation_overview: pd.DataFrame = total_occurrence_relation_overview
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=1, distinct=True, top_k=5)

        scored_relation_overview: pd.DataFrame = selection_strategy.compute_score(relation_overview)
        selected_relation_overview: pd.DataFrame = selection_strategy.select_rows(scored_relation_overview)
        expected_index: pd.MultiIndex = pd.MultiIndex.from_arrays(
            ((2, 2, 3, 3, 0), (0, 1, 0, 1, 0)),
            names=("sentence_index", "relation_index"),
        )
        pd.testing.assert_index_equal(selected_relation_overview.index, expected_index)
