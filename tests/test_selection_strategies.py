from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest
from flair.data import Sentence

from selfrel.selection_strategies import PredictionConfidence, SelectionStrategy, TotalOccurrence
from selfrel.utils.inspect_relations import build_relation_overview


def test_prediction_confidence(prediction_confidence_sentences: list[Sentence]) -> None:
    """
    Tests the PredictionConfidence selection strategy `select_relations` method.
    For all further selection strategies, we test the `compute_score` and `select_rows` methods
    since the underlying `select_relation` is the same for all selection strategies.
    """
    sentences: list[Sentence] = prediction_confidence_sentences
    selection_strategy: SelectionStrategy = PredictionConfidence(min_confidence=0.8, top_k=2)

    # TODO: Test `SelectionReport` correctness and `precomputed_relation_overview` parameter
    selected_sentences: list[Sentence] = list(
        selection_strategy.select_relations(sentences, entity_label_types="ner", relation_label_type="relation")
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


def assert_scores_and_selected_indices(
    selection_strategy: SelectionStrategy,
    relation_overview: pd.DataFrame,
    score_name: str,
    expected_scores: Sequence[Any],
    expected_scores_indices: Sequence[tuple[int, int]],
    expected_selected_indices: Sequence[tuple[int, int]],
) -> None:
    """Utility function to assert the correctness of the `compute_score` and `select_rows` methods."""
    scored_relation_overview: pd.DataFrame = selection_strategy.compute_score(relation_overview)
    expected_score_series: pd.Series[Any] = pd.Series(
        expected_scores,
        index=pd.MultiIndex.from_tuples(expected_scores_indices, names=("sentence_index", "relation_index")),
        name=score_name,
    )
    pd.testing.assert_series_equal(scored_relation_overview[score_name], expected_score_series)

    expected_selected_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
        expected_selected_indices, names=("sentence_index", "relation_index")
    )
    selected_relation_overview: pd.DataFrame = selection_strategy.select_rows(scored_relation_overview)
    pd.testing.assert_index_equal(selected_relation_overview.index, expected_selected_index)


@pytest.fixture()
def total_occurrence_relation_overview(sentences_with_relation_annotations: list[Sentence]) -> pd.DataFrame:
    sentences: list[Sentence] = sentences_with_relation_annotations
    return build_relation_overview(sentences, entity_label_types="ner", relation_label_type="relation")


class TestTotalOccurrence:
    def test_distinct(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=2, distinct=True)
        assert_scores_and_selected_indices(
            selection_strategy=selection_strategy,
            relation_overview=total_occurrence_relation_overview,
            score_name="occurrence",
            expected_scores=(1, 1, 2, 2, 2, 2, 1),
            expected_scores_indices=((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)),
            expected_selected_indices=((2, 0), (2, 1), (3, 0), (3, 1)),
        )

    def test_not_distinct(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=2, distinct=False)
        assert_scores_and_selected_indices(
            selection_strategy=selection_strategy,
            relation_overview=total_occurrence_relation_overview,
            score_name="occurrence",
            expected_scores=(2, 2, 2, 2, 2, 2, 1),
            expected_scores_indices=((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)),
            expected_selected_indices=((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
        )

    def test_top_k(self, total_occurrence_relation_overview: pd.DataFrame) -> None:
        selection_strategy: TotalOccurrence = TotalOccurrence(min_occurrence=1, distinct=True, top_k=5)
        assert_scores_and_selected_indices(
            selection_strategy=selection_strategy,
            relation_overview=total_occurrence_relation_overview,
            score_name="occurrence",
            expected_scores=(1, 1, 2, 2, 2, 2, 1),
            expected_scores_indices=((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)),
            expected_selected_indices=((2, 0), (2, 1), (3, 0), (3, 1), (0, 0)),
        )
