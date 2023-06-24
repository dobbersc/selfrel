import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
import scipy as sp
from flair.data import Label, Relation, Sentence
from tqdm import tqdm

from selfrel.utils.copy import deepcopy_flair_sentence
from selfrel.utils.inspect_relations import build_relation_overview, get_in_between_text

if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = ["SelectionReport", "SelectionStrategy", "PredictionConfidence", "Occurrence", "Entropy"]

RELATION_IDENTIFIER: Final[list[str]] = ["head_text", "tail_text", "head_label", "tail_label", "label"]
ENTITY_PAIR_IDENTIFIER: Final[list[str]] = ["head_text", "tail_text", "head_label", "tail_label"]

logger: logging.Logger = logging.getLogger("flair")


def relation_label_counts(relation_overview: pd.DataFrame) -> "pd.Series[int]":
    return relation_overview.groupby("label")["label"].count().sort_values().rename("label_counts")


class SelectionReport(Iterator[Sentence]):
    def __init__(
        self,
        sentences: Iterator[Sentence],
        relation_overview: pd.DataFrame,
        scored_relation_overview: pd.DataFrame,
        selected_relation_overview: pd.DataFrame,
    ) -> None:
        self.sentences = sentences
        self.relation_overview = relation_overview
        self.scored_relation_overview = scored_relation_overview
        self.selected_relation_overview = selected_relation_overview

    def selected_relation_label_counts(self) -> "pd.Series[int]":
        return relation_label_counts(self.selected_relation_overview)

    def __next__(self) -> Sentence:
        return next(self.sentences)


class SelectionStrategy(ABC):
    @abstractmethod
    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def _select_distinct_relation_rows(
        relation_overview: pd.DataFrame, distinct_relations_by: Literal["sentence", "in-between-text"] = "sentence"
    ) -> pd.DataFrame:
        if distinct_relations_by == "in-between-text" and "in_between_text" not in relation_overview:
            relation_overview = relation_overview.assign(in_between_text=get_in_between_text(relation_overview))

        if distinct_relations_by == "sentence":
            return relation_overview.drop_duplicates(subset=[*RELATION_IDENTIFIER, "sentence_text"])

        if distinct_relations_by == "in-between-text":
            return relation_overview.drop_duplicates(subset=[*RELATION_IDENTIFIER, "in_between_text"])

        # pragma: no cover
        msg = (  # type: ignore[unreachable]
            f"Provided invalid value for 'distinct_relations_by': {distinct_relations_by!r}. "
            f"Expected 'sentence' or 'in-between-text'."
        )
        raise ValueError(msg)

    @staticmethod
    def _select_top_rows(
        relation_overview: pd.DataFrame,
        score_column: str,
        ascending: bool = True,
        top_k: Optional[int] = None,
        label_distribution: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        if top_k is None and label_distribution is None:
            return relation_overview

        selected: pd.DataFrame
        if label_distribution is None:
            assert top_k is not None

            # Select top-k data points
            if ascending:
                selected = relation_overview.nsmallest(top_k, columns=score_column, keep="all").head(top_k)
            else:
                selected = relation_overview.nlargest(top_k, columns=score_column, keep="all").head(top_k)

            # Check expected correctness
            if (total_selected_data_points := relation_label_counts(selected).sum()) < top_k:
                warnings.warn(
                    f"Not enough data points are available to select the top {top_k} data points. "
                    f"Proceeding with the top {total_selected_data_points} data points.",
                    stacklevel=3,  # Direct to call of `select_rows`
                )

        else:
            # Normalize the label distribution if its weights don't sum to 1
            weights: npt.NDArray[np.float_] = np.fromiter(label_distribution.values(), dtype=float)
            weights = weights / weights.sum()
            normalized_label_distribution: dict[str, float] = dict(zip(label_distribution.keys(), weights))

            # Select top-k data points under the specified label distribution
            total_data_points: int = len(relation_overview.index) if top_k is None else top_k
            selection_mask: pd.Series[bool] = relation_overview.groupby("label", sort=False)[score_column].transform(
                lambda label_group: label_group.rank(method="first", ascending=ascending)
                <= normalized_label_distribution.get(cast(str, label_group.name), 0.0) * total_data_points
            )
            selected = relation_overview[selection_mask]

            # Check expected correctness
            for label, count in relation_label_counts(selected).items():
                expected_count: int = math.floor(
                    normalized_label_distribution.get(cast(str, label), 0.0) * total_data_points
                )
                if count < expected_count:
                    warnings.warn(
                        f"Not enough data points of label {label!r} are available to select "
                        f"the top {expected_count} data points unter its specified label weight. "
                        f"Proceeding with the top {count} data points.",
                        stacklevel=3,  # Direct to call of `select_rows`
                    )

        return selected

    @staticmethod
    def _create_sentence_with_relations(
        sentence: Sentence, relations: Sequence[Relation], relation_label_type: str
    ) -> Sentence:
        assert all(
            relation.sentence is sentence for relation in relations
        ), "The passed relations do not originate from the passed sentence."
        sentence = deepcopy_flair_sentence(sentence)
        sentence.remove_labels(relation_label_type)  # Remove relation annotations

        for relation in relations:
            label: Label = relation.get_label(relation_label_type)
            selected_relation: Relation = Relation(
                first=sentence[relation.first[0].idx - 1 : relation.first[-1].idx],
                second=sentence[relation.second[0].idx - 1 : relation.second[-1].idx],
            )
            selected_relation.add_label(relation_label_type, value=label.value, score=label.score)

        return sentence

    @classmethod
    def _create_sentences_with_selected_relations(
        cls, sentences: Sequence[Sentence], selected_relation_overview: pd.DataFrame, relation_label_type: str
    ) -> Iterator[Sentence]:
        with tqdm(desc="Creating Selected Data Points", total=len(selected_relation_overview.index)) as progress_bar:
            for sentence_index, group in selected_relation_overview.groupby("sentence_index"):
                assert isinstance(sentence_index, int)
                sentence: Sentence = sentences[sentence_index]
                relations: list[Relation] = sentence.get_relations(relation_label_type)
                selected_relations: list[Relation] = [
                    relations[relation_index] for relation_index in group.index.get_level_values("relation_index")
                ]

                yield cls._create_sentence_with_relations(
                    sentence=sentence, relations=selected_relations, relation_label_type=relation_label_type
                )
                progress_bar.update(len(group.index))

    def select_relations(
        self,
        sentences: Sequence[Sentence],
        entity_label_types: Optional[Union[set[Optional[str]], str]],
        relation_label_type: str,
        precomputed_relation_overview: Optional[pd.DataFrame] = None,
    ) -> SelectionReport:
        relation_overview: pd.DataFrame = (
            build_relation_overview(
                sentences, entity_label_types=entity_label_types, relation_label_type=relation_label_type
            )
            if precomputed_relation_overview is None
            else precomputed_relation_overview
        )

        scored_relation_overview: pd.DataFrame = self.compute_score(relation_overview)
        selected_relation_overview: pd.DataFrame = self.select_rows(scored_relation_overview)
        selected_sentences: Iterator[Sentence] = self._create_sentences_with_selected_relations(
            sentences, selected_relation_overview, relation_label_type
        )

        return SelectionReport(
            sentences=selected_sentences,
            relation_overview=relation_overview,
            scored_relation_overview=scored_relation_overview,
            selected_relation_overview=selected_relation_overview,
        )

    def __repr__(self) -> str:
        attributes: list[str] = [f"{attribute}={value!r}" for attribute, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(attributes)})"


class PredictionConfidence(SelectionStrategy):
    def __init__(
        self,
        min_confidence: float = 0.8,
        distinct_relations_by: Optional[Literal["sentence", "in-between-text"]] = "sentence",
        top_k: Optional[int] = None,
        label_distribution: Optional[dict[str, float]] = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.distinct_relations_by = distinct_relations_by
        self.top_k = top_k
        self.label_distribution = label_distribution

    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        return relation_overview

    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        selected: pd.DataFrame = relation_overview[relation_overview.confidence >= self.min_confidence]
        if self.distinct_relations_by is not None:
            selected = self._select_distinct_relation_rows(selected, self.distinct_relations_by)
        return self._select_top_rows(
            selected,
            score_column="confidence",
            ascending=False,
            top_k=self.top_k,
            label_distribution=self.label_distribution,
        )


class Occurrence(SelectionStrategy):
    def __init__(
        self,
        min_occurrence: int = 2,
        distinct: Optional[Literal["sentence", "in-between-text"]] = None,
        distinct_relations_by: Optional[Literal["sentence", "in-between-text"]] = "sentence",
        top_k: Optional[int] = None,
        label_distribution: Optional[dict[str, float]] = None,
    ) -> None:
        self.min_occurrence = min_occurrence
        self.distinct = distinct
        self.distinct_relations_by = distinct_relations_by
        self.top_k = top_k
        self.label_distribution = label_distribution

    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        if "occurrence" in relation_overview:
            logger.info("Using pre-computed 'occurrence' column")
            return relation_overview

        relation_groups = relation_overview.groupby(RELATION_IDENTIFIER, sort=False)

        occurrences: pd.Series[int]
        if self.distinct is None:
            occurrences = relation_groups["sentence_text"].transform("count")

        elif self.distinct == "sentence":
            occurrences = relation_groups["sentence_text"].transform("nunique")

        elif self.distinct == "in-between-text":
            relation_overview = relation_overview.assign(in_between_text=get_in_between_text(relation_overview))
            relation_groups = relation_overview.groupby(RELATION_IDENTIFIER, sort=False)
            occurrences = relation_groups["in_between_text"].transform("nunique")

        else:  # pragma: no cover
            msg = (  # type: ignore[unreachable]
                f"Provided invalid value for 'distinct': {self.distinct!r}. "
                f"Expected 'None', 'sentence' or 'in-between-text'."
            )
            raise ValueError(msg)

        return relation_overview.assign(occurrence=occurrences)

    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        selected: pd.DataFrame = relation_overview[relation_overview.occurrence >= self.min_occurrence]
        if self.distinct_relations_by is not None:
            selected = self._select_distinct_relation_rows(selected, self.distinct_relations_by)
        return self._select_top_rows(
            selected,
            score_column="occurrence",
            ascending=False,
            top_k=self.top_k,
            label_distribution=self.label_distribution,
        )


class Entropy(SelectionStrategy):
    def __init__(
        self,
        base: Optional[float] = None,
        max_entropy: float = 0.5,
        min_occurrence: Optional[int] = None,
        max_occurrence: Optional[int] = None,
        min_confidence: Optional[float] = None,
        distinct: Optional[Literal["sentence", "in-between-text"]] = None,
        distinct_relations_by: Optional[Literal["sentence", "in-between-text"]] = "sentence",
        top_k: Optional[int] = None,
        label_distribution: Optional[dict[str, float]] = None,
    ) -> None:
        self.base = base
        self.max_entropy = max_entropy
        self.min_occurrence = min_occurrence
        self.max_occurrence = max_occurrence
        self.min_confidence = min_confidence
        self.distinct_relations_by = distinct_relations_by
        self.distinct = distinct
        self.top_k = top_k
        self.label_distribution = label_distribution

    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        if "entropy" in relation_overview:
            logger.info("Using pre-computed 'entropy' column")
            return relation_overview

        relation_overview = Occurrence(distinct=self.distinct).compute_score(relation_overview)
        entropies: pd.DataFrame = (
            relation_overview.drop_duplicates(RELATION_IDENTIFIER)  # type: ignore[call-overload]
            .groupby(ENTITY_PAIR_IDENTIFIER, sort=False, observed=True)["occurrence"]
            .aggregate(entropy=lambda x: sp.stats.entropy(x, base=self.base))
        )
        return relation_overview.join(entropies, on=entropies.index.names)

    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        max_entity_pair_occurrences: pd.Series[int] = relation_overview.groupby(
            ENTITY_PAIR_IDENTIFIER, sort=False, observed=True
        )["occurrence"].transform("max")
        selected: pd.DataFrame = relation_overview[max_entity_pair_occurrences == relation_overview["occurrence"]]

        selected = selected[
            (selected["entropy"] <= self.max_entropy)
            & (selected["occurrence"] >= self.min_occurrence if self.min_occurrence is not None else True)
            & (selected["occurrence"] <= self.max_occurrence if self.max_occurrence is not None else True)
            & (selected["confidence"] >= self.min_confidence if self.min_confidence is not None else True)
        ]

        if self.distinct_relations_by is not None:
            selected = self._select_distinct_relation_rows(selected, self.distinct_relations_by)

        return self._select_top_rows(
            selected,
            score_column="entropy",
            ascending=True,
            top_k=self.top_k,
            label_distribution=self.label_distribution,
        )
