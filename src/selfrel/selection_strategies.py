from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Optional, Union

import pandas as pd
from flair.data import Label, Relation, Sentence
from tqdm import tqdm

from selfrel.utils.copy import deepcopy_flair_sentence
from selfrel.utils.inspect_relations import build_relation_overview

__all__ = ["SelectionStrategy", "DFSelectionStrategy", "PredictionConfidence", "TotalOccurrence"]


class SelectionStrategy(ABC):
    @staticmethod
    def _create_selected_sentence(
        sentence: Sentence, relations: Sequence[Relation], relation_label_type: str
    ) -> Sentence:
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

    @abstractmethod
    def select_relations(
        self,
        sentences: Sequence[Sentence],
        entity_label_types: Optional[Union[set[Optional[str]], str]],
        relation_label_type: str,
    ) -> Iterator[Sentence]:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class DFSelectionStrategy(SelectionStrategy, ABC):
    @abstractmethod
    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    def select_relations(
        self,
        sentences: Sequence[Sentence],
        entity_label_types: Optional[Union[set[Optional[str]], str]],
        relation_label_type: str,
    ) -> Iterator[Sentence]:
        relation_overview: pd.DataFrame = build_relation_overview(
            sentences, entity_label_types=entity_label_types, relation_label_type=relation_label_type
        )

        scored_relation_overview: pd.DataFrame = self.compute_score(relation_overview)
        selected_relation_overview: pd.DataFrame = self.select_rows(scored_relation_overview)

        with tqdm(desc="Creating Data Points", total=len(selected_relation_overview.index)) as progress_bar:
            for sentence_index, group in selected_relation_overview.groupby("sentence_index"):
                assert isinstance(sentence_index, int)
                sentence: Sentence = sentences[sentence_index]
                relations: list[Relation] = sentence.get_relations(relation_label_type)
                selected_relations: list[Relation] = [
                    relations[relation_index] for relation_index in group.reset_index()["relation_index"]
                ]

                progress_bar.update(len(group.index))
                yield self._create_selected_sentence(
                    sentence=sentence, relations=selected_relations, relation_label_type=relation_label_type
                )


class PredictionConfidence(DFSelectionStrategy):
    def __init__(self, min_confidence: float = 0.8, top_k: Optional[int] = None) -> None:
        self.min_confidence = min_confidence
        self.top_k = top_k

    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        return relation_overview

    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        selected: pd.DataFrame = relation_overview[relation_overview.confidence >= self.min_confidence]
        if self.top_k is not None:
            return selected.nlargest(self.top_k, columns="confidence", keep="all").head(self.top_k)
        return selected

    def __repr__(self) -> str:
        return f"{type(self).__name__}(min_confidence={self.min_confidence!r}, top_k={self.top_k!r})"


class TotalOccurrence(DFSelectionStrategy):
    def __init__(self, min_occurrence: int = 2, distinct: bool = True, top_k: Optional[int] = None) -> None:
        self.min_occurrence = min_occurrence
        self.distinct = distinct
        self.top_k = top_k

    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        relation_identifier: list[str] = ["head_text", "tail_text", "head_label", "tail_label", "label"]
        occurrences: pd.Series[int] = relation_overview.groupby(relation_identifier)["sentence_text"].transform(
            "nunique" if self.distinct else "count"
        )
        return relation_overview.assign(occurrence=occurrences)

    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        selected: pd.DataFrame = relation_overview[relation_overview.occurrence >= self.min_occurrence]
        if self.top_k is not None:
            return selected.nlargest(self.top_k, columns="occurrence", keep="all").head(self.top_k)
        return selected

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"min_occurrence={self.min_occurrence!r}, distinct={self.distinct}, top_k={self.top_k!r})"
        )
