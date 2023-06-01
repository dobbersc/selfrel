from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import pandas as pd
from flair.data import Label, Relation, Sentence

from selfrel.utils.copy import deepcopy_flair_sentence

__all__ = ["SelectionStrategy", "PredictionConfidence", "build_relation_overview"]


def build_relation_overview(
    sentences: Sequence[Sentence], entity_label_type: str, relation_label_type: str
) -> pd.DataFrame:
    relation_overview: dict[str, list[Any]] = {
        "sentence_index": [],
        "relation_index": [],
        "sentence_text": [],
        "head_text": [],
        "tail_text": [],
        "head_label": [],
        "tail_label": [],
        "label": [],
        "confidence": [],
    }
    for sentence_index, sentence in enumerate(sentences):
        sentence_text: str = sentence.to_original_text()

        relation: Relation
        for relation_index, relation in enumerate(sentence.get_relations(relation_label_type)):
            relation_overview["sentence_index"].append(sentence_index)
            relation_overview["relation_index"].append(relation_index)

            relation_overview["sentence_text"].append(sentence_text)

            relation_label: Label = relation.get_label(relation_label_type)
            relation_overview["label"].append(relation_label.value)
            relation_overview["confidence"].append(relation_label.score)

            relation_overview["head_text"].append(relation.first.text)
            relation_overview["tail_text"].append(relation.second.text)

            relation_overview["head_label"].append(relation.first.get_label(entity_label_type).value)
            relation_overview["tail_label"].append(relation.second.get_label(entity_label_type).value)

    return pd.DataFrame.from_dict(relation_overview)


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
        self, sentences: Sequence[Sentence], entity_label_type: str, relation_label_type: str
    ) -> Iterator[Sentence]:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class RelationOverviewSelectionStrategy(SelectionStrategy, ABC):
    @abstractmethod
    def compute_score(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def select_rows(self, relation_overview: pd.DataFrame) -> pd.DataFrame:
        pass

    def select_relations(
        self, sentences: Sequence[Sentence], relation_label_type: str, entity_label_type: str
    ) -> Iterator[Sentence]:
        relation_overview: pd.DataFrame = build_relation_overview(
            sentences, entity_label_type=entity_label_type, relation_label_type=relation_label_type
        )

        scored_relation_overview: pd.DataFrame = self.compute_score(relation_overview)
        selected_relation_overview: pd.DataFrame = self.select_rows(scored_relation_overview)

        for sentence_index, group in selected_relation_overview.groupby("sentence_index"):
            sentence: Sentence = sentences[sentence_index]
            relations: list[Relation] = sentence.get_relations(relation_label_type)
            selected_relations: list[Relation] = [
                relations[relation_index] for relation_index in group["relation_index"]
            ]

            yield self._create_selected_sentence(
                sentence=sentence, relations=selected_relations, relation_label_type=relation_label_type
            )


class PredictionConfidence(RelationOverviewSelectionStrategy):
    def __init__(self, min_confidence: float, top_k: Optional[int] = None) -> None:
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
