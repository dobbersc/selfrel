from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from flair.data import Label, Relation, Sentence

from selfrel.utils.copy import deepcopy_flair_sentence

__all__ = ["SelectionStrategy", "PredictionConfidence", "TotalOccurrence", "PMI"]


class SelectionStrategy(ABC):
    @staticmethod
    def _create_selected_sentence(sentence: Sentence, relations: list[Relation], label_type: str) -> Sentence:
        sentence = deepcopy_flair_sentence(sentence)
        sentence.remove_labels(label_type)  # Remove relation annotations

        for relation in relations:
            label: Label = relation.get_label(label_type)
            selected_relation: Relation = Relation(
                first=sentence[relation.first[0].idx - 1 : relation.first[-1].idx],
                second=sentence[relation.second[0].idx - 1 : relation.second[-1].idx],
            )
            selected_relation.add_label(label_type, value=label.value, score=label.score)

        return sentence

    @abstractmethod
    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        pass


class PredictionConfidence(SelectionStrategy):
    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        for sentence in sentences:
            selected_relations: list[Relation] = [
                relation
                for relation in sentence.get_relations(label_type)
                if relation.get_label(label_type).score >= self.confidence_threshold
            ]
            if selected_relations:
                yield self._create_selected_sentence(
                    sentence=sentence, relations=selected_relations, label_type=label_type
                )


class TotalOccurrence(SelectionStrategy):
    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        pass


class PMI(SelectionStrategy):
    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        pass
