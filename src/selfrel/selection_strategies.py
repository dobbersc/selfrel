from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from flair.data import Relation, Sentence

from selfrel.utils.copy import deepcopy_flair_sentence

__all__ = ["SelectionStrategy", "PredictionConfidence", "TotalOccurrence", "PMI"]


class SelectionStrategy(ABC):
    @abstractmethod
    def select_relations(self, sentences: Iterable[Sentence], label_type: str) -> Iterator[Sentence]:
        pass


class PredictionConfidence(SelectionStrategy):
    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def select_relations(self, sentences: Iterable[Sentence], label_type: str) -> Iterator[Sentence]:
        for sentence in sentences:
            selected_sentence: Sentence = deepcopy_flair_sentence(sentence)
            selected_sentence.remove_labels(label_type)  # Remove relation annotations

            selected_relations: list[Relation] = []
            for relation in sentence.get_relations(label_type):
                if (label := relation.get_label(label_type)).score >= self.confidence_threshold:
                    selected_relation: Relation = Relation(
                        first=selected_sentence[relation.first[0].idx - 1 : relation.first[-1].idx],
                        second=selected_sentence[relation.second[0].idx - 1 : relation.second[-1].idx],
                    )
                    selected_relation.add_label(label_type, value=label.value, score=label.score)
                    selected_relations.append(selected_relation)

            if selected_relations:
                yield selected_sentence


class TotalOccurrence(SelectionStrategy):
    def select_relations(self, sentences: Iterable[Sentence], label_type: str) -> Iterator[Sentence]:
        pass


class PMI(SelectionStrategy):
    def select_relations(self, sentences: Iterable[Sentence], label_type: str) -> Iterator[Sentence]:
        pass
