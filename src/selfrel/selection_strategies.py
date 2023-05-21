import itertools
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Optional

from flair.data import Label, Relation, Sentence
from tqdm import tqdm

from selfrel.utils.copy import deepcopy_flair_sentence

__all__ = ["SelectionStrategy", "PredictionConfidence", "TotalOccurrence", "PMI"]


# TODO: Write some documentation for this module


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

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PredictionConfidence(SelectionStrategy):
    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        for sentence in tqdm(sentences, desc="Selecting Confident Relations"):
            selected_relations: list[Relation] = [
                relation
                for relation in sentence.get_relations(label_type)
                if relation.get_label(label_type).score >= self.confidence_threshold
            ]
            if selected_relations:
                yield self._create_selected_sentence(
                    sentence=sentence, relations=selected_relations, label_type=label_type
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(confidence_threshold={self.confidence_threshold!r})"


class TotalOccurrence(SelectionStrategy):
    def __init__(self, occurrence_threshold: int = 2, confidence_threshold: Optional[float] = None) -> None:
        self.occurrence_threshold = occurrence_threshold
        self.confidence_threshold = confidence_threshold

    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        # Mapping from the relation instance identifier to the location in the dataset
        # {(head_text, tail_text, label): [(sentence_index, relation_index), ...], ...}
        occurrences: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        for sentence_index, sentence in enumerate(tqdm(sentences, desc="Counting Relation Occurrences")):
            for relation_index, relation in enumerate(sentence.get_relations(label_type)):
                label: Label = relation.get_label(label_type)
                if self.confidence_threshold is None or label.score >= self.confidence_threshold:
                    occurrences[relation.first.text, relation.second.text, label.value].append(
                        (sentence_index, relation_index)
                    )

        selected_locations: list[tuple[int, int]] = [
            relation_location
            for relation_locations in tqdm(occurrences.values(), desc="Selecting Confident Relations")
            for relation_location in relation_locations
            if len(relation_locations) >= self.occurrence_threshold
        ]
        selected_locations.sort(key=operator.itemgetter(0, 1))

        for sentence_index, locations_group in itertools.groupby(
            tqdm(selected_locations, desc="Creating Selected Sentences"), key=operator.itemgetter(0)
        ):
            sentence = sentences[sentence_index]
            relations: list[Relation] = sentence.get_relations(label_type)
            selected_relations: list[Relation] = [
                relations[relation_index] for relation_index in map(operator.itemgetter(1), locations_group)
            ]
            yield self._create_selected_sentence(sentence=sentence, relations=selected_relations, label_type=label_type)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"occurrence_threshold={self.occurrence_threshold!r}, "
            f"confidence_threshold={self.confidence_threshold!r})"
        )


class PMI(SelectionStrategy):
    def select_relations(self, sentences: Sequence[Sentence], label_type: str) -> Iterator[Sentence]:
        assert label_type
        yield from sentences
