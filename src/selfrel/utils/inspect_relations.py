from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Optional, Union

import pandas as pd
from flair.data import Label, Relation, Sentence
from tqdm import tqdm

__all__ = ["inspect_relations", "infer_entity_pair_labels", "build_relation_overview"]


def inspect_relations(
    sentences: Iterable[Sentence],
    relation_label_type: str,
    entity_label_types: Optional[Union[set[Optional[str]], str]] = None,
) -> defaultdict[str, Counter[tuple[str, str]]]:
    if not isinstance(entity_label_types, set):
        entity_label_types = {entity_label_types}

    # Dictionary of [<relation label>, <counter of relation entity labels (HEAD, TAIL)>]
    relations: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    for sentence in sentences:
        for relation in sentence.get_relations(relation_label_type):
            head_label: str = next(relation.first.get_label(label_type).value for label_type in entity_label_types)
            tail_label: str = next(relation.second.get_label(label_type).value for label_type in entity_label_types)

            entity_counter = relations[relation.get_label(relation_label_type).value]
            entity_counter.update([(head_label, tail_label)])

    return relations


def infer_entity_pair_labels(
    sentences: Iterable[Sentence],
    relation_label_type: str,
    entity_label_types: Optional[Union[set[Optional[str]], str]] = None,
) -> set[tuple[str, str]]:
    relations: defaultdict[str, Counter[tuple[str, str]]] = inspect_relations(
        sentences, relation_label_type=relation_label_type, entity_label_types=entity_label_types
    )
    return {
        entity_pair_labels for entity_pair_counter in relations.values() for entity_pair_labels in entity_pair_counter
    }


def build_relation_overview(
    sentences: Iterable[Sentence],
    entity_label_types: Optional[Union[set[Optional[str]], str]],
    relation_label_type: str,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    if not isinstance(entity_label_types, set):
        entity_label_types = {entity_label_types}

    sentence_indices: list[int] = []
    relation_indices: list[int] = []
    sentence_texts: list[str] = []
    head_texts: list[str] = []
    tail_texts: list[str] = []
    head_labels: list[str] = []
    tail_labels: list[str] = []
    labels: list[str] = []
    confidences: list[float] = []
    head_start_positions: list[int] = []
    head_end_positions: list[int] = []
    tail_start_positions: list[int] = []
    tail_end_positions: list[int] = []

    for sentence_index, sentence in enumerate(
        tqdm(sentences, desc="Building Relation Overview", disable=not show_progress_bar)
    ):
        sentence_text: str = sentence.to_original_text()

        relation: Relation
        for relation_index, relation in enumerate(sentence.get_relations(relation_label_type)):
            sentence_indices.append(sentence_index)
            relation_indices.append(relation_index)

            sentence_texts.append(sentence_text)

            relation_label: Label = relation.get_label(relation_label_type)
            labels.append(relation_label.value)
            confidences.append(relation_label.score)

            head_texts.append(relation.first.text)
            tail_texts.append(relation.second.text)

            head_label: str = next(relation.first.get_label(label_type).value for label_type in entity_label_types)
            tail_label: str = next(relation.second.get_label(label_type).value for label_type in entity_label_types)
            head_labels.append(head_label)
            tail_labels.append(tail_label)

            head_start_positions.append(relation.first.start_position)
            head_end_positions.append(relation.first.end_position)
            tail_start_positions.append(relation.second.start_position)
            tail_end_positions.append(relation.second.end_position)

    index: pd.MultiIndex = pd.MultiIndex.from_arrays(
        (sentence_indices, relation_indices),
        names=("sentence_index", "relation_index"),
    )
    return pd.DataFrame(
        {
            "sentence_text": pd.Series(sentence_texts, index=index, dtype="string"),
            "head_text": pd.Series(head_texts, index=index, dtype="string"),
            "tail_text": pd.Series(tail_texts, index=index, dtype="string"),
            "head_label": pd.Series(head_labels, index=index, dtype="category"),
            "tail_label": pd.Series(tail_labels, index=index, dtype="category"),
            "label": pd.Series(labels, index=index, dtype="category"),
            "confidence": confidences,
            "head_start_position": head_start_positions,
            "head_end_position": head_end_positions,
            "tail_start_position": tail_start_positions,
            "tail_end_position": tail_end_positions,
        },
        index=index,
    )
