from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Optional, Union

from flair.data import Sentence

__all__ = ["inspect_relations", "infer_entity_pair_labels"]


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
