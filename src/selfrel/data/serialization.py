import bisect
from collections.abc import Set
from typing import Any, NamedTuple, Optional

import conllu
from flair.data import (
    DataPoint,
    Label,
    Relation,
    Sentence,
    Span,
    Token,
    get_spans_from_bio,
)
from typing_extensions import Self

__all__ = ["to_conllu", "from_conllu"]


# The reserved metadata specifies CoNLL-U metadata keys reserved for special cases.
# They are not valid for custom sentence-level label annotations.
__RESERVED_METADATA: set[str] = {"global.columns", "text", "relations"}


class _LabelTypes(NamedTuple):
    token_level: list[str]
    span_level: list[str]

    def as_global_columns(self) -> list[str]:
        """
        Returns the labele types' CoNLL-U Plus global.columns as list of columns.

        Order:
            1. ID
            2. FORM
            3. LABEL_TYPE_1:TOKEN, ..., LABEL_TYPE_N:TOKEN
            4. LABEL_TYPE_1:SPAN, ..., LABEL_TYPE_N:SPAN
            5. LABEL_TYPE_1:TOKEN_SCORE, ..., LABEL_TYPE_N:TOKEN_SCORE
            6. LABEL_TYPE_1:SPAN_SCORE, ..., LABEL_TYPE_N:SPAN_SCORE
            7. MISC
        """
        global_columns: list[str] = ["ID", "FORM"]
        global_columns.extend(f"{label_type.upper()}:TOKEN" for label_type in self.token_level)
        global_columns.extend(f"{label_type.upper()}:SPAN" for label_type in self.span_level)
        global_columns.extend(f"{label_type.upper()}:TOKEN_SCORE" for label_type in self.token_level)
        global_columns.extend(f"{label_type.upper()}:SPAN_SCORE" for label_type in self.span_level)
        global_columns.append("MISC")
        return global_columns

    @classmethod
    def from_global_columns(cls, global_columns: list[str]) -> Self:
        """Returns the Flair label-types from the given CoNLL-U Plus global.columns."""
        return cls(
            token_level=[column[:-6].lower() for column in global_columns if column.endswith(":TOKEN")],
            span_level=[column[:-5].lower() for column in global_columns if column.endswith(":SPAN")],
        )

    @classmethod
    def from_flair_sentence(cls, sentence: Sentence) -> Self:
        """Returns the label-types from the given Flair sentence sorted alphabetically."""
        label_types: Self = cls(token_level=[], span_level=[])

        label_type: str
        labels: list[Label]
        for label_type, labels in sentence.annotation_layers.items():
            data_point: DataPoint = labels[0].data_point
            if isinstance(data_point, Token):
                label_types.token_level.append(label_type)
            elif isinstance(data_point, Span):
                label_types.span_level.append(label_type)

        label_types.token_level.sort()
        label_types.span_level.sort()

        return label_types


def _add_default_token_fields(label_types: _LabelTypes, default_token_fields: Set[str]) -> None:
    """
    Includes the default token fields (label-types) that are not annotated in the sentence
    i.e. not present in the given `label_types`. The label-types remain alphabetically sorted.
    """
    token_fields: set[str] = set(label_types.token_level)
    for default_token_field in default_token_fields:
        if default_token_field not in token_fields:
            bisect.insort(label_types.token_level, default_token_field)


def _add_default_span_fields(label_types: _LabelTypes, default_span_fields: Set[str]) -> None:
    """
    Includes the default span fields (label-types) that are not annotated in the sentence
    i.e. not present in the given `label_types`. The label-types remain alphabetically sorted.
    """
    span_fields: set[str] = set(label_types.span_level)
    for default_span_field in default_span_fields:
        if default_span_field not in span_fields:
            bisect.insort(label_types.span_level, default_span_field)


# noinspection PyRedundantParentheses
def _get_bioes_representation(label: str, span_length: int) -> tuple[str, ...]:
    assert label != "O"
    if span_length == 1:
        return (f"S-{label}",)
    if span_length == 2:
        return (f"B-{label}", f"E-{label}")
    return (f"B-{label}",) + tuple(f"I-{label}" for _ in range(span_length - 2)) + (f"E-{label}",)


def to_conllu(
    sentence: Sentence,
    include_global_columns: bool = True,
    default_token_fields: Set[str] = frozenset(),
    default_span_fields: Set[str] = frozenset(),
) -> str:
    """
    Serializes a Flair sentence to CoNLL-U (Plus).

    # TODO: to_conllu does not serialize
    #   - the sentence start position.
    #   - the sentence's label scores.
    #   - variable relation label-types. Relations are hardcoded as "relations".
    #   - label-type is not case-preserved

    :param sentence: The sentence to serialize
    :param include_global_columns: If True, the CoNLL-U Plus global.columns header is included in the serialization.
    :param default_token_fields: An optional set of token-level label-type strings to include in the serialization
                                 regardless if they are annotated in the input sentence.
    :param default_span_fields: An optional set of span-level label-type strings to include in the serialization
                                regardless if they are annotated in the input sentence.
    :return: The serialized sentence as CoNLL-U (Plus) string
    """
    if not len(sentence):
        raise ValueError("Can't serialize the empty sentence")

    label_types: _LabelTypes = _LabelTypes.from_flair_sentence(sentence)
    if default_token_fields:
        _add_default_token_fields(label_types, default_token_fields)
    if default_span_fields:
        _add_default_span_fields(label_types, default_span_fields)

    conllu_sentence: conllu.TokenList = conllu.TokenList(
        [
            conllu.Token(
                # Important: Has to be the same order as the global.columns
                id=token_index,
                form=token.form,
                **{f"{label_type}:token": "O" for label_type in label_types.token_level},
                **{f"{label_type}:span": "O" for label_type in label_types.span_level},
                **{f"{label_type}:token_score": "_" for label_type in label_types.token_level},
                **{f"{label_type}:span_score": "_" for label_type in label_types.span_level},
                misc="_" if token.whitespace_after else "SpaceAfter=No",
            )
            for token_index, token in enumerate(sentence.tokens, start=1)
        ],
        metadata=conllu.Metadata(text=sentence.to_original_text()),
    )

    # Add annotations (token-level, span-level, relation-level, sentence-level)
    label_type: str
    labels: list[Label]
    relations: list[str] = []
    for label_type, labels in sentence.annotation_layers.items():
        for label in labels:
            data_point: DataPoint = label.data_point

            if isinstance(data_point, Token):
                conllu_token: conllu.Token = conllu_sentence[data_point.idx - 1]
                conllu_token[f"{label_type}:token"] = label.value
                conllu_token[f"{label_type}:token_score"] = label.score

            elif isinstance(data_point, Span):
                conllu_span: list[conllu.Token] = conllu_sentence[data_point[0].idx - 1 : data_point[-1].idx]
                for token, bioes_label in zip(conllu_span, _get_bioes_representation(label.value, len(conllu_span))):
                    token[f"{label_type}:span"] = bioes_label
                    token[f"{label_type}:span_score"] = label.score

            elif isinstance(data_point, Relation):
                head: str = f"{data_point.first[0].idx};{data_point.first[-1].idx}"
                tail: str = f"{data_point.second[0].idx};{data_point.second[-1].idx}"
                relations.append(f"{head};{tail};{label.value};{label.score}")

            elif isinstance(data_point, Sentence):
                if label_type in __RESERVED_METADATA:
                    raise ValueError(f"Unsupported sentence annotation of label-type {label_type!r}")
                conllu_sentence.metadata[label_type] = label.value

    # Add relation metadata
    if relations:
        conllu_sentence.metadata["relations"] = "|".join(relations)

    return (
        f"# global.columns = {' '.join(label_types.as_global_columns())}\n{conllu_sentence.serialize()}"
        if include_global_columns
        else conllu_sentence.serialize()
    )


def _infer_whitespace_after(token: conllu.Token) -> int:
    misc: Optional[dict[str, str]] = token.get("misc")
    if misc is not None:
        return 0 if misc.get("SpaceAfter") == "No" else 1
    return 1


def _score_as_float(score: str) -> float:
    try:
        return float(score)
    except ValueError:
        return 1.0


def from_conllu(serialized: str, **kwargs: Any) -> Sentence:
    """
    Creates a Flair sentence from a CoNLL-U (Plus) serialized sentence.
    :param serialized: The CoNLL-U (Plus) serialized sentence.
    :param kwargs: Keywords arguments are passed to `conllu.parse`
    :return: The deserialized Flair sentence
    """
    raw_global_columns: str = serialized.split("\n", 1)[0]
    if not raw_global_columns.startswith("# global.columns = "):
        raise ValueError("Missing CoNLL-U Plus required 'global.columns'")

    # Parse global columns and gather annotated label-types
    global_columns: list[str] = raw_global_columns[19:].split()
    label_types: _LabelTypes = _LabelTypes.from_global_columns(global_columns)

    # Parse serialized sentence
    conllu_sentences: conllu.SentenceList = conllu.parse(serialized, **kwargs)
    if len(conllu_sentences) != 1:
        raise ValueError("Received multiple sentences but expected single serialized CoNLL-U Plus sentence")
    conllu_sentence: conllu.TokenList = conllu_sentences[0]

    # Initialize Flair sentence
    flair_tokens: list[Token] = [
        Token(
            text=conllu_token["form"],
            whitespace_after=_infer_whitespace_after(conllu_token),
        )
        for conllu_token in conllu_sentence
    ]
    flair_sentence: Sentence = Sentence(flair_tokens)

    # Add token labels
    for token_label_type in label_types.token_level:
        for flair_token, conllu_token in zip(flair_sentence.tokens, conllu_sentence):
            label: str = conllu_token[f"{token_label_type}:token"]
            score: float = _score_as_float(conllu_token[f"{token_label_type}:token_score"])
            if label not in ["O", "_"]:
                flair_token.add_label(token_label_type, value=label, score=score)

    # Add span labels
    for span_label_type in label_types.span_level:
        bioes_tags: list[str] = []
        bioes_scores: list[float] = []
        for token in conllu_sentence:
            bioes_tags.append(token[f"{span_label_type}:span"])
            bioes_scores.append(_score_as_float(token[f"{span_label_type}:span_score"]))

        for span, score, label in get_spans_from_bio(bioes_tags=bioes_tags, bioes_scores=bioes_scores):
            flair_sentence[span[0] : span[-1] + 1].add_label(span_label_type, value=label, score=score)

    # Add relation labels
    if "relations" in conllu_sentence.metadata:
        for relation in conllu_sentence.metadata["relations"].split("|"):
            sections: list[str] = relation.split(";")
            head_start: int = int(sections[0])
            head_end: int = int(sections[1])
            tail_start: int = int(sections[2])
            tail_end: int = int(sections[3])
            label = sections[4]
            score = float(sections[5])

            relation = Relation(
                first=flair_sentence[head_start - 1 : head_end], second=flair_sentence[tail_start - 1 : tail_end]
            )
            relation.add_label("relation", value=label, score=score)

    # Add metadata as sentence label
    for key, value in conllu_sentence.metadata.items():
        if key in __RESERVED_METADATA:
            continue
        flair_sentence.add_label(key, value)

    return flair_sentence
