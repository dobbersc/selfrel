from typing import NamedTuple, Optional

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
        """Returns the Flair label types from the given CoNLL-U Plus global.columns."""
        return cls(
            token_level=[column[:-6].lower() for column in global_columns if column.endswith(":TOKEN")],
            span_level=[column[:-5].lower() for column in global_columns if column.endswith(":SPAN")],
        )

    @classmethod
    def from_flair_sentence(cls, sentence: Sentence) -> Self:
        """Returns the label types from the given Flair sentence sorted alphabetically."""
        label_types: _LabelTypes = cls(token_level=[], span_level=[])

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


# noinspection PyRedundantParentheses
def _get_bioes_representation(label: str, span_length: int) -> tuple[str, ...]:
    assert label != "O"
    if span_length == 1:
        return (f"S-{label}",)
    if span_length == 2:
        return (f"B-{label}", f"E-{label}")
    return (f"B-{label}",) + tuple(f"I-{label}" for _ in range(span_length - 2)) + (f"E-{label}",)


def to_conllu(sentence: Sentence, include_global_columns: bool = True) -> str:
    """

    :param sentence:
    :param include_global_columns:
    :return:
    """
    # TODO: Serialize sentence start position

    if not len(sentence):
        raise ValueError("Can't serialize the empty sentence")

    label_types: _LabelTypes = _LabelTypes.from_flair_sentence(sentence)

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
                # TODO: Currently, the score is not serialized for sentence-level labels
                conllu_sentence.metadata[label_type] = label.value

    # Add relation metadata
    # TODO: Relations are hardcoded for now. Support variable relation label types.
    if relations:
        conllu_sentence.metadata["relations"] = "|".join(relations)

    return (
        f"# global.columns = {' '.join(label_types.as_global_columns())}\n{conllu_sentence.serialize()}"
        if include_global_columns
        else conllu_sentence.serialize()
    )
