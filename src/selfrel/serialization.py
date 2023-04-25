import conllu
from flair.data import DataPoint, Label, Relation, Sentence, Span, Token


# noinspection PyRedundantParentheses
def _get_bioes_representation(label: str, span_length: int) -> tuple[str, ...]:
    assert label != "O"
    if span_length == 1:
        return (f"S-{label}",)
    if span_length == 2:
        return (f"B-{label}", f"E-{label}")
    return (f"B-{label}",) + tuple(f"I-{label}" for _ in range(span_length - 2)) + (f"E-{label}",)


def _add_token_label(token: conllu.Token, label_type: str, label: str, score: float) -> None:
    token[f"{label_type}:token"] = label
    token[f"{label_type}:token_score"] = score


def _add_span_label(span: list[conllu.Token], label_type: str, label: str, score: float) -> None:
    for token, bioes_label in zip(span, _get_bioes_representation(label, len(span))):
        token[f"{label_type}:span"] = label
        token[f"{label_type}:span_score"] = score


def _get_annotation_abstractions(sentence: Sentence) -> dict[str, list[str]]:
    annotation_abstractions: dict[str, list[str]] = {"token": [], "span": []}
    label_type: str
    labels: list[Label]
    for label_type, labels in sentence.annotation_layers.items():
        data_point: DataPoint = labels[0].data_point
        if isinstance(data_point, Token):
            annotation_abstractions["token"].append(label_type)
        elif isinstance(data_point, Span):
            annotation_abstractions["span"].append(label_type)

    annotation_abstractions["token"].sort()
    annotation_abstractions["span"].sort()

    return annotation_abstractions


def to_conllu(sentence: Sentence, include_global_columns: bool = True) -> str:
    """

    :param sentence:
    :param include_global_columns:
    :return:
    """
    # TODO: Serialize sentence start position

    if not len(sentence):
        raise ValueError("Can't serialize the empty sentence")

    annotation_abstractions: dict[str, list[str]] = _get_annotation_abstractions(sentence)

    conllu_sentence: conllu.TokenList = conllu.TokenList(
        [
            conllu.Token(
                id=token_index,
                form=token.form,
                **{f"{label_type}:token": "O" for label_type in annotation_abstractions["token"]},
                **{f"{label_type}:span": "O" for label_type in annotation_abstractions["span"]},
                **{f"{label_type}:token_score": "_" for label_type in annotation_abstractions["token"]},
                **{f"{label_type}:span_score": "_" for label_type in annotation_abstractions["span"]},
                misc="_" if token.whitespace_after else "SpaceAfter=No",
            )
            for token_index, token in enumerate(sentence.tokens, start=1)
        ],
        metadata=conllu.Metadata(text=sentence.to_original_text()),
    )

    global_columns: list[str] = ["ID", "FORM"]
    global_columns.extend(f"{label_type.upper()}:TOKEN" for label_type in annotation_abstractions["token"])
    global_columns.extend(f"{label_type.upper()}:SPAN" for label_type in annotation_abstractions["span"])
    global_columns.extend(f"{label_type.upper()}:TOKEN_SCORE" for label_type in annotation_abstractions["token"])
    global_columns.extend(f"{label_type.upper()}:SPAN_SCORE" for label_type in annotation_abstractions["span"])
    global_columns.append("MISC")

    # Add annotations (token-level, span-level, relation-level, sentence-level)
    label_type: str
    labels: list[Label]
    relations: list[str] = []
    for label_type, labels in sentence.annotation_layers.items():
        for label in labels:
            data_point: DataPoint = label.data_point

            if isinstance(data_point, Token):
                conllu_token: conllu.Token = conllu_sentence[data_point.idx - 1]
                _add_token_label(conllu_token, label_type, label.value, label.score)

            elif isinstance(data_point, Span):
                conllu_span: list[conllu.Token] = conllu_sentence[data_point[0].idx - 1 : data_point[-1].idx]
                _add_span_label(conllu_span, label_type, label.value, label.score)

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
        f"# global.columns = {' '.join(global_columns)}\n{conllu_sentence.serialize()}"
        if include_global_columns
        else conllu_sentence.serialize()
    )
