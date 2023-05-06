import pytest
from flair.data import Relation, Sentence

from selfrel.data import from_conllu, to_conllu


def test_serialize_deserialize_unannotated_sentence() -> None:
    sentence: Sentence = Sentence(
        "Albert Einstein, who was born in Ulm, Germany, later emigrated to the USA.",
        start_position=10,
    )

    # Serialize -> Deserialize
    parsed_sentence: Sentence = from_conllu(to_conllu(sentence))

    # Validate sentence text and start position
    assert sentence.to_original_text() == parsed_sentence.to_original_text()
    assert sentence.start_position != parsed_sentence.start_position  # Currently, not serialized


def test_serialize_deserialize_annotated_sentence() -> None:
    sentence: Sentence = Sentence(
        "Albert Einstein, who was born in Ulm, Germany, later emigrated to the United States of America.",
        start_position=10,
    )

    # Add sentence annotations
    sentence.add_label("sentence_id", value="1")
    sentence.add_label("sentiment", value="neutral", score=0.75)

    # Add token annotations
    sentence[0].add_label("upos", value="PROPN", score=0.75)
    sentence[1].add_label("upos", value="PROPN", score=0.75)
    sentence[2].add_label("upos", value="PUNCT", score=0.75)

    # Add span annotations
    sentence[:2].add_label("ner", value="PER", score=0.75)
    sentence[7:8].add_label("ner", value="LOC", score=0.75)
    sentence[15:19].add_label("ner", value="LOC", score=0.75)

    # Add relation annotations
    Relation(first=sentence[:2], second=sentence[7:8]).add_label("relation", value="born_in")
    Relation(first=sentence[7:8], second=sentence[9:10]).add_label("relation", value="located_in")

    # Serialize -> Deserialize
    parsed_sentence: Sentence = from_conllu(to_conllu(sentence))

    # Validate sentence text and start position
    assert sentence.to_original_text() == parsed_sentence.to_original_text()
    assert sentence.start_position != parsed_sentence.start_position  # Currently, not serialized

    # Validate sentence annotations
    for label_type in ["sentence_id", "sentiment"]:
        assert sentence.get_label(label_type).value == parsed_sentence.get_label(label_type).value
        # Currently, not serialized
        # assert sentence.get_label(label_type).score == parsed_sentence.get_label(label_type).score

    # Validate token annotations
    for token, parsed_token in zip(sentence.tokens, parsed_sentence.tokens):
        assert token.get_label("upos").value == parsed_token.get_label("upos").value
        assert token.get_label("upos").score == parsed_token.get_label("upos").score

    # Validate span annotations
    for span, parsed_span in zip(sentence.get_spans("ner"), parsed_sentence.get_spans("ner")):
        assert span.text == parsed_span.text
        assert span.get_label("ner").value == parsed_span.get_label("ner").value
        assert span.get_label("ner").score == parsed_span.get_label("ner").score

    # Validate relation annotations
    for relation, parsed_relation in zip(sentence.get_relations("relation"), parsed_sentence.get_relations("relation")):
        assert relation.text == parsed_relation.text
        assert relation.get_label("relation").value == parsed_relation.get_label("relation").value
        assert relation.get_label("relation").score == parsed_relation.get_label("relation").score


def test_serialize_sentence_with_default_token_fields() -> None:
    sentence: Sentence = Sentence("This is a sentence.")
    serialized: str = to_conllu(sentence, default_token_fields={"pos"})
    assert serialized == (
        "# global.columns = ID FORM POS:TOKEN POS:TOKEN_SCORE MISC\n"
        "# text = This is a sentence.\n"
        "1\tThis\tO\t_\t_\n"
        "2\tis\tO\t_\t_\n"
        "3\ta\tO\t_\t_\n"
        "4\tsentence\tO\t_\tSpaceAfter=No\n"
        "5\t.\tO\t_\tSpaceAfter=No\n\n"
    )


def test_serialize_sentence_with_default_span_fields() -> None:
    sentence: Sentence = Sentence("This is a sentence.")
    serialized: str = to_conllu(sentence, default_span_fields={"ner"})
    assert serialized == (
        "# global.columns = ID FORM NER:SPAN NER:SPAN_SCORE MISC\n"
        "# text = This is a sentence.\n"
        "1\tThis\tO\t_\t_\n"
        "2\tis\tO\t_\t_\n"
        "3\ta\tO\t_\t_\n"
        "4\tsentence\tO\t_\tSpaceAfter=No\n"
        "5\t.\tO\t_\tSpaceAfter=No\n\n"
    )


def test_serialize_empty_sentence() -> None:
    sentence: Sentence = Sentence("")
    with pytest.raises(ValueError, match=r"Can't serialize the empty sentence"):
        to_conllu(sentence)


@pytest.mark.parametrize("reserved_label_type", ["global.columns", "text", "relations"])
def test_serialize_sentence_with_reserved_annotations(reserved_label_type: str) -> None:
    sentence: Sentence = Sentence("This is a sentence.")
    sentence.add_label(reserved_label_type, value="test")
    with pytest.raises(ValueError, match=rf"Unsupported sentence annotation of label-type {reserved_label_type!r}"):
        to_conllu(sentence)


def test_deserialize_empty_string() -> None:
    with pytest.raises(ValueError, match=r"Missing CoNLL-U Plus required 'global.columns'"):
        from_conllu("")


def test_deserialize_sentence_with_missing_global_columns() -> None:
    with pytest.raises(ValueError, match=r"Missing CoNLL-U Plus required 'global.columns'"):
        from_conllu(
            "# text = This is a sentence.\n"
            "1\tThis\t_\n"
            "2\tis\t_\n"
            "3\ta\t_\n"
            "4\tsentence\tSpaceAfter=No\n"
            "5\t.\tSpaceAfter=No\n\n",
        )


def test_deserialize_multiple_sentences() -> None:
    with pytest.raises(
        ValueError,
        match=r"Received multiple sentences but expected single serialized CoNLL-U Plus sentence",
    ):
        from_conllu(
            "# global.columns = ID FORM MISC\n"
            "# text = This is a sentence.\n"
            "1\tThis\t_\n"
            "2\tis\t_\n"
            "3\ta\t_\n"
            "4\tsentence\tSpaceAfter=No\n"
            "5\t.\tSpaceAfter=No\n\n"
            "# text = This is another sentence.\n"
            "1\tThis\t_\n"
            "2\tis\t_\n"
            "3\tanother\t_\n"
            "4\tsentence\tSpaceAfter=No\n"
            "5\t.\tSpaceAfter=No\n\n",
        )
