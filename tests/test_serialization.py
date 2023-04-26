from flair.data import Sentence, Relation

from selfrel.serialization import to_conllu, from_conllu


def test_unannotated_sentence_serialization() -> None:
    sentence: Sentence = Sentence(
        "Albert Einstein, who was born in Ulm, Germany, later emigrated to the USA.", start_position=10
    )

    # Serialize -> Deserialize
    parsed_sentence: Sentence = from_conllu(to_conllu(sentence))

    # Validate sentence text and start position
    assert sentence.to_original_text() == parsed_sentence.to_original_text()
    assert sentence.start_position != parsed_sentence.start_position  # Currently, not serialized


def test_annotated_sentence_serialization() -> None:
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

    print(sentence)

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
