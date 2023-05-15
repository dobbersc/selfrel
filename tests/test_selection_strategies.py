from flair.data import Relation, Sentence

from selfrel.selection_strategies import PredictionConfidence, SelectionStrategy


def test_prediction_confidence() -> None:
    sentences: list[Sentence] = [
        Sentence("Berlin is the capital of Germany."),
        Sentence("Paris is the capital of Germany."),
        Sentence("This is a sentence."),
    ]

    for sentence, score in zip(sentences[:3], (0.9, 0.6)):
        sentence[:1].add_label("ner", value="LOC")
        sentence[5:6].add_label("ner", value="LOC")
        Relation(first=sentence[:1], second=sentence[5:6]).add_label("relation", value="capital_of", score=score)

    selection_strategy: SelectionStrategy = PredictionConfidence(confidence_threshold=0.8)

    selected_sentences: list[Sentence] = list(selection_strategy.select_relations(sentences, label_type="relation"))
    assert len(selected_sentences) == 1

    selected_sentence: Sentence = selected_sentences[0]

    assert selected_sentence is not sentences[0]
    assert selected_sentence.to_original_text() == "Berlin is the capital of Germany."
    assert {(span.text, span.get_label("ner").value) for span in selected_sentence.get_spans("ner")} == {
        ("Berlin", "LOC"),
        ("Germany", "LOC"),
    }
    assert {
        (
            relation.first.text,
            relation.second.text,
            relation.get_label("relation").value,
            relation.get_label("relation").score,
        )
        for relation in selected_sentence.get_relations("relation")
    } == {("Berlin", "Germany", "capital_of", 0.9)}
