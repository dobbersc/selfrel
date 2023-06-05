import pandas as pd
from flair.data import Sentence

from selfrel.utils.inspect_relations import build_relation_overview


def test_build_relation_overview(sentences_with_relation_annotations: list[Sentence]) -> None:
    sentences: list[Sentence] = sentences_with_relation_annotations
    result: pd.DataFrame = build_relation_overview(sentences, entity_label_types="ner", relation_label_type="relation")

    index: pd.MultiIndex = pd.MultiIndex.from_arrays(
        ((0, 1, 2, 2, 3, 3, 4), (0, 0, 0, 1, 0, 1, 0)),
        names=("sentence_index", "relation_index"),
    )
    expected: pd.DataFrame = pd.DataFrame(
        {
            "sentence_text": pd.Series(
                (
                    "Berlin is the capital of Germany.",
                    "Berlin is the capital of Germany.",
                    "Albert Einstein was born in Ulm, Germany.",
                    "Albert Einstein was born in Ulm, Germany.",
                    "Ulm, located in Germany, is the birthplace of Albert Einstein.",
                    "Ulm, located in Germany, is the birthplace of Albert Einstein.",
                    "Amazon was founded by Jeff Bezos.",
                ),
                index=index,
                dtype="string",
            ),
            "head_text": pd.Series(
                ("Berlin", "Berlin", "Albert Einstein", "Ulm", "Albert Einstein", "Ulm", "Amazon"),
                index=index,
                dtype="string",
            ),
            "tail_text": pd.Series(
                ("Germany", "Germany", "Ulm", "Germany", "Ulm", "Germany", "Jeff Bezos"), index=index, dtype="string"
            ),
            "head_label": pd.Series(("LOC", "LOC", "PER", "LOC", "PER", "LOC", "ORG"), index=index, dtype="category"),
            "tail_label": pd.Series(("LOC", "LOC", "LOC", "LOC", "LOC", "LOC", "PER"), index=index, dtype="category"),
            "label": pd.Series(
                ("capital_of", "capital_of", "born_in", "located_in", "born_in", "located_in", "founded_by"),
                index=index,
                dtype="category",
            ),
            "confidence": (1.0,) * 7,
        },
        index=index,
    )

    pd.testing.assert_frame_equal(result, expected)
