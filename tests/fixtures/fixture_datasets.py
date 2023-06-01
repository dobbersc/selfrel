import pytest
from flair.data import Relation, Sentence
from flair.tokenization import SpaceTokenizer


@pytest.fixture()
def sentences_with_relation_annotations() -> list[Sentence]:
    """List of general sentence with relation annotations."""
    sentences: list[Sentence] = [
        Sentence("Berlin is the capital of Germany."),
        Sentence("Berlin is the capital of Germany."),
        Sentence("Albert Einstein was born in Ulm, Germany."),
        Sentence("Ulm, located in Germany, is the birthplace of Albert Einstein."),
        Sentence("Amazon was founded by Jeff Bezos."),
        Sentence("This is a sentence."),
    ]
    sentence: Sentence

    # Annotate "Berlin is the capital of Germany."
    for sentence in sentences[:2]:
        sentence[:1].add_label("ner", value="LOC")
        sentence[5:6].add_label("ner", value="LOC")
        Relation(first=sentence[:1], second=sentence[5:6]).add_label("relation", value="capital_of")

    # Annotate "Albert Einstein was born in Ulm, Germany."
    sentence = sentences[2]
    sentence[:2].add_label("ner", value="PER")
    sentence[5:6].add_label("ner", value="LOC")
    sentence[7:8].add_label("ner", value="LOC")
    Relation(first=sentence[:2], second=sentence[5:6]).add_label("relation", value="born_in")
    Relation(first=sentence[5:6], second=sentence[7:8]).add_label("relation", value="located_in")

    # Annotate "Ulm, located in Germany, is the birthplace of Albert Einstein."
    sentence = sentences[3]
    sentence[:1].add_label("ner", value="LOC")
    sentence[4:5].add_label("ner", value="LOC")
    sentence[10:12].add_label("ner", value="PER")
    Relation(first=sentence[10:12], second=sentence[:1]).add_label("relation", value="born_in")
    Relation(first=sentence[:1], second=sentence[4:5]).add_label("relation", value="located_in")

    # Annotate "Amazon was founded by Jeff Bezos."
    sentence = sentences[4]
    sentence[:1].add_label("ner", value="ORG")
    sentence[4:6].add_label("ner", value="PER")
    Relation(first=sentence[:1], second=sentence[4:6]).add_label("relation", value="founded_by")

    return sentences


@pytest.fixture()
def prediction_confidence_sentences() -> list[Sentence]:
    """List of sentences with relation annotations of different confidences."""
    sentences: list[Sentence] = [
        Sentence("Berlin located in Germany and Hamburg located in Germany."),
        Sentence("Berlin located in Germany."),
        Sentence("This is a sentence."),
    ]

    sentence: Sentence = sentences[0]
    for span in (sentence[:1], sentence[3:4], sentence[5:6], sentence[8:9]):
        span.add_label("ner", value="LOC")
    Relation(first=sentence[:1], second=sentence[3:4]).add_label("relation", value="located_in", score=0.9)
    Relation(first=sentence[5:6], second=sentence[8:9]).add_label("relation", value="located_in", score=0.9)

    sentence = sentences[1]
    sentence[:1].add_label("ner", value="LOC")
    sentence[3:4].add_label("ner", value="LOC")
    Relation(first=sentence[:1], second=sentence[3:4]).add_label("relation", value="located_in", score=0.6)

    return sentences


@pytest.fixture()
def entropy_sentences() -> list[Sentence]:
    """
    List of sentences with relation annotations useful for testing entropy calculations.
    In total, the sentences contain the following relations:

    | Relation Candidate | Relation    | Occurrence |
    |--------------------|-------------|------------|
    | Berlin -> Germany  | located_in  | 8          |
    |                    | born_in     | 0          |
    |                    | no_relation | 2          |
    |                    |             |            |
    | Obama  -> USA      | located_in  | 0          |
    |                    | born_in     | 4          |
    |                    | no_relation | 6          |

    """
    sentences: list[Sentence] = [
        *[Sentence("Berlin located_in Germany", use_tokenizer=SpaceTokenizer()) for _ in range(8)],
        *[Sentence("Berlin no_relation Germany", use_tokenizer=SpaceTokenizer()) for _ in range(2)],
        *[Sentence("Obama born_in USA", use_tokenizer=SpaceTokenizer()) for _ in range(4)],
        *[Sentence("Obama no_relation USA", use_tokenizer=SpaceTokenizer()) for _ in range(6)],
    ]

    for sentence in sentences:
        head, tail = sentence[:1], sentence[2:3]
        head.add_label("ner", value="HEAD")
        tail.add_label("ner", value="TAIL")
        Relation(first=head, second=tail).add_label("relation", value=sentence[1].text)

    return sentences
