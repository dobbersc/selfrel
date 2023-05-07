from collections.abc import Callable
from typing import Union

import pytest
import ray
from flair.data import Sentence
from flair.nn import Classifier
from ray.actor import ActorHandle

from selfrel.predictor import Predictor, buffered_map, initialize_predictor_pool


@pytest.mark.usefixtures("_init_ray")
@pytest.mark.parametrize(
    "model_factory",
    [lambda: "flair/ner-english-fast", lambda: Classifier.load("flair/ner-english-fast")],
    ids=["from_path", "from_instance"],
)
def test_predictor(model_factory: Callable[[], Union[str, Classifier[Sentence]]]) -> None:
    sentence = Sentence("Berlin is the capital of Germany.")

    model_ref = ray.put(model_factory())
    predictor = Predictor.remote(model_ref, index=1, label_name="prediction")  # type: ignore[attr-defined]

    result: Sentence = ray.get(predictor.predict.remote(sentence))
    assert {(entity.text, entity.get_label("prediction").value) for entity in result.get_spans("prediction")} == {
        ("Berlin", "LOC"),
        ("Germany", "LOC"),
    }


@pytest.mark.usefixtures("_init_ray")
@pytest.mark.parametrize(
    "model_factory",
    [lambda: "flair/ner-english-fast", lambda: Classifier.load("flair/ner-english-fast")],
    ids=["from_path", "from_instance"],
)
def test_predictor_pool(model_factory: Callable[[], Union[str, Classifier[Sentence]]]) -> None:
    sentence = Sentence("Berlin is the capital of Germany.")

    # Set-up predictor pool
    model_ref = ray.put(model_factory())
    predictor_pool = initialize_predictor_pool(2, model=model_ref, label_name="prediction")

    def remote_predict(pipeline_actor: ActorHandle, sentence_: Sentence) -> Sentence:
        return pipeline_actor.predict.remote(sentence_)  # type: ignore[no-any-return]

    # Test buffered_map
    result_sentences = list(buffered_map(predictor_pool, fn=remote_predict, values=[sentence], buffer_size=1))
    assert len(result_sentences) == 1
    result_sentence = result_sentences[0]

    expected: set[tuple[str, str]] = {
        (entity.text, entity.get_label("prediction").value) for entity in result_sentence.get_spans("prediction")
    }
    assert expected == {("Berlin", "LOC"), ("Germany", "LOC")}