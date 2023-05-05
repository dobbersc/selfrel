from collections.abc import Callable
from typing import Union

import pytest
import ray
from flair.data import Sentence
from flair.nn import Classifier

from selfrel.predictor import Predictor


@pytest.mark.parametrize(
    "model_factory",
    [lambda: "flair/ner-english-fast", lambda: Classifier.load("flair/ner-english-fast")],
    ids=["from_path", "from_instance"],
)
def test_predictor(model_factory: Callable[[], Union[str, Classifier[Sentence]]], init_ray: None) -> None:
    sentence = Sentence("Berlin is the capital of Germany.")

    model_ref = ray.put(model_factory())
    predictor = Predictor.remote(model_ref, index=1, label_name="prediction")  # type: ignore[attr-defined]

    result: Sentence = ray.get(predictor.predict.remote(sentence))
    assert {(entity.text, entity.get_label("prediction").value) for entity in result.get_spans("prediction")} == {
        ("Berlin", "LOC"),
        ("Germany", "LOC"),
    }
