from pathlib import Path
from typing import Union, Any

import ray
from flair.data import Sentence
from flair.nn import Classifier

from selfrel.serialization import to_conllu, from_conllu


def _register_sentence_serializer() -> None:
    ray.util.register_serializer(Sentence, serializer=to_conllu, deserializer=from_conllu)


_register_sentence_serializer()


@ray.remote
class Predictor:
    def __init__(self, model_path: Union[str, Path], **kwargs: Any) -> None:
        _register_sentence_serializer()

        self._model_path: Union[str, Path] = model_path
        self._model: Classifier = Classifier.load(model_path)
        self._kwargs = kwargs

    def predict(self, sentences: Union[Sentence, list[Sentence]]) -> Union[Sentence, list[Sentence]]:
        self._model.predict(sentences, **self._kwargs)
        return sentences
