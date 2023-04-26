from pathlib import Path
from typing import Any, Optional, Union

import ray
from flair.data import Sentence
from flair.nn import Classifier

from selfrel.serialization import from_conllu, to_conllu


__all__ = ["register_sentence_serializer", "Predictor"]


def register_sentence_serializer() -> None:
    ray.util.register_serializer(Sentence, serializer=to_conllu, deserializer=from_conllu)


register_sentence_serializer()


@ray.remote
class Predictor:
    def __init__(self, model_path: Union[str, Path], index: Optional[int] = None, **kwargs: Any) -> None:
        register_sentence_serializer()

        if index is None:
            print(f"Loading Flair model {model_path!r}")
        else:
            print(f"Loading Flair model {model_path!r} at index {index!r}")

        self._model_path: Union[str, Path] = model_path
        self._model: Classifier = Classifier.load(model_path)
        self._index = index
        self._kwargs = kwargs

    def predict(self, sentences: Union[Sentence, list[Sentence]]) -> Union[Sentence, list[Sentence]]:
        self._model.predict(sentences, **self._kwargs)
        return sentences

    def __repr__(self) -> str:
        if self._index is None:
            return f"{type(self).__name__}(model={self._model_path!r})"
        return f"{type(self).__name__}(index={self._index!r}, model={self._model_path!r})"
