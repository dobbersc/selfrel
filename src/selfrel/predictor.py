from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Iterator, Optional, TypeVar, Union

import more_itertools
import ray
from flair.data import Sentence
from flair.nn import Classifier
from ray.actor import ActorHandle
from ray.util import ActorPool

from selfrel.serialization import from_conllu, to_conllu

__all__ = ["register_sentence_serializer", "Predictor", "initialize_predictor_pool", "buffered_map"]


T = TypeVar("T")
V = TypeVar("V")


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
        self._model: Classifier[Sentence] = Classifier.load(model_path)
        self._index = index
        self._kwargs = kwargs

    def predict(self, sentences: Union[Sentence, list[Sentence]]) -> Union[Sentence, list[Sentence]]:
        self._model.predict(sentences, **self._kwargs)
        return sentences

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self._index!r})"


def initialize_predictor_pool(
    num_actors: int, actor_options: Optional[dict[str, Any]] = None, **predictor_kwargs: Any
) -> ActorPool:
    """
    Initializes a `ray.util.ActorPool` with :class:`Predictor` actors.
    :param num_actors: The number of Ray actors in the predictor pool.
    :param actor_options: An optional dictionary of options passed to the actor.
    :param predictor_kwargs: Keyword arguments are passed to the :class:`Predictor` init.
                             The index is set automatically.
    :return: An actor pool of :class:`Predictor` actors.
    """
    predictor_kwargs.pop("index", None)
    predictors: list[ActorHandle] = [
        Predictor.options(**actor_options).remote(**predictor_kwargs, index=index)  # type: ignore[attr-defined]
        if actor_options
        else Predictor.remote(**predictor_kwargs)  # type: ignore[attr-defined]
        for index in range(num_actors)
    ]
    return ActorPool(predictors)


def buffered_map(
    actor_pool: ActorPool, fn: Callable[[ActorHandle, V], T], values: Iterable[V], buffer_size: int
) -> Iterator[T]:
    """Buffered version of `ray.util.ActorPool`'s `map` function."""
    value_buffer: Iterator[list[V]] = more_itertools.batched(values, n=buffer_size)
    for buffered_values in value_buffer:
        yield from actor_pool.map(fn, buffered_values)  # type: ignore[arg-type]
