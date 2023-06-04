import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union, overload

import more_itertools
import ray
from flair.data import Sentence
from flair.nn import Classifier
from ray.actor import ActorHandle
from ray.util import ActorPool

from selfrel.data import from_conllu, to_conllu

__all__ = [
    "Predictor",
    "PredictorPool",
    "register_sentence_serializer",
    "register_classifier_serializers",
    "buffered_map",
]


T = TypeVar("T")
V = TypeVar("V")

SentenceT = TypeVar("SentenceT", bound=Sentence)


def register_sentence_serializer() -> None:
    ray.util.register_serializer(Sentence, serializer=to_conllu, deserializer=from_conllu)


def _get_classifier_subclasses(cls: type[Classifier[Sentence]] = Classifier) -> list[type[Classifier[Sentence]]]:
    subclasses: list[type[Classifier[Sentence]]] = []
    for subclass in cls.__subclasses__():
        subclasses.extend(_get_classifier_subclasses(subclass))
        subclasses.append(subclass)
    return subclasses


# TODO: For most Flair models, e.g. "flair/ner-english-fast" or "flair/ner-english",
#  Ray does the serialization and deserialization correctly.
#  For some reason, loading "flair/ner-english-large" throws no errors, but the predictions are completely incorrect.


def register_classifier_serializers() -> None:
    """Registers Ray serializers for all subclasses of Flair's Classifier model."""

    def serializer(classifier: Classifier[Sentence]) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
            model_path: Path = Path(file.name)

        try:
            classifier.save(model_path)
            serialized: bytes = model_path.read_bytes()
        finally:
            model_path.unlink()

        return serialized

    def deserializer(model_binary: bytes) -> Classifier[Sentence]:  # pragma: no cover
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
            model_path: Path = Path(file.name)

        try:
            model_path.write_bytes(model_binary)
            classifier: Classifier[Sentence] = Classifier.load(model_path)
        finally:
            model_path.unlink()

        return classifier

    for classifier_subclass in _get_classifier_subclasses():
        ray.util.register_serializer(classifier_subclass, serializer=serializer, deserializer=deserializer)


register_sentence_serializer()
register_classifier_serializers()


@ray.remote
class Predictor(Generic[SentenceT]):
    """A Ray actor that wraps the predict function of Flair's Classifier model."""

    def __init__(
        self, model: Union[str, Path, Classifier[SentenceT]], index: Optional[int] = None
    ) -> None:  # pragma: no cover
        """
        Initializes a :class:`Predictor` as Ray actor from a Flair classifier.
        :param model: A model path or identifier for a Flair classifier.
                      If providing a Flair classifier, it is recommended to put in into Ray's object store.
                      Reference: https://docs.ray.io/en/latest/ray-core/objects.html.
        :param index: An optional index that is attached to the predictor for logging purposes.
        """
        register_sentence_serializer()

        self._model = model if isinstance(model, Classifier) else Classifier.load(model)
        self._index = index

        model_message: str = "from instance" if isinstance(model, Classifier) else f"from {model!r}"
        index_message: str = "" if index is None else f" at index {self._index!r} "
        print(
            f"Loaded Flair model {model_message} as {type(self._model).__name__!r} "
            f"predicting labels of label-type {self._model.label_type!r}{index_message}",
        )

    @overload
    def predict(self, sentences: SentenceT) -> SentenceT:
        ...

    @overload
    def predict(self, sentences: list[SentenceT]) -> list[SentenceT]:
        ...

    def predict(
        self, sentences: Union[SentenceT, list[SentenceT]], **kwargs: Any
    ) -> Union[SentenceT, list[SentenceT]]:  # pragma: no cover
        self._model.predict(sentences, **kwargs)
        return sentences

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self._index!r})"


def buffered_map(
    actor_pool: ActorPool,
    fn: Callable[[ActorHandle, V], T],
    values: Iterable[V],
    buffer_size: int,
) -> Iterator[T]:
    """Buffered version of `ray.util.ActorPool`'s `map` function."""
    value_buffer: Iterator[list[V]] = more_itertools.batched(values, n=buffer_size)
    for buffered_values in value_buffer:
        yield from actor_pool.map(fn, buffered_values)  # type: ignore[arg-type]


class PredictorPool(Generic[SentenceT]):
    def __init__(
        self,
        model: Union[str, Path, Classifier[SentenceT], ray.ObjectRef],  # type: ignore[type-arg]
        num_actors: int,
        **actor_options: Any,
    ):
        """
        Initializes a :class:`PredictorPool`.
        :param model: The underlying model for the predictors
        :param num_actors: The number of Ray actors in the predictor pool
        :param actor_options: Keyword arguments are passed to the actor as options
        """
        self._num_actors = num_actors
        self._actor_options = actor_options

        model_ref: ray.ObjectRef = model if isinstance(model, ray.ObjectRef) else ray.put(model)  # type: ignore[type-arg]  # noqa: E501
        predictors: list[ActorHandle] = [
            Predictor.options(**actor_options).remote(model_ref, index=index)  # type: ignore[attr-defined]
            if actor_options
            else Predictor.remote(model_ref, index=index)  # type: ignore[attr-defined]
            for index in range(num_actors)
        ]
        self._pool = ActorPool(predictors)

    def predict(
        self,
        sentences: Iterable[SentenceT],
        mini_batch_size: int = 32,
        buffer_size: Optional[int] = 1,
        **kwargs: Any,
    ) -> Iterator[Sentence]:
        """
        Submits the given sentences to the actor pool and predicts the class labels.
        Contrary to Flair's `Classifier.predict` function, this function does not annotate the sentence in-place.
        :param sentences: An iterable of sentences to predict
        :param mini_batch_size: The mini batch size to use
        :param buffer_size: The buffer size of how many batches of sentences are loaded in memory at once.
                            Per default, the buffer size is the number of ray actors in the pool.
        :param kwargs: Keyword arguments are passed to Flair's predict function
        :return: Yields annotated Flair sentences
        """
        buffer_size = self._num_actors if buffer_size is None else buffer_size

        def remote_predict(pipeline_actor: ActorHandle, sentences_: list[SentenceT]) -> list[SentenceT]:
            return pipeline_actor.predict.remote(sentences_, **kwargs)  # type: ignore[no-any-return]

        sentence_batches: Iterator[list[SentenceT]] = more_itertools.batched(sentences, n=mini_batch_size)
        for processed_sentences in buffered_map(
            actor_pool=self._pool, fn=remote_predict, values=sentence_batches, buffer_size=buffer_size
        ):
            yield from processed_sentences

    @property
    def num_actors(self) -> int:
        return self._num_actors

    @property
    def actor_options(self) -> dict[str, Any]:
        return self._actor_options
