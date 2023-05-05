import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, Union

import more_itertools
import ray
from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier, RelationClassifier
from flair.nn import Classifier
from ray.actor import ActorHandle
from ray.util import ActorPool
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.predictor import buffered_map, initialize_predictor_pool
from selfrel.data.serialization import to_conllu

__all__ = ["annotate"]

AbstractionLevel = Literal["token", "span", "relation", "sentence"]


def _infer_abstraction_level(classifier: Classifier) -> AbstractionLevel:
    if isinstance(classifier, SequenceTagger):
        # noinspection PyTypeChecker
        return "span" if classifier.predict_spans else "token"

    if isinstance(classifier, RelationClassifier):
        return "relation"

    if isinstance(classifier, TextClassifier):
        return "sentence"

    raise ValueError(f"Received unsupported classifier of type {type(classifier).__name__}")


def _get_sentence_serializer(abstraction_level: AbstractionLevel, label_type: str) -> functools.partial[str]:
    if abstraction_level == "token":
        return functools.partial(to_conllu, default_token_fields={label_type})
    if abstraction_level == "span":
        return functools.partial(to_conllu, default_span_fields={label_type})
    # relation and sentence
    return functools.partial(to_conllu)


def annotate(
    dataset_path: Union[str, Path],
    out_path: Union[str, Path] = None,
    model_path: str = "flair/ner-english-large",
    label_type: Optional[str] = None,
    abstraction_level: Optional[AbstractionLevel] = None,
    batch_size: int = 32,
    num_actors: int = 1,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = 1,
    buffer_size: Optional[int] = None,
) -> None:
    """See `selfrel annotate --help`."""
    # Set default buffer size
    buffer_size = num_actors if buffer_size is None else 2 * buffer_size

    # Initialize ray cluster
    ray.init()

    # Load Flair classifier
    classifier: Classifier = Classifier.load(model_path)
    classifier_ref: ray.ObjectRef = ray.put(classifier)

    label_type = classifier.label_type if label_type is None else label_type

    # Initialize predictor actor pool
    predictor_pool: ActorPool = initialize_predictor_pool(
        num_actors,
        model=classifier_ref,
        label_name=label_type,
        mini_batch_size=batch_size,
        actor_options={"num_cpus": num_cpus, "num_gpus": num_gpus},
    )

    def remote_predict(pipeline_actor: ActorHandle, sentences: list[Sentence]) -> list[Sentence]:
        return pipeline_actor.predict.remote(sentences)  # type: ignore[no-any-return]

    # Set dataset and output path
    dataset_path = Path(dataset_path)
    if out_path is None:
        out_path = dataset_path.parent / f"{dataset_path.stem}-{label_type}{dataset_path.suffix}"
    out_path = Path(out_path)

    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset: CoNLLUPlusDataset = CoNLLUPlusDataset(dataset_path, persist=False)
    sentence_batches: Iterator[list[Sentence]] = more_itertools.batched(
        tqdm(dataset, desc="Submitting to Actor Pool", position=0),
        n=batch_size,
    )

    # Set serialization function based on abstraction level
    abstraction_level = _infer_abstraction_level(classifier) if abstraction_level is None else abstraction_level
    sentence_to_conllu: functools.partial[str] = _get_sentence_serializer(abstraction_level, label_type)

    # Get global.columns including the new label type
    global_columns: str = sentence_to_conllu(dataset[0]).split("\n", 1)[0]

    # Process dataset
    with out_path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"{global_columns}\n")

        with tqdm(desc="Processing Sentences", total=len(dataset), position=1) as progress_bar:
            for processed_sentences in buffered_map(
                predictor_pool, fn=remote_predict, values=sentence_batches, buffer_size=buffer_size
            ):
                output_file.writelines(
                    sentence_to_conllu(processed_sentence, include_global_columns=False)
                    for processed_sentence in processed_sentences
                )
                progress_bar.update(len(processed_sentences))
