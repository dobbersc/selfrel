import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import more_itertools
import ray
import torch.cuda
from flair.data import Sentence
from flair.models import RelationClassifier, SequenceTagger, TextClassifier
from flair.nn import Classifier
from ray.actor import ActorHandle
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.data.serialization import LabelTypes, to_conllu
from selfrel.predictor import buffered_map, initialize_predictor_pool

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ray.util import ActorPool

__all__ = ["annotate"]

AbstractionLevel = Literal["token", "span", "relation", "sentence"]


def _infer_abstraction_level(classifier: Classifier[Sentence]) -> AbstractionLevel:
    if isinstance(classifier, SequenceTagger):
        # noinspection PyTypeChecker
        return "span" if classifier.predict_spans else "token"

    if isinstance(classifier, RelationClassifier):
        return "relation"

    if isinstance(classifier, TextClassifier):
        return "sentence"

    msg = f"Received unsupported classifier of type {type(classifier).__name__}"
    raise ValueError(msg)


def _infer_global_label_types(
    dataset: CoNLLUPlusDataset,
    new_label_type: str,
    abstraction_level: AbstractionLevel,
) -> LabelTypes:
    global_label_types: LabelTypes = LabelTypes(token_level=[], span_level=[])
    if abstraction_level == "token":
        global_label_types.add_token_label_type(new_label_type)
    elif abstraction_level == "span":
        global_label_types.add_span_label_type(new_label_type)

    for sentence in tqdm(dataset, desc="Inspecting Sentence Annotations"):
        label_types: LabelTypes = LabelTypes.from_flair_sentence(sentence)
        global_label_types.add_token_label_type(label_types.token_level)
        global_label_types.add_span_label_type(label_types.span_level)
    return global_label_types


def annotate(
    dataset: Union[str, Path, CoNLLUPlusDataset],
    out: Optional[Union[str, Path]] = None,
    model: Union[str, Path, Classifier[Sentence]] = "flair/ner-english-large",
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

    # Load Flair classifier
    classifier: Classifier[Sentence] = model if isinstance(model, Classifier) else Classifier.load(model)
    classifier_ref = ray.put(classifier)

    label_type = classifier.label_type if label_type is None else label_type
    abstraction_level = _infer_abstraction_level(classifier) if abstraction_level is None else abstraction_level

    # The classifier is no longer needed in the main process if has been loaded in this function
    if not isinstance(model, Classifier):
        classifier_is_on_gpu: bool = next(classifier.parameters()).is_cuda
        del classifier
        if classifier_is_on_gpu:
            torch.cuda.empty_cache()

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

    # Load dataset
    dataset = dataset if isinstance(dataset, CoNLLUPlusDataset) else CoNLLUPlusDataset(dataset, persist=False)
    sentence_batches: Iterator[list[Sentence]] = more_itertools.batched(
        tqdm(dataset, desc="Submitting to Actor Pool", position=0),
        n=batch_size,
    )

    # Get global.columns including the new label type
    global_label_types: LabelTypes = _infer_global_label_types(dataset, label_type, abstraction_level)
    global_columns: str = f"# global.columns = {' '.join(global_label_types.as_global_columns())}"

    # Set serialization function including the global label types
    sentence_to_conllu = functools.partial(
        to_conllu,
        default_token_fields=frozenset(global_label_types.token_level),
        default_span_fields=frozenset(global_label_types.span_level),
    )

    # Set output path
    if out is None:
        dataset_path: Path = dataset.dataset_path
        out = dataset_path.parent / f"{dataset_path.stem}-{label_type}{dataset_path.suffix}"
    out = Path(out)

    # Create output directory
    out.parent.mkdir(parents=True, exist_ok=True)

    # Process dataset
    with out.open("w", encoding="utf-8") as output_file:
        output_file.write(f"{global_columns}\n")

        with tqdm(desc="Processing Sentences", total=len(dataset), position=1) as progress_bar:
            for processed_sentences in buffered_map(
                predictor_pool,
                fn=remote_predict,
                values=sentence_batches,
                buffer_size=buffer_size,
            ):
                output_file.writelines(
                    sentence_to_conllu(processed_sentence, include_global_columns=False)
                    for processed_sentence in processed_sentences
                )
                progress_bar.update(len(processed_sentences))
