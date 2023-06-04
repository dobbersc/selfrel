from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch.cuda
from flair.data import Sentence
from flair.models import RelationClassifier, SequenceTagger, TextClassifier
from flair.nn import Classifier
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.data.serialization import LabelTypes, export_to_conllu
from selfrel.predictor import PredictorPool

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _add_new_label_type(
    global_label_types: LabelTypes, new_label_type: str, abstraction_level: AbstractionLevel
) -> None:
    if abstraction_level == "token":
        global_label_types.add_token_label_type(new_label_type)
    elif abstraction_level == "span":
        global_label_types.add_span_label_type(new_label_type)


def annotate(
    dataset_path: Union[str, Path],
    out: Optional[Union[str, Path]] = None,
    model: Union[str, Path] = "flair/ner-english-large",
    label_type: Optional[str] = None,
    abstraction_level: Optional[AbstractionLevel] = None,
    batch_size: int = 32,
    num_actors: int = 1,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = 1,
    buffer_size: Optional[int] = None,
) -> None:
    """See `selfrel annotate --help`."""
    # Load Flair classifier and infer its label-type and abstraction level
    classifier: Classifier[Sentence] = Classifier.load(model)
    label_type = classifier.label_type if label_type is None else label_type
    abstraction_level = _infer_abstraction_level(classifier) if abstraction_level is None else abstraction_level

    # Initialize predictor pool
    predictor_pool: PredictorPool[Sentence] = PredictorPool(
        classifier, num_actors=num_actors, num_cpus=num_cpus, num_gpus=num_gpus
    )

    # The classifier is no longer needed in the main process
    classifier_is_on_gpu: bool = next(classifier.parameters()).is_cuda
    del classifier
    if classifier_is_on_gpu:
        torch.cuda.empty_cache()

    # Load dataset
    dataset_path = Path(dataset_path)
    dataset: CoNLLUPlusDataset[Sentence] = CoNLLUPlusDataset(dataset_path, persist=False)

    # Get global label-types including the new label-type
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        global_label_types: LabelTypes = LabelTypes.from_conllu_file(dataset_file)
    _add_new_label_type(global_label_types, new_label_type=label_type, abstraction_level=abstraction_level)

    # Set output path
    if out is None:
        out = dataset_path.parent / f"{dataset_path.stem}-{label_type}{dataset_path.suffix}"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Process dataset
    processed_sentences: Iterator[Sentence] = predictor_pool.predict(
        dataset, mini_batch_size=batch_size, buffer_size=buffer_size
    )

    with out.open("w", encoding="utf-8") as output_file:
        export_to_conllu(
            output_file,
            sentences=tqdm(processed_sentences, desc="Processing Sentences", total=len(dataset)),
            global_label_types=global_label_types,
        )
