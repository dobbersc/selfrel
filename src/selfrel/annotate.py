import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, Union

import more_itertools
import ray
from flair.data import Sentence
from ray.actor import ActorHandle
from ray.util import ActorPool
from tqdm import tqdm

from selfrel.data.conllu import CoNLLUPlusDataset
from selfrel.predictor import buffered_map, initialize_predictor_pool
from selfrel.serialization import to_conllu

__all__ = ["annotate_entities"]


def annotate_entities(
    dataset_path: Union[str, Path],
    out_path: Union[str, Path],
    model_path: str = "flair/ner-english-large",
    label_type: str = "ner",
    abstraction_level: Literal["token", "span", "relation", "sentence"] = "span",
    batch_size: int = 32,
    num_actors: int = 1,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = None,
    buffer_size: Optional[int] = None,
) -> None:
    """See `selfrel annotate --help`."""

    # Create output directory
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set default buffer size
    buffer_size = num_actors if buffer_size is None else 2 * buffer_size

    # Initialize ray cluster
    ray.init()

    # Initialize predictor actor pool
    predictor_pool: ActorPool = initialize_predictor_pool(
        num_actors,
        model_path=model_path,
        label_name=label_type,
        mini_batch_size=batch_size,
        actor_options={"num_cpus": num_cpus, "num_gpus": num_gpus},
    )

    def remote_predict(pipeline_actor: ActorHandle, sentences: list[Sentence]) -> list[Sentence]:
        return pipeline_actor.predict.remote(sentences)  # type: ignore[no-any-return]

    # Load dataset
    dataset: CoNLLUPlusDataset = CoNLLUPlusDataset(dataset_path, persist=False)
    sentence_batches: Iterator[list[Sentence]] = more_itertools.batched(
        tqdm(dataset, desc="Submitting to Actor Pool", position=1),
        n=batch_size,
    )

    # Set serialization function based on abstraction level
    to_conllu_partial: functools.partial[str]
    if abstraction_level == "token":
        to_conllu_partial = functools.partial(to_conllu, default_token_fields={label_type})
    elif abstraction_level == "span":
        to_conllu_partial = functools.partial(to_conllu, default_span_fields={label_type})
    else:
        # relation and sentence
        to_conllu_partial = functools.partial(to_conllu)

    # Get global.columns including the new label type
    global_columns: str = to_conllu_partial(dataset[0]).split("\n", 1)[0]

    # Process dataset
    with out_path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"{global_columns}\n")

        with tqdm(desc="Processing Sentences", total=len(dataset), position=0) as progress_bar:
            for processed_sentences in buffered_map(
                predictor_pool, fn=remote_predict, values=sentence_batches, buffer_size=buffer_size
            ):
                output_file.writelines(
                    to_conllu_partial(processed_sentence, include_global_columns=False)
                    for processed_sentence in processed_sentences
                )
                progress_bar.update(len(processed_sentences))
