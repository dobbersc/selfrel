import functools
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Optional, Union

import flair
from flair.data import Corpus, Sentence
from flair.datasets import RE_ENGLISH_CONLL04, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier

from selfrel.data import CoNLLUPlusDataset
from selfrel.selection_strategies import Entropy, Occurrence, PredictionConfidence, SelectionStrategy
from selfrel.trainer import SelfTrainer
from selfrel.utils.inspect_relations import infer_entity_pair_labels

__all__ = ["train"]

logger: logging.Logger = logging.getLogger("flair")

_CORPORA: Final[dict[str, Callable[[], Corpus[Sentence]]]] = {
    "conll04": functools.partial(
        RE_ENGLISH_CONLL04, label_name_map={"Peop": "PER", "Org": "ORG", "Loc": "LOC", "Other": "MISC"}
    ),
}


def _load_corpus(corpus_name: str) -> Corpus[Sentence]:
    try:
        return _CORPORA[corpus_name]()
    except KeyError as e:
        msg = f"The corpus {corpus_name!r} is not supported"
        raise ValueError(msg) from e


def infer_selection_strategy(
    selection_strategy: Literal["prediction-confidence", "occurrence", "entropy"],
    min_confidence: Optional[float],
    min_occurrence: Optional[int],
    max_occurrence: Optional[int],
    distinct: Optional[Literal["sentence", "in-between-text"]],
    base: Optional[int],
    max_entropy: Optional[float],
    top_k: Optional[int],
    label_distribution: Optional[dict[str, float]],
) -> SelectionStrategy:
    strategy: SelectionStrategy
    if selection_strategy == "prediction-confidence":
        strategy = (
            PredictionConfidence(top_k=top_k, label_distribution=label_distribution)
            if min_confidence is None
            else PredictionConfidence(
                min_confidence=min_confidence,
                top_k=top_k,
                label_distribution=label_distribution,
            )
        )
    elif selection_strategy == "occurrence":
        strategy = (
            Occurrence(distinct=distinct, top_k=top_k, label_distribution=label_distribution)
            if min_occurrence is None
            else Occurrence(
                min_occurrence=min_occurrence,
                distinct=distinct,
                top_k=top_k,
                label_distribution=label_distribution,
            )
        )
    elif selection_strategy == "entropy":
        strategy = (
            Entropy(
                base=base,
                min_occurrence=min_occurrence,
                max_occurrence=max_occurrence,
                min_confidence=min_confidence,
                distinct=distinct,
                top_k=top_k,
                label_distribution=label_distribution,
            )
            if max_entropy is None
            else Entropy(
                base=base,
                max_entropy=max_entropy,
                min_occurrence=min_occurrence,
                max_occurrence=max_occurrence,
                min_confidence=min_confidence,
                distinct=distinct,
                top_k=top_k,
                label_distribution=label_distribution,
            )
        )

    else:  # pragma: no cover
        msg = f"Specified invalid selection strategy {selection_strategy!r}"  # type: ignore[unreachable]
        raise ValueError(msg)

    return strategy


def train(
    corpus_name: Literal["conll04"],
    support_dataset: Union[str, Path, CoNLLUPlusDataset[Sentence]],
    base_path: Union[str, Path] = Path(),
    down_sample_train: Optional[float] = None,
    train_with_dev: bool = False,
    transformer: str = "bert-base-uncased",
    max_epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    cross_augmentation: bool = True,
    entity_pair_label_filter: bool = True,
    encoding_strategy: Literal[
        "entity-mask",
        "typed-entity-mask",
        "entity-marker",
        "entity-marker-punct",
        "typed-entity-marker",
        "typed-entity-marker-punct",
    ] = "typed-entity-marker-punct",
    self_training_iterations: int = 1,
    reinitialize: bool = True,
    selection_strategy: Literal["prediction-confidence", "occurrence", "entropy"] = "prediction-confidence",
    min_confidence: Optional[float] = None,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[int] = None,
    distinct: Optional[Literal["sentence", "in-between-text"]] = None,
    base: Optional[int] = None,
    max_entropy: Optional[float] = None,
    top_k: Optional[int] = None,
    label_distribution: Optional[dict[str, float]] = None,
    precomputed_annotated_support_datasets: Sequence[Union[str, Path, None]] = (),
    precomputed_relation_overviews: Sequence[Union[str, Path, None]] = (),
    num_actors: int = 1,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = 1.0,
    buffer_size: Optional[int] = None,
    prediction_batch_size: int = 32,
    evaluation_split: Literal["train", "dev", "test"] = "test",
    use_final_model_for_evaluation: bool = True,
    exclude_labels_from_evaluation: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> None:
    hyperparameters: dict[str, Any] = locals()

    if seed is not None:
        flair.set_seed(seed)

    # Step 1: Create the training data and support dataset
    # The relation extractor is *not* trained end-to-end.
    # A corpus for training the relation extractor requires annotated entities and relations.
    corpus: Corpus[Sentence] = _load_corpus(corpus_name)
    if down_sample_train is not None:
        corpus = corpus.downsample(
            percentage=down_sample_train,
            downsample_train=True,
            downsample_dev=train_with_dev,
            downsample_test=False,
        )
    if evaluation_split in ["train", "dev"]:
        corpus = Corpus(
            train=corpus.train,
            dev=corpus.dev,
            test=getattr(corpus, evaluation_split),
            # Keep original corpus structure
            name=corpus.name,
            sample_missing_splits=False,
        )

    support_dataset = (
        support_dataset
        if isinstance(support_dataset, CoNLLUPlusDataset)
        else CoNLLUPlusDataset(support_dataset, persist=False)
    )

    # Step 2: Make the label dictionary and infer entity-pair labels from the corpus
    label_dictionary = corpus.make_label_dictionary("relation")
    entity_pair_labels: Optional[set[tuple[str, str]]] = (
        infer_entity_pair_labels(
            (batch[0] for batch in DataLoader(corpus.train, batch_size=1)),
            relation_label_type="relation",
            entity_label_types="ner",
        )
        if entity_pair_label_filter
        else None
    )

    # Step 3: Initialize fine-tunable transformer embedding
    embeddings = TransformerDocumentEmbeddings(model=transformer, layers="-1", fine_tune=True)

    # Step 4: Initialize relation classifier
    model: RelationClassifier = RelationClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="relation",
        entity_label_types="ner",
        entity_pair_labels=entity_pair_labels,
        cross_augmentation=cross_augmentation,
        zero_tag_value="no_relation",
        encoding_strategy=getattr(
            sys.modules["flair.models.relation_classifier_model"], encoding_strategy.title().replace("-", "")
        )(),
        allow_unk_tag=False,
    )

    # Step 4: Initialize self-trainer and selection strategy
    trainer: SelfTrainer = SelfTrainer(
        model=model,
        corpus=corpus,
        support_dataset=support_dataset,
        num_actors=num_actors,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        buffer_size=buffer_size,
        prediction_batch_size=prediction_batch_size,
    )

    strategy: SelectionStrategy = infer_selection_strategy(
        selection_strategy=selection_strategy,
        min_confidence=min_confidence,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
        distinct=distinct,
        base=base,
        max_entropy=max_entropy,
        top_k=top_k,
        label_distribution=label_distribution,
    )

    logger.info("-" * 100)
    logger.info("Parameters:")
    for parameter, parameter_value in hyperparameters.items():
        logger.info(" - %s: %s", parameter, repr(parameter_value))
    logger.info("-" * 100)

    # Step 5: Run self-trainer
    trainer.train(
        base_path,
        self_training_iterations=self_training_iterations,
        selection_strategy=strategy,
        reinitialize=reinitialize,
        precomputed_annotated_support_datasets=precomputed_annotated_support_datasets,
        precomputed_relation_overviews=precomputed_relation_overviews,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        train_with_dev=train_with_dev,
        main_evaluation_metric=("macro avg", "f1-score"),
        exclude_labels=[] if exclude_labels_from_evaluation is None else exclude_labels_from_evaluation,
        eval_batch_size=prediction_batch_size,
        use_final_model_for_eval=use_final_model_for_evaluation,
    )
