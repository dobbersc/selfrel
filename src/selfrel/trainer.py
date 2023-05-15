import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from flair.data import Corpus, Sentence
from flair.models import RelationClassifier
from flair.trainers import ModelTrainer
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.data.serialization import LabelTypes, export_to_conllu
from selfrel.predictor import PredictorPool
from selfrel.selection_strategies import PredictionConfidence, SelectionStrategy
from selfrel.utils.copy import deepcopy_flair_model

if TYPE_CHECKING:
    from collections.abc import Iterator

    from flair.datasets import FlairDatapointDataset
    from flair.models.relation_classifier_model import EncodedSentence

logger = logging.getLogger("flair")


class SelfTrainer:
    def __init__(
        self,
        model: RelationClassifier,
        annotated_corpus: Corpus[Sentence],
        unlabelled_dataset: CoNLLUPlusDataset,
        num_actors: int = 1,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = 1,
        buffer_size: Optional[int] = None,
        prediction_batch_size: int = 32,
    ) -> None:
        if model.zero_tag_value == "O":
            msg = f"Relation classifiers with zero_tag_value={model.zero_tag_value!r} are currently not supported."
            raise ValueError(msg)

        self._model = model

        self._annotated_corpus = annotated_corpus
        self._transformed_annotated_corpus: Corpus[EncodedSentence] = model.transform_corpus(annotated_corpus)

        self._unlabelled_dataset = unlabelled_dataset

        self.num_actors = num_actors
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.buffer_size = num_actors if buffer_size is None else buffer_size
        self.prediction_batch_size = prediction_batch_size

    def _train_model(self, base_path: Path, corpus: Corpus[Sentence], **kwargs: Any) -> RelationClassifier:
        trained_model: RelationClassifier = deepcopy_flair_model(self._model)
        trainer: ModelTrainer = ModelTrainer(model=trained_model, corpus=corpus)
        trainer.fine_tune(base_path, **kwargs)
        return trained_model

    def _annotate_unlabelled_dataset(self, teacher: RelationClassifier, output_path: Path) -> CoNLLUPlusDataset:
        # Initialize predictor pool
        predictor_pool: PredictorPool[Sentence] = PredictorPool(
            teacher,  # type: ignore[arg-type]
            num_actors=self.num_actors,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )

        # Get global label-types
        with self._unlabelled_dataset.dataset_path.open("r", encoding="utf-8") as dataset_file:
            global_label_types: LabelTypes = LabelTypes.from_conllu_file(dataset_file)

        # Process dataset
        processed_sentences: Iterator[Sentence] = predictor_pool.predict(
            self._unlabelled_dataset, mini_batch_size=self.prediction_batch_size, buffer_size=self.buffer_size
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            export_to_conllu(
                output_file,
                sentences=tqdm(
                    processed_sentences, desc="Annotating Unlabelled Dataset", total=len(self._unlabelled_dataset)
                ),
                global_label_types=global_label_types,
            )

        return CoNLLUPlusDataset(output_path, persist=False, disable_progress_bar=True)

    def _select_datapoints(
        self, dataset: CoNLLUPlusDataset, selection_strategy: SelectionStrategy, output_path: Path
    ) -> CoNLLUPlusDataset:
        with dataset.dataset_path.open("r", encoding="utf-8") as dataset_file:
            global_label_types: LabelTypes = LabelTypes.from_conllu_file(dataset_file)

        selected_sentences: Iterator[Sentence] = selection_strategy.select_relations(
            dataset, label_type=self._model.label_type
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            export_to_conllu(
                output_file,
                sentences=tqdm(selected_sentences, desc="Selecting Confident Data Points", total=len(dataset)),
                global_label_types=global_label_types,
            )

        # TODO: Actually is makes more sense to load it into memory right away.
        #  Then we would not need to load the sentences as conllu dataset and could return a list of sentences
        return CoNLLUPlusDataset(output_path, persist=False, disable_progress_bar=True)

    def train(
        self,
        base_path: Union[str, Path],
        self_training_iterations: int = 1,
        selection_strategy: Optional[SelectionStrategy] = None,
        **kwargs: Any,
    ) -> RelationClassifier:
        # Create output folder
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Set default selection strategy
        selection_strategy = PredictionConfidence() if selection_strategy is None else selection_strategy

        # Train initial teacher model on transformed annotated corpus
        logger.info("Training initial teacher model")
        teacher_model: RelationClassifier = self._train_model(
            base_path=base_path / "iteration-0", corpus=self._transformed_annotated_corpus, **kwargs
        )

        for self_training_iteration in range(1, self_training_iterations + 1):
            logger.info("Self-training iteration: %s", self_training_iteration)
            iteration_base_path: Path = base_path / f"iteration-{self_training_iteration}"

            # Predict unlabelled dataset
            annotated_dataset: CoNLLUPlusDataset = self._annotate_unlabelled_dataset(
                teacher_model, output_path=base_path / f"iteration-{self_training_iterations}" / "data.conllup"
            )

            # Select confident data points
            annotated_dataset = self._select_datapoints(
                annotated_dataset,
                selection_strategy=selection_strategy,
                output_path=iteration_base_path / "data.conllup",
            )
            transformed_annotated_dataset: FlairDatapointDataset[EncodedSentence] = self._model.transform_dataset(
                annotated_dataset
            )

            # Train student model on transformed augmented corpus
            assert self._transformed_annotated_corpus.train
            transformed_augmented_corpus: Corpus[Sentence] = Corpus(
                train=ConcatDataset([self._transformed_annotated_corpus.train, transformed_annotated_dataset]),
                dev=self._transformed_annotated_corpus.dev,
                test=self._transformed_annotated_corpus.test,
            )

            student_model: RelationClassifier = self._train_model(
                base_path=iteration_base_path, corpus=transformed_augmented_corpus, **kwargs
            )

            # Set student as new teacher model
            teacher_model = student_model

        return teacher_model
