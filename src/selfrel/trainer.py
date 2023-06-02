import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

from flair.data import Corpus, Sentence
from flair.models import RelationClassifier
from flair.models.relation_classifier_model import EncodedSentence
from flair.trainers import ModelTrainer
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.data.serialization import LabelTypes, export_to_conllu
from selfrel.predictor import PredictorPool
from selfrel.selection_strategies import PredictionConfidence, SelectionStrategy
from selfrel.utils.copy import deepcopy_flair_model

__all__ = ["SelfTrainer"]

logger = logging.getLogger("flair")


@contextlib.contextmanager
def _set_cross_augmentation(relation_classifier: RelationClassifier, cross_augmentation: bool) -> Iterator[None]:
    original_cross_augmentation: bool = relation_classifier.cross_augmentation
    relation_classifier.cross_augmentation = cross_augmentation
    yield
    relation_classifier.cross_augmentation = original_cross_augmentation


class SelfTrainer:
    def __init__(
        self,
        model: RelationClassifier,
        corpus: Corpus[Sentence],
        support_dataset: CoNLLUPlusDataset[Sentence],
        num_actors: int = 1,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = 1.0,
        buffer_size: Optional[int] = None,
        prediction_batch_size: int = 32,
    ) -> None:
        if model.zero_tag_value == "O":
            msg = f"Relation classifiers with zero_tag_value={model.zero_tag_value!r} are currently not supported."
            raise ValueError(msg)

        self._model = model

        self.corpus = corpus
        self._encoded_corpus: Corpus[EncodedSentence] = model.transform_corpus(corpus)

        self._support_dataset = support_dataset

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

    def _annotate_support_dataset(self, teacher: RelationClassifier, output_path: Path) -> CoNLLUPlusDataset[Sentence]:
        # Initialize predictor pool
        predictor_pool: PredictorPool[Sentence] = PredictorPool(
            teacher,  # type: ignore[arg-type]
            num_actors=self.num_actors,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
        )

        # Get global label-types
        with self._support_dataset.dataset_path.open("r", encoding="utf-8") as dataset_file:
            global_label_types: LabelTypes = LabelTypes.from_conllu_file(dataset_file)

        # Select sentences with NER annotations
        sentences: Iterator[Sentence] = (
            sentence
            for sentence in self._support_dataset
            if any(label_type in sentence.annotation_layers for label_type in self._model.entity_label_types)
        )

        # Process dataset
        processed_sentences: Iterator[Sentence] = predictor_pool.predict(
            sentences, mini_batch_size=self.prediction_batch_size, buffer_size=self.buffer_size
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            export_to_conllu(
                output_file,
                sentences=tqdm(
                    processed_sentences, desc="Annotating Support Dataset", total=len(self._support_dataset)
                ),
                global_label_types=global_label_types,
            )

        return CoNLLUPlusDataset(output_path, persist=False)

    def _select_support_datapoints(
        self, dataset: CoNLLUPlusDataset[Sentence], selection_strategy: SelectionStrategy, output_path: Path
    ) -> CoNLLUPlusDataset[Sentence]:
        with dataset.dataset_path.open("r", encoding="utf-8") as dataset_file:
            global_label_types: LabelTypes = LabelTypes.from_conllu_file(dataset_file)

        selected_sentences: Iterator[Sentence] = selection_strategy.select_relations(
            dataset,
            entity_label_types=set(self._model.entity_label_types.keys()),
            relation_label_type=self._model.label_type,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            export_to_conllu(
                output_file,
                sentences=selected_sentences,
                global_label_types=global_label_types,
            )

        return CoNLLUPlusDataset(output_path, persist=False)

    def _encode_support_dataset(
        self, dataset: CoNLLUPlusDataset[Sentence], output_path: Path
    ) -> CoNLLUPlusDataset[EncodedSentence]:
        with _set_cross_augmentation(self._model, cross_augmentation=False):
            encoded_sentences: Iterator[EncodedSentence] = (
                encoded_sentence
                for sentence in tqdm(dataset, desc="Encoding Support Dataset")
                for encoded_sentence in self._model.transform_sentence(sentence)
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            export_to_conllu(
                output_file,
                sentences=encoded_sentences,
                global_label_types=LabelTypes(token_level=[], span_level=[]),
            )

        return CoNLLUPlusDataset(output_path, sentence_type=EncodedSentence, persist=False)

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

        # Train initial teacher model on transformed corpus
        logger.info("Training initial teacher model")
        teacher_model: RelationClassifier = self._train_model(
            base_path=base_path / "iteration-0", corpus=self._encoded_corpus, **kwargs
        )

        for self_training_iteration in range(1, self_training_iterations + 1):
            logger.info("Self-training iteration: %s", self_training_iteration)
            iteration_base_path: Path = base_path / f"iteration-{self_training_iteration}"

            # Predict support dataset
            annotated_support_dataset: CoNLLUPlusDataset[Sentence] = self._annotate_support_dataset(
                teacher_model, output_path=iteration_base_path / "support-dataset-full.conllup"
            )

            # Select confident data points
            annotated_support_dataset = self._select_support_datapoints(
                annotated_support_dataset,
                selection_strategy=selection_strategy,
                output_path=iteration_base_path / "support-dataset-selection.conllup",
            )

            # Encode annotated support dataset
            encoded_support_dataset: CoNLLUPlusDataset[EncodedSentence] = self._encode_support_dataset(
                annotated_support_dataset, output_path=iteration_base_path / "support-dataset-encoded.conllup"
            )

            # Train student model on encoded augmented corpus
            assert self._encoded_corpus.train
            encoded_augmented_corpus: Corpus[EncodedSentence] = Corpus(
                train=ConcatDataset([self._encoded_corpus.train, encoded_support_dataset]),
                dev=self._encoded_corpus.dev,
                test=self._encoded_corpus.test,
            )

            student_model: RelationClassifier = self._train_model(
                base_path=iteration_base_path, corpus=encoded_augmented_corpus, **kwargs
            )

            # Set student as new teacher model
            teacher_model = student_model

        return teacher_model
