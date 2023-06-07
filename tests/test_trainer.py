from pathlib import Path

import flair
import pandas as pd
import pytest
import torch.cuda
from flair.data import Corpus, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier

from selfrel.data import CoNLLUPlusDataset
from selfrel.selection_strategies import PredictionConfidence
from selfrel.trainer import SelfTrainer


@pytest.fixture(scope="class")
def self_trainer(resources_dir: Path) -> SelfTrainer:
    flair.set_seed(42)

    # Step 1: Create the training data and support dataset
    # The relation extractor is *not* trained end-to-end.
    # A corpus for training the relation extractor requires annotated entities and relations.
    corpus: Corpus[Sentence] = Corpus(
        train=CoNLLUPlusDataset(resources_dir / "training_data" / "train.conllup"),
        dev=CoNLLUPlusDataset(resources_dir / "training_data" / "test.conllup"),
        test=CoNLLUPlusDataset(resources_dir / "training_data" / "test.conllup"),
    )
    support_dataset: CoNLLUPlusDataset[Sentence] = CoNLLUPlusDataset(
        resources_dir / "training_data" / "support-dataset.conllup", persist=False
    )

    # Step 3: Initialize fine-tunable transformer embedding
    embeddings = TransformerDocumentEmbeddings(model="distilbert-base-uncased", layers="-1", fine_tune=True)

    # Step 2: Make the label dictionary
    label_dictionary = corpus.make_label_dictionary("relation")

    # Step 4: Initialize relation classifier
    model: RelationClassifier = RelationClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="relation",
        entity_label_types="ner",
        entity_pair_labels={("ORG", "PER"), ("LOC", "PER")},
        zero_tag_value="no_relation",
    )

    # Step 4: Initialize self-trainer
    trainer: SelfTrainer = SelfTrainer(
        model=model, corpus=corpus, support_dataset=support_dataset, num_gpus=1 if torch.cuda.is_available() else 0
    )

    return trainer


@pytest.mark.usefixtures("_init_ray")
class TestSelfTrainerArtifacts:
    @staticmethod
    def assert_artifacts(base_path: Path) -> None:
        last_iteration_path: Path = base_path / "iteration-1"
        support_dataset_dir: Path = last_iteration_path / "support-datasets"
        relation_overview_dir: Path = last_iteration_path / "relation-overviews"

        # Test the result model for each iteration
        for iteration in range(2):
            RelationClassifier.load(base_path / f"iteration-{iteration}" / "final-model.pt")

        # Test support dataset artifacts
        support_dataset_full = CoNLLUPlusDataset(support_dataset_dir / "support-dataset.conllup")
        assert len(support_dataset_full) == 3  # Three sentences
        support_dataset_selection = CoNLLUPlusDataset(support_dataset_dir / "selected-support-dataset.conllup")
        assert len(support_dataset_selection) == 3  # All sentences are selected
        support_dataset_encoded = CoNLLUPlusDataset(support_dataset_dir / "encoded-support-dataset.conllup")
        assert len(support_dataset_encoded) == 6  # Two relations for each sentence

        # Test relation overview artifacts
        relation_overview = pd.read_parquet(relation_overview_dir / "relation-overview.parquet")
        assert len(relation_overview.index) == 6
        scored_relation_overview = pd.read_parquet(relation_overview_dir / "scored-relation-overview.parquet")
        assert len(scored_relation_overview.index) == 6
        selected_relation_overview = pd.read_parquet(relation_overview_dir / "selected-relation-overview.parquet")
        assert len(selected_relation_overview.index) == 6

    def test_standard(self, self_trainer: SelfTrainer, tmp_path: Path) -> None:
        base_path: Path = tmp_path / "self-trainer"
        self_trainer.train(
            base_path,
            max_epochs=2,
            self_training_iterations=1,
            selection_strategy=PredictionConfidence(min_confidence=0.8),
            main_evaluation_metric=("macro avg", "f1-score"),
        )
        self.assert_artifacts(base_path)

    def test_precomputed(self, self_trainer: SelfTrainer, tmp_path: Path, resources_dir: Path) -> None:
        base_path: Path = tmp_path / "self-trainer"
        precomputed_dir: Path = resources_dir / "training_data" / "precomputed"
        self_trainer.train(
            base_path,
            max_epochs=2,
            self_training_iterations=1,
            selection_strategy=PredictionConfidence(min_confidence=0.8),
            precomputed_annotated_support_datasets=[precomputed_dir / "support-dataset.conllup"],
            precomputed_relation_overviews=[precomputed_dir / "relation-overview.parquet"],
            main_evaluation_metric=("macro avg", "f1-score"),
        )
        self.assert_artifacts(base_path)
