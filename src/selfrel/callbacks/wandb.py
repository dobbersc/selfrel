import itertools
from collections.abc import Set
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from flair.data import Corpus, Sentence
from flair.models import RelationClassifier

from selfrel.callbacks.callback import Callback
from selfrel.trainer import SelfTrainer

if TYPE_CHECKING:
    from flair.training_utils import Result

try:
    import wandb
    from wandb.sdk.wandb_run import Run
except ImportError as e:
    msg = "Please install selfrel with the wandb option 'selfrel[wandb]' before using any wandb callbacks."
    raise ImportError(msg) from e


__all__ = ["WandbLoggerCallback"]

sentinel: object = object()


class WandbLoggerCallback(Callback):
    def __init__(
        self,
        run: Run,
        evaluation_splits: Set[Literal["train", "dev", "test"]] = frozenset(["train", "dev", "test"]),
    ) -> None:
        self.run = run
        self.evaluation_splits = evaluation_splits

        self._self_training_iteration_metric: str = "Self-Training Iteration"

    def _define_metrics(self, split: str, prediction_class: str) -> None:
        section: str = f"{split}/{prediction_class}"
        self.run.define_metric(f"{section}/precision", step_metric=self._self_training_iteration_metric)
        self.run.define_metric(f"{section}/recall", step_metric=self._self_training_iteration_metric)
        self.run.define_metric(f"{section}/f1-score", step_metric=self._self_training_iteration_metric)

    def setup(self, self_trainer: SelfTrainer, **train_parameters: Any) -> None:  # noqa: ARG002
        self.run.define_metric(self._self_training_iteration_metric)
        for split, prediction_class in itertools.product(
            self.evaluation_splits, self_trainer.model.label_dictionary.get_items()
        ):
            self._define_metrics(split, prediction_class)
        self.run.watch(self_trainer.model)

    def _evaluate_and_report_to_wandb(
        self,
        corpus: Corpus[Sentence],
        model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        self.run.log({self._self_training_iteration_metric: self_training_iteration}, step=self_training_iteration)

        for split in self.evaluation_splits:
            result: Result = model.evaluate(
                getattr(corpus, split),
                gold_label_type=model.label_type,
                mini_batch_size=train_parameters.get("eval_batch_size", 64),
                **{
                    parameter: value
                    for parameter in (
                        "embedding_storage_modemain_evaluation_metric",
                        "exclude_labels",
                        "gold_label_dictionary",
                    )
                    if (value := train_parameters.get(parameter, sentinel)) is not sentinel
                },
            )

            # Log summary metrics
            for prediction_class, scores in result.classification_report.items():
                self.run.log(
                    {
                        f"{split}/{prediction_class}/precision": scores["precision"],
                        f"{split}/{prediction_class}/recall": scores["recall"],
                        f"{split}/{prediction_class}/f1-score": scores["f1-score"],
                    },
                    step=self_training_iteration,
                )

            # Log metrics as table
            classification_report: pd.DataFrame = (
                pd.DataFrame(result.classification_report).transpose().rename_axis("class").reset_index().round(4)
            )
            classification_report = classification_report.assign(support=classification_report["support"].astype(int))
            self.run.log(
                {
                    f"{split}/classification-report/{self_training_iteration}": wandb.Table(
                        dataframe=classification_report
                    )
                },
                step=self_training_iteration,
            )

    def on_teacher_model_trained(
        self,
        self_trainer: SelfTrainer,
        teacher_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        if self_training_iteration == 0:
            self._evaluate_and_report_to_wandb(
                corpus=self_trainer.corpus,
                model=teacher_model,
                self_training_iteration=self_training_iteration,
                **train_parameters,
            )

    def on_student_model_trained(
        self,
        self_trainer: SelfTrainer,
        student_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        self._evaluate_and_report_to_wandb(
            corpus=self_trainer.corpus,
            model=student_model,
            self_training_iteration=self_training_iteration,
            **train_parameters,
        )
