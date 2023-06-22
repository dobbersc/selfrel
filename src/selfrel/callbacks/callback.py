from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union, overload

from flair.models import RelationClassifier

if TYPE_CHECKING:
    # To avoid circular imports
    from selfrel.trainer import SelfTrainer

__all__ = ["Callback", "CallbackSequence"]


class Callback(ABC):
    @abstractmethod
    def setup(self, self_trainer: "SelfTrainer", **train_parameters: Any) -> None:
        pass

    @abstractmethod
    def on_teacher_model_trained(
        self,
        self_trainer: "SelfTrainer",
        teacher_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        pass

    @abstractmethod
    def on_student_model_trained(
        self,
        self_trainer: "SelfTrainer",
        student_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        pass


class CallbackSequence(Callback, Sequence[Callback]):
    def __init__(self, callbacks: Union[Callback, Sequence[Callback]]) -> None:
        self._callbacks = callbacks if isinstance(callbacks, Sequence) else (callbacks,)

    def setup(self, self_trainer: "SelfTrainer", **train_parameters: Any) -> None:
        for callback in self:
            callback.setup(self_trainer, **train_parameters)

    def on_teacher_model_trained(
        self,
        self_trainer: "SelfTrainer",
        teacher_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        for callback in self:
            callback.on_teacher_model_trained(self_trainer, teacher_model, self_training_iteration, **train_parameters)

    def on_student_model_trained(
        self,
        self_trainer: "SelfTrainer",
        student_model: RelationClassifier,
        self_training_iteration: int,
        **train_parameters: Any,
    ) -> None:
        for callback in self:
            callback.on_student_model_trained(self_trainer, student_model, self_training_iteration, **train_parameters)

    @overload
    def __getitem__(self, index: int) -> Callback:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Callback]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Callback, Sequence[Callback]]:
        return self._callbacks[index]

    def __len__(self) -> int:
        return len(self._callbacks)
