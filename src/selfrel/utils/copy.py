import tempfile
from pathlib import Path
from typing import Any, TypeVar, cast

from flair.data import Sentence
from flair.nn import Model

from selfrel.data import from_conllu, to_conllu

ModelT = TypeVar("ModelT", bound=Model[Any])


def deepcopy_flair_model(model: ModelT, copy_optimizer_state: bool = False) -> ModelT:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as file:
        model_path: Path = Path(file.name)

    try:
        model.save(model_path, checkpoint=copy_optimizer_state)
        model_copy: ModelT = cast(ModelT, model.load(model_path))
    finally:
        model_path.unlink()

    return model_copy


def deepcopy_flair_sentence(sentence: Sentence) -> Sentence:
    return from_conllu(to_conllu(sentence))
