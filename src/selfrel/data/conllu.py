import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Optional, TextIO, TypeVar, Union, cast, overload

from flair.data import Sentence
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from selfrel.data.serialization import from_conllu

__all__ = ["CoNLLUPlusDataset"]

SentenceT = TypeVar("SentenceT", bound=Sentence)


def _serialized_conllu_plus_sentence_iter(
    fp: TextIO,
    dataset_name: str,
    show_progress_bar: bool = True,
) -> Iterator[str]:
    dataset_size: int = os.fstat(fp.fileno()).st_size
    with tqdm(
        total=dataset_size,
        unit="B",
        unit_scale=True,
        desc=f"Loading Dataset {dataset_name!r}",
        disable=not show_progress_bar,
    ) as progress_bar:
        global_columns: str = fp.readline()
        progress_bar.update(len(global_columns.encode("utf-8")))
        if not global_columns.startswith("# global.columns"):
            msg = "Missing CoNLL-U Plus required 'global.columns'"
            raise ValueError(msg)

        sentence_lines: list[str] = []
        for line in fp:
            sentence_lines.append(line)

            # There must be exactly one blank line after every sentence, including the last sentence in the file.
            # Empty sentences are not allowed. (From https://universaldependencies.org/format.html)
            # Therefore, a blank line ends the current CoNLL-U sentence.
            if line == "\n":
                sentence = "".join(sentence_lines)
                yield f"{global_columns}{sentence}"
                progress_bar.update(len(sentence.encode("utf-8")))
                sentence_lines = []

        if sentence_lines:
            sentence = "".join(sentence_lines)
            yield f"{global_columns}{sentence}"
            progress_bar.update(len(sentence.encode("utf-8")))


class CoNLLUPlusDataset(Dataset[SentenceT], Sequence[Sentence]):
    """
    Dataset of Flair Sentences parsed from a CoNLL-U Plus file.

    - Universal Dependencies CoNLL-U format: https://universaldependencies.org/format.html
    - Universal Dependencies CoNLL-U Plus format: https://universaldependencies.org/ext-format.html
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        persist: bool = True,
        show_progress_bar: bool = True,
        # Typing workaround for specifying a default value for a generic parameter
        # https://github.com/python/mypy/issues/3737
        sentence_type: type[SentenceT] = Sentence,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> None:
        """
        Initializes a :class:`CoNLLUPlusDataset` from a CoNLL-U Plus file.
        :param dataset_path: The path to the dataset file
        :param dataset_name: A name for the dataset. Per default, the dataset's path is used.
        :param persist: If True, the entire dataset will be converted to Flair Sentences
                        immediately upon initialization and loaded into memory.
                        If the dataset is very large, temporarily disabling Python's generational garbage collector
                        during the initialization may help.
                        If False, only the dataset file is loaded into memory as a string.
                        Calls to `__getitem__` or `__iter__` parse the dataset file on the fly to Flair Sentences.
                        The returned Sentences are not persistent.
        :param show_progress_bar: Flag to show the progress bar when loading the dataset
        :param kwargs: Keywords arguments are passed to `conllu.parse`
        """
        self._dataset_path = Path(dataset_path)
        self._dataset_name = str(self._dataset_path) if dataset_name is None else dataset_name
        self._persist = persist
        self._sentence_type = sentence_type
        self._kwargs: dict[str, Any] = kwargs

        with self.dataset_path.open("r", encoding="utf-8") as dataset_file:
            self._sentences: Union[tuple[SentenceT, ...], tuple[str, ...]]
            if self.is_persistent:
                self._sentences = tuple(
                    from_conllu(sentence, cls=self._sentence_type, **self._kwargs)
                    for sentence in _serialized_conllu_plus_sentence_iter(
                        dataset_file, self._dataset_name, show_progress_bar
                    )
                )
            else:
                self._sentences = tuple(
                    _serialized_conllu_plus_sentence_iter(dataset_file, self._dataset_name, show_progress_bar)
                )

        if not self._sentences:
            msg = "Empty datasets are not supported"
            raise ValueError(msg)

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def is_persistent(self) -> bool:
        return self._persist

    @overload
    def __getitem__(self, item: int) -> SentenceT:
        ...

    @overload
    def __getitem__(self, item: slice) -> tuple[SentenceT, ...]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[SentenceT, tuple[SentenceT, ...]]:
        if self.is_persistent:
            assert isinstance(self._sentences[0], Sentence)
            return cast(tuple[SentenceT, ...], self._sentences)[item]

        assert isinstance(self._sentences[0], str)
        sentences: tuple[str, ...] = cast(tuple[str, ...], self._sentences)
        return (
            from_conllu(sentences[item], cls=self._sentence_type, **self._kwargs)
            if isinstance(item, int)
            else tuple(from_conllu(sentence, cls=self._sentence_type, **self._kwargs) for sentence in sentences[item])
        )

    def __iter__(self) -> Iterator[SentenceT]:
        if self.is_persistent:
            assert isinstance(self._sentences[0], Sentence)
            yield from cast(tuple[SentenceT, ...], self._sentences)
        else:
            assert isinstance(self._sentences[0], str)
            for sentence in cast(tuple[str, ...], self._sentences):
                yield from_conllu(sentence, cls=self._sentence_type, **self._kwargs)

    def __len__(self) -> int:
        return len(self._sentences)

    def __repr__(self) -> str:
        attributes: list[tuple[str, str]] = [("dataset_path", repr(self._dataset_path))]
        if self._dataset_name != str(self._dataset_path):
            attributes.append(("dataset_name", repr(self._dataset_name)))
        attributes.append(("persist", repr(self._persist)))
        return f"{type(self).__name__}({', '.join(f'{attribute}={value}' for attribute, value in attributes)})"
