import os
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, Optional, TextIO, Union, cast, overload

from flair.data import Sentence
from joblib import Parallel, delayed
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from selfrel.data.serialization import from_conllu

__all__ = ["CoNLLUPlusDataset"]


def _serialized_conllu_plus_sentence_iter(
    fp: TextIO,
    dataset_name: str,
    disable_progress_bar: bool = False,
) -> Iterator[str]:
    dataset_size: int = os.fstat(fp.fileno()).st_size
    with tqdm(
        total=dataset_size,
        unit="B",
        unit_scale=True,
        desc=f"Loading Dataset {dataset_name!r}",
        disable=disable_progress_bar,
    ) as progress_bar:
        global_columns: str = fp.readline()
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
                progress_bar.update(len(sentence.encode("utf-8")))
                yield f"{global_columns}{sentence}"
                sentence_lines = []

        if sentence_lines:
            sentence = "".join(sentence_lines)
            progress_bar.update(len(sentence.encode("utf-8")))
            yield f"{global_columns}{sentence}"


def _parse_conllu_plus(
    fp: TextIO,
    processes: int,
    dataset_name: str,
    disable_progress_bar: bool = False,
    **kwargs: Any,
) -> Iterator[Sentence]:
    kwargs.pop("n_jobs", None)
    kwargs.pop("return_generator", None)

    parallel = Parallel(processes, return_generator=True, **kwargs)
    sentences: Iterator[Sentence] = parallel(
        delayed(from_conllu)(sentence)
        for sentence in _serialized_conllu_plus_sentence_iter(
            fp, dataset_name, disable_progress_bar=disable_progress_bar
        )
    )
    return sentences


class CoNLLUPlusDataset(Dataset[Sentence], Sized):
    """
    Dataset of Flair Sentences parsed from a CoNLL-U Plus file.

    - Universal Dependencies CoNLL-U format: https://universaldependencies.org/format.html
    - Universal Dependencies CoNLL-U Plus format: https://universaldependencies.org/ext-format.html
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        processes: int = 1,
        persist: bool = True,
        disable_progress_bar: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a :class:`CoNLLUPlusDataset` from a CoNLL-U Plus file.
        :param dataset_path: The path to the dataset file
        :param dataset_name: A name for the dataset. Per default, the dataset's path is used.
        :param processes: The number of processes to parse the dataset file
        :param persist: If True, the entire dataset will be converted to Flair Sentences
                        immediately upon initialization and loaded into memory.
                        If the dataset is very large, temporarily disabling Python's generational garbage collector
                        during the initialization may help.
                        If False, only the dataset file is loaded into memory as a string.
                        Calls to `__getitem__` or `__iter__` parse the dataset file on the fly to Flair Sentences.
                        The returned Sentences are not persistent.
        :param disable_progress_bar: Disables the progress bar for loading the dataset
        :param kwargs: Keyword arguments are passed down to the internal use of joblib.
                       Parameters `n_jobs` and `return_generator` will be ignored.
        """
        self._dataset_path = Path(dataset_path)
        self._dataset_name = str(self._dataset_path) if dataset_name is None else dataset_name
        self._processes = processes
        self._persist = persist
        self._kwargs = kwargs

        with self.dataset_path.open("r", encoding="utf-8") as dataset_file:
            self._sentences: Union[tuple[Sentence, ...], tuple[str, ...]]
            if self.is_persistent:
                self._sentences = tuple(
                    _parse_conllu_plus(dataset_file, processes, self._dataset_name, disable_progress_bar, **kwargs)
                )
            else:
                self._sentences = tuple(
                    _serialized_conllu_plus_sentence_iter(dataset_file, self._dataset_name, disable_progress_bar)
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
    def __getitem__(self, item: int) -> Sentence:
        ...

    @overload
    def __getitem__(self, item: slice) -> tuple[Sentence, ...]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[Sentence, tuple[Sentence, ...]]:
        if self.is_persistent:
            assert isinstance(self._sentences[0], Sentence)
            return cast(tuple[Sentence, ...], self._sentences)[item]

        assert isinstance(self._sentences[0], str)
        sentences: tuple[str, ...] = cast(tuple[str, ...], self._sentences)
        return (
            from_conllu(sentences[item])
            if isinstance(item, int)
            else tuple(from_conllu(sentence) for sentence in sentences[item])
        )

    def __iter__(self) -> Iterator[Sentence]:
        if self.is_persistent:
            assert isinstance(self._sentences[0], Sentence)
            yield from cast(tuple[Sentence, ...], self._sentences)
        with self._dataset_path.open("r", encoding="utf-8") as dataset_file:
            yield from _parse_conllu_plus(
                dataset_file,
                self._processes,
                self._dataset_name,
                disable_progress_bar=True,
                return_generator=True,
                **self._kwargs,
            )

    def __len__(self) -> int:
        return len(self._sentences)

    def __repr__(self) -> str:
        attributes: list[tuple[str, str]] = [("dataset_path", repr(self._dataset_path))]
        if self._dataset_name != str(self._dataset_path):
            attributes.append(("dataset_name", repr(self._dataset_name)))
        attributes.append(("persist", repr(self._persist)))
        return f"{type(self).__name__}({', '.join(f'{attribute}={value}' for attribute, value in attributes)})"
