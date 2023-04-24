import os
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, Optional, TextIO, Union, cast, overload

import conllu
from flair.data import Sentence, Token
from joblib import Parallel, delayed
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

__all__ = ["CoNLLUPlusDataset"]


def _serialized_conllu_plus_sentence_iter(fp: TextIO, disable_progress_bar: bool = False) -> Iterator[str]:
    dataset_size: int = os.fstat(fp.fileno()).st_size
    with tqdm(
        total=dataset_size, unit="B", unit_scale=True, desc="Loading Dataset", disable=disable_progress_bar
    ) as progress_bar:
        global_columns: str = fp.readline()
        if not global_columns.startswith("# global.columns"):
            raise ValueError("Missing CoNLL-U Plus required 'global.columns'")

        sentence_lines: list[str] = []
        while line := fp.readline():
            sentence_lines.append(line)

            # There must be exactly one blank line after every sentence, including the last sentence in the file.
            # Empty sentences are not allowed. (From https://universaldependencies.org/format.html)
            # Therefore, a blank line ends the current CoNLL-U sentence.
            if line == "\n":
                sentence: str = "".join(sentence_lines)
                progress_bar.update(len(sentence.encode("utf-8")))
                yield f"{global_columns}{sentence}"
                sentence_lines = []


def _infer_whitespace_after(token: conllu.Token) -> int:
    misc: Optional[dict[str, str]] = token.get("misc")
    if misc is not None:
        return 0 if misc.get("SpaceAfter") == "No" else 1
    return 1


def _parse_serialized_conllu_plus_sentence(serialized_conllu_sentence: str) -> Sentence:
    conllu_sentences: conllu.SentenceList = conllu.parse(serialized_conllu_sentence)
    assert len(conllu_sentences) == 1
    conllu_sentence: conllu.TokenList = conllu_sentences[0]

    flair_tokens: list[Token] = [
        Token(
            text=conllu_token["form"],
            whitespace_after=_infer_whitespace_after(conllu_token),
        )
        for conllu_token in conllu_sentence
    ]

    flair_sentence: Sentence = Sentence(flair_tokens)

    # Add metadata as sentence label
    for key, value in conllu_sentence.metadata.items():
        if key == "global.columns":
            continue
        flair_sentence.add_label(key, value)

    return flair_sentence


def _parse_conllu_plus(
    fp: TextIO, processes: int, disable_progress_bar: bool = False, **kwargs: Any
) -> Iterator[Sentence]:
    kwargs.pop("n_jobs", None)
    kwargs.pop("return_generator", None)

    parallel = Parallel(processes, return_generator=True, **kwargs)
    sentences: Iterator[Sentence] = parallel(
        delayed(_parse_serialized_conllu_plus_sentence)(sentence)
        for sentence in _serialized_conllu_plus_sentence_iter(fp, disable_progress_bar=disable_progress_bar)
    )
    return sentences


class CoNLLUPlusDataset(Dataset[Sentence], Sized):
    """
    Dataset of Flair Sentences parsed from a CoNLL-U Plus file.

    - Universal Dependencies CoNLL-U format: https://universaldependencies.org/format.html
    - Universal Dependencies CoNLL-U Plus format: https://universaldependencies.org/ext-format.html
    """

    def __init__(self, dataset_path: Union[str, Path], processes: int = 1, persist: bool = True, **kwargs: Any) -> None:
        """
        Initializes a :class:`CoNLLUPlusDataset` from a CoNLL-U Plus file.
        :param dataset_path: The path to the dataset file
        :param processes: The number of processes to parse the dataset file
        :param persist: If True, the entire dataset will be converted to Flair Sentences
                        immediately upon initialization and loaded into memory.
                        If the dataset is very large, temporarily disabling Python's generational garbage collector
                        during the initialization may help.
                        If False, only the dataset file is loaded into memory as a string.
                        Calls to `__getitem__` or `__iter__` parse the dataset file on the fly to Flair Sentences.
                        The returned Sentences are not persistent.
        :param kwargs: Keyword arguments are passed down to the internal use of joblib.
                       Parameters `n_jobs` and `return_generator` will be ignored.
        """
        self._dataset_path = Path(dataset_path)
        self._processes = processes
        self._persist = persist
        self._kwargs = kwargs

        with self.dataset_path.open("r", encoding="utf-8") as dataset_file:
            self._sentences: Union[tuple[Sentence, ...], tuple[str, ...]]
            if self.is_persistent:
                self._sentences = tuple(_parse_conllu_plus(dataset_file, processes, **kwargs))
            else:
                self._sentences = tuple(_serialized_conllu_plus_sentence_iter(dataset_file))

        if not self._sentences:
            raise ValueError("Empty datasets are not supported")

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

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
            _parse_serialized_conllu_plus_sentence(sentences[item])
            if isinstance(item, int)
            else tuple(_parse_serialized_conllu_plus_sentence(sentence) for sentence in sentences[item])
        )

    def __iter__(self) -> Iterator[Sentence]:
        if self.is_persistent:
            return iter(self._sentences)
        with self._dataset_path.open("r", encoding="utf-8") as dataset_file:
            yield from _parse_conllu_plus(
                dataset_file, self._processes, disable_progress_bar=True, return_generator=True, **self._kwargs
            )

    def __len__(self) -> int:
        return len(self._sentences)
