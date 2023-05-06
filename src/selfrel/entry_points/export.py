import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import conllu
import datasets
import more_itertools
import syntok.segmenter as segmenter
from joblib import Parallel, delayed
from tqdm import tqdm

from selfrel.preprocessing import preprocess, segment

__all__ = ["get_cc_news", "export_metadata_to_json", "export_cc_news"]


def get_cc_news(dataset_slice: Optional[str] = None) -> datasets.Dataset:
    """Returns the original CC-News dataset from https://huggingface.co/datasets/cc_news with an added ID column."""
    dataset: datasets.Dataset = datasets.load_dataset(
        "cc_news",
        split="train" if dataset_slice is None else f"train[{dataset_slice}]",
    )
    dataset = dataset.add_column("article_id", range(1, len(dataset) + 1))
    return dataset


def export_metadata_to_json(articles: Iterable[dict[str, Any]], fp: TextIO, **kwargs: Any) -> None:
    """
    Exports the CC-News article's metadata as JSON.
    :param articles: An iterable of CC-News articles
    :param fp: A TextIO filepointer to write to
    :param kwargs: Keyword arguments passed to `json.dump`
    """
    metadata: list[dict[str, Any]] = [
        {
            "article_id": article["article_id"],
            "title": article["title"],
            "domain": article["domain"],
            "date": article["date"],
            "description": article["description"],
            "url": article["url"],
            "image_url": article["image_url"],
        }
        for article in articles
    ]
    json.dump(metadata, fp, **kwargs)
    fp.write("\n")


def _sentence_to_conllu_token_list(
    sentence: list[segmenter.Token],
    article_id: int,
    paragraph_id: int,
    sentence_id: int,
) -> conllu.TokenList:
    """Converts a list of Syntok Tokens to a CoNLL-U TokenList, including metadata information."""

    # Filter empty tokens at the end of the sentence
    sentence = sentence.copy()
    if not sentence[-1].value:
        del sentence[-1]

    # `token.spacing` is the prefixed seperator of the current token.
    # We replace the token spacing with a simple whitespace.
    # Because of our previous preprocessing steps and segmenter configuration,
    # `token.spacing` is a combination of " " and "\n".
    text: str = sentence[0].value + "".join(
        f"{' ' if token.spacing else ''}{token.value}" for token in more_itertools.islice_extended(sentence)[1:]
    )

    tokens: list[conllu.Token] = [
        conllu.Token(
            id=token_index,
            form=token.value,
            misc="_" if next_token.spacing else "SpaceAfter=No",
        )
        for token_index, (token, next_token) in enumerate(more_itertools.pairwise(sentence), start=1)
    ]
    # Append last remaining token
    tokens.append(conllu.Token(id=len(sentence), form=sentence[-1].value, misc="SpaceAfter=No"))

    return conllu.TokenList(
        tokens=tokens,
        metadata=conllu.Metadata(article_id=article_id, paragraph_id=paragraph_id, sentence_id=sentence_id, text=text),
    )


def _article_to_conllu(article: str, article_id: int) -> tuple[tuple[str, ...], int]:
    """
    Preprocesses the article and returns its sentences as token lists in the CoNLL-U format.
    It also returns the length of the longest sentence in the article.
    """
    article = preprocess(article, normalization="NFKC")

    sentence_lengths: list[int] = []
    serialized_sentences: list[str] = []
    for paragraph_index, paragraph in enumerate(segment(article), start=1):
        for sentence_index, sentence in enumerate(paragraph, start=1):
            conllu_sentence: conllu.TokenList = _sentence_to_conllu_token_list(
                sentence,
                article_id=article_id,
                paragraph_id=paragraph_index,
                sentence_id=sentence_index,
            )
            sentence_lengths.append(len(conllu_sentence))
            serialized_sentences.append(conllu_sentence.serialize())

    return tuple(serialized_sentences), max(sentence_lengths)


def _cc_news_to_conllu(
    cc_news: datasets.Dataset,
    max_sentence_length: Optional[int],
    processes: int,
    **kwargs: Any,
) -> Iterator[str]:
    """Yields CoNLL-U serialized sentences from the CC-News dataset."""
    kwargs.pop("n_jobs", None)
    kwargs.pop("return_generator", None)

    parallel = Parallel(processes, return_generator=True, **kwargs)
    articles: Iterator[tuple[tuple[str, ...], int]] = parallel(
        delayed(_article_to_conllu)(article["text"], article["article_id"])
        for article in tqdm(cc_news, desc="Submitting Articles", position=0)
    )

    for sentences, longest_sentence_length in tqdm(
        articles,
        desc="Processing Articles",
        total=len(cc_news),
        position=1,
    ):
        if max_sentence_length is None or longest_sentence_length <= max_sentence_length:
            yield from sentences


def export_cc_news(
    out_dir: Union[str, Path],
    export_metadata: bool = True,
    dataset_slice: Optional[str] = None,
    max_sentence_length: Optional[int] = None,
    processes: int = 1,
    **kwargs: Any,
) -> None:
    """See `selfrel export --help`. (Keyword arguments are passed to joblib Parallel.)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cc_news: datasets.Dataset = get_cc_news(dataset_slice)

    # Process and export to CoNLL-U Plus format
    with (out_dir / "cc-news.conllup").open("w", encoding="utf-8") as output_file:
        output_file.write("# global.columns = ID FORM MISC\n")  # Write CoNLL-U Plus header

        for sentence_index, sentence in enumerate(
            _cc_news_to_conllu(cc_news, max_sentence_length=max_sentence_length, processes=processes, **kwargs),
            start=1,
        ):
            output_file.write(f"# global_sentence_id = {sentence_index}\n")  # Minor hack for the global sentence ID
            output_file.write(sentence)

    # Export metadata
    if export_metadata:
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            export_metadata_to_json(tqdm(cc_news, desc="Exporting Metadata"), metadata_file, indent=4, sort_keys=True)
