import argparse
import contextlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, TextIO, TypeVar, Union

import conllu
import datasets
import importlib_resources
import ray
from flair.data import Label, Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
from flair.tokenization import SegtokTokenizer
from ray.actor import ActorHandle
from ray.util import ActorPool
from tqdm import tqdm

T = TypeVar("T")
V = TypeVar("V")


def get_cc_news(dataset_slice: Optional[str]) -> datasets.Dataset:
    """Returns the CC-News dataset from https://huggingface.co/datasets/cc_news with an added ID column."""
    dataset: datasets.Dataset = datasets.load_dataset(
        "cc_news", split="train" if dataset_slice is None else f"train[{dataset_slice}]"
    )
    dataset = dataset.add_column("article_id", range(1, len(dataset) + 1))
    return dataset


def export_metadata_to_json(articles: Iterable[dict[str, Any]], fp: TextIO, **kwargs: Any) -> None:
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


def map_unordered(actor_pool: ActorPool, fn: Callable[[ActorHandle, V], T], values: Iterable[V]) -> Iterator[T]:
    """
    Typed variant of ray's `ActorPool.map_unordered`
    https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.html
    """
    while actor_pool.has_next():
        with contextlib.suppress(TimeoutError):
            actor_pool.get_next_unordered(timeout=0)

    for value in values:
        actor_pool.submit(fn, value)
    while actor_pool.has_next():
        yield actor_pool.get_next_unordered()


def map_ordered(actor_pool: ActorPool, fn: Callable[[ActorHandle, V], T], values: Iterable[V]) -> Iterator[T]:
    """
    Typed variant of ray's `ActorPool.map_ordered`
    https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.html
    """
    while actor_pool.has_next():
        with contextlib.suppress(TimeoutError):
            actor_pool.get_next(timeout=0, ignore_if_timedout=True)

    for value in values:
        actor_pool.submit(fn, value)
    while actor_pool.has_next():
        yield actor_pool.get_next()


@ray.remote
class Pipeline:
    """
    The annotation pipeline.
    text -> tokenization + sentence splitting -> ner -> conllu serialization
    """

    def __init__(self, model_path: Union[str, Path], label_type: str, batch_size: int) -> None:
        self._sentence_splitter: SegtokSentenceSplitter = SegtokSentenceSplitter(tokenizer=SegtokTokenizer())
        self._model_path: Union[str, Path] = model_path
        self._model: SequenceTagger = SequenceTagger.load(model_path)
        self._label_type = label_type
        self._batch_size = batch_size

    def _build_conllu_token_list(self, sentence: Sentence, sentence_id: int, article_id: int) -> conllu.TokenList:
        text: str = "".join(f"{token.text}{' ' if token.whitespace_after else ''}" for token in sentence.tokens)

        tokens: list[conllu.Token] = []
        for token_id, token in enumerate(sentence.tokens, start=1):
            label: Label = token.get_label(self._label_type)
            tokens.append(
                conllu.Token(
                    id=token_id,
                    form=token.form,
                    ner=label.value,
                    ner_score=label.score,
                    misc="_" if token.whitespace_after else "SpaceAfter=No",
                )
            )

        return conllu.TokenList(
            tokens=tokens,
            metadata=conllu.Metadata(article_id=article_id, sentence_id=sentence_id, text=text),
        )

    def process(self, article_id: int, text: str) -> tuple[str, ...]:
        """Processes the input text through the pipeline."""
        sentences: list[Sentence] = self._sentence_splitter.split(text)
        self._model.predict(
            sentences,
            label_name=self._label_type,
            mini_batch_size=self._batch_size,
            force_token_predictions=True,
        )

        conllu_sentences: tuple[str, ...] = tuple(
            self._build_conllu_token_list(
                sentence,
                sentence_id=sentence_id,
                article_id=article_id,
            ).serialize()
            for sentence_id, sentence in enumerate(sentences, start=1)
        )
        return conllu_sentences

    def __repr__(self) -> str:
        attributes: list[str] = [
            f"model_path={self._model_path!r}",
            f"label_type={self._label_type!r}",
            f"batch_size={self._batch_size!r}",
        ]
        return f"{type(self).__name__}({', '.join(attributes)}"


def main() -> None:
    # Parse arguments
    description: str = (
        importlib_resources.files("selfrel.resources.entry_descriptions")
        .joinpath("cc_news.txt")
        .read_text(encoding="utf-8")
    )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=".",
        help="The output directory for the processed articles.",
    )
    # noinspection PyTypeChecker
    parser.add_argument(
        "--ordered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, the output is guaranteed to be ordered by the article ID.",
    )
    # noinspection PyTypeChecker
    parser.add_argument(
        "--export-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, the article's metadata will be exported to '<OUT>/metadata.json'.",
    )
    parser.add_argument(
        "--slice",
        default=None,
        help=(
            "A huggingface datasets slice, e.g. ':100', '25%%:75%%'. "  # Double '%' to escape formatted string
            "Reference: https://huggingface.co/docs/datasets/v1.11.0/splits.html"
        ),
    )
    parser.add_argument(
        "--model-path",
        default="flair/ner-english-large",
        help="A model path or huggingface ID for a Flair NER model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The model prediction batch size.",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help=(
            "An optional address to an existing Ray cluster. "
            "If the address is empty, a new Ray cluster will be started."
        ),
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="The number of Ray actors to start.",
    )
    parser.add_argument(
        "--num-cpus",
        type=float,
        default=None,
        help="The number of CPUs required for each Ray actor.",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=1.0,
        help="The number of GPUs required for each Ray actor.",
    )

    args: argparse.Namespace = parser.parse_args()

    out_dir: Path = args.out
    ordered: bool = args.ordered
    export_metadata: bool = args.export_metadata
    dataset_slice: Optional[str] = args.slice
    model_path: str = args.model_path
    batch_size: int = args.batch_size
    ray_address: Optional[str] = args.ray_address
    num_actors: int = args.num_actors
    num_cpus: Optional[float] = args.num_cpus
    num_gpus: Optional[float] = args.num_gpus

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ray cluster
    ray.init(address=ray_address)

    # Initialize pipeline actor pool
    pipeline_actors: list[ActorHandle] = [
        Pipeline.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(  # type: ignore[attr-defined]
            model_path, label_type="ner", batch_size=batch_size
        )
        for _ in range(num_actors)
    ]
    pipeline_pool: ActorPool = ActorPool(pipeline_actors)

    def remote_pipeline(pipeline_actor: ActorHandle, article: dict[str, Any]) -> tuple[str, ...]:
        result: tuple[str, ...] = pipeline_actor.process.remote(article_id=article["article_id"], text=article["text"])
        return result

    # Load dataset and process
    cc_news: datasets.Dataset = get_cc_news(dataset_slice)
    with (out_dir / "cc-news.conllup").open("w", encoding="utf-8") as output_file:
        output_file.write("# global.columns = ID FORM NER:FLAIR NER_SCORE:FLAIR MISC\n")

        map_function = map_ordered if ordered else map_unordered
        for conllu_sentences in tqdm(
            map_function(pipeline_pool, remote_pipeline, tqdm(cc_news, desc="Submitting to Actor Pool")),
            desc="Processing Articles",
            total=len(cc_news),
        ):
            # Export sentences in CoNLL-U Plus format
            output_file.writelines(conllu_sentences)

    # Export metadata
    if export_metadata:
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            export_metadata_to_json(tqdm(cc_news, desc="Exporting Metadata"), metadata_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
