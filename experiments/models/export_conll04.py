from pathlib import Path
from typing import Optional, Union

import flair
from flair.datasets import RE_ENGLISH_CONLL04

from selfrel.data.serialization import LabelTypes, export_to_conllu


def export(
    corpus_directory: Union[str, Path], down_sample_train: Optional[float] = None, seed: Optional[int] = None
) -> None:
    corpus_directory = Path(corpus_directory)
    corpus_directory.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        flair.set_seed(seed)

    conll04: RE_ENGLISH_CONLL04 = RE_ENGLISH_CONLL04(
        label_name_map={"Peop": "PER", "Org": "ORG", "Loc": "LOC", "Other": "MISC"}
    )
    if down_sample_train is not None:
        conll04 = conll04.downsample(
            percentage=down_sample_train,
            downsample_train=True,
            downsample_dev=False,
            downsample_test=False,
        )

    for split in ("train", "dev", "test"):
        with (corpus_directory / f"{split}.conllup").open("w", encoding="utf-8") as split_file:
            export_to_conllu(
                split_file,
                getattr(conll04, split),
                global_label_types=LabelTypes(token_level=[], span_level=["ner"]),
            )


def main() -> None:
    for down_sample_train, seed in ((0.01, 42), (0.05, 8), (0.10, 8)):
        export(corpus_directory=f"{down_sample_train:.2f}", down_sample_train=down_sample_train, seed=seed)
    export(corpus_directory="1.00")


if __name__ == "__main__":
    main()
