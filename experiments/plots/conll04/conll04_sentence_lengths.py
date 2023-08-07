import argparse
import itertools
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from flair.data import Corpus, Sentence
from flair.datasets import RE_ENGLISH_CONLL04
from matplotlib.figure import figaspect

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter


def build_sentence_lengths_dataframe(corpus: Corpus[Sentence]) -> pd.DataFrame:
    assert corpus.train is not None
    assert corpus.dev is not None
    assert corpus.test is not None
    return pd.DataFrame(
        {
            "sentence_length": [  # type: ignore[var-annotated]
                len(sentence)
                for sentence in itertools.chain(corpus.train, corpus.dev, corpus.test)  # type: ignore[arg-type]
            ]
        }
    )


def plot_sentence_length_distribution(dataframe: pd.DataFrame, out: Optional[Path]) -> None:
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath, siunitx}")

    sns.set_theme(style="whitegrid")

    # noinspection PyTypeChecker
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=figaspect(0.5), height_ratios=(0.15, 0.85))

    sns.boxplot(dataframe, x="sentence_length", ax=ax_box)
    ax_box.set(xlabel="")
    sns.despine(ax=ax_box)

    sns.histplot(
        dataframe,
        x="sentence_length",
        stat="percent",
        discrete=True,
        binwidth=1,
        ax=ax_hist,
    )
    ax_hist.set(xlabel="Sentence Length", ylabel="Percentage of Dataset")
    ax_hist.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    sns.despine(ax=ax_hist)

    plt.xlim(0)

    text: str = (
        r"\begin{alignat*}{2}"
        r"&\#\text{Sentences:}\, &&\num{" + str(len(dataframe.index)) + r"} \\"
        r"&\#\text{Training:}\, &&\num{910} \\"
        r"&\#\text{Validation:}\, &&\num{243} \\"
        r"&\#\text{Testing:}\, &&\num{288}"
        r"\end{alignat*}"
    )
    ax_hist.text(
        100,
        2.5,
        text,
        ha="center",
        va="center",
        bbox={"boxstyle": "square, pad=1", "fc": "white", "ec": "black"},
    )

    if out is not None:
        out_figure: Path = out / "conll04_sentence_distribution.pdf"
        print(f"Exporting Plot to {out_figure}")
        plt.savefig(out_figure, dpi=300)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="The output directory. Per default, no output files are generated.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    conll04: Corpus[Sentence] = RE_ENGLISH_CONLL04()
    plot_sentence_length_distribution(build_sentence_lengths_dataframe(conll04), args.out)


if __name__ == "__main__":
    main()
