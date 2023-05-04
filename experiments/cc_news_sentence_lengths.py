import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from flair.data import Sentence
from matplotlib.figure import figaspect
from tqdm import tqdm

from selfrel.data.conllu import CoNLLUPlusDataset
from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter


def build_sentence_lengths_dataframe(cc_news: CoNLLUPlusDataset) -> pd.DataFrame:
    """
    Returns a dataframe containing all sentences with their token-level sentence length.

    Dataframe columns: article_id | sentence_id | sentence_length
    """
    dataframe_dictionary: dict[str, list[int]] = {"article_id": [], "sentence_id": [], "sentence_length": []}

    sentence: Sentence
    for sentence in tqdm(cc_news, desc="Building Dataframe"):
        dataframe_dictionary["article_id"].append(int(sentence.get_label("article_id").value))
        dataframe_dictionary["sentence_id"].append(int(sentence.get_label("sentence_id").value))
        dataframe_dictionary["sentence_length"].append(len(sentence))

    return pd.DataFrame.from_dict(dataframe_dictionary)


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
    ax_hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    sns.despine(ax=ax_hist)

    plt.xlim(0)

    text: str = (
        r"\begin{alignat*}{2}"
        r"&\#\text{Articles:}\, &&\num{" + str(dataframe["article_id"].nunique()) + r"} \\"
        r"&\#\text{Sentences:}\, &&\num{" + str(len(dataframe.index)) + r"}"
        r"\end{alignat*}"
    )
    ax_hist.text(
        90,
        2.5,
        text,
        ha="center",
        va="center",
        bbox=dict(boxstyle="square, pad=1", fc="white", ec="black"),
    )

    if out is not None:
        out_figure: Path = out / "cc_news_sentence_distribution.pdf"
        print(f"Exporting Plot to {out_figure}")
        plt.savefig(out_figure, dpi=300)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset_or_dataframe",
        type=Path,
        help=(
            "Option 1: The path to the CC-News CoNLL-U Plus dataset (cc-news.conllup).\n"
            "Option 2: The path to a previously generated dataframe (cc_news_sentence_lengths.parquet)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="The output directory. Per default, not output files are generated."
    )
    args: argparse.Namespace = parser.parse_args()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    sentence_lengths: pd.DataFrame
    if args.dataset_or_dataframe.suffix == ".parquet":
        print(f"Loading Dataframe from {args.dataset_or_dataframe}")
        sentence_lengths = pd.read_parquet(args.dataset_or_dataframe)
    else:
        cc_news: CoNLLUPlusDataset = CoNLLUPlusDataset(args.dataset_or_dataframe, persist=False)
        sentence_lengths = build_sentence_lengths_dataframe(cc_news)

        if args.out is not None:
            print(f"Exporting Dataframe to {args.out}")
            sentence_lengths.to_parquet(args.out / "cc_news_sentence_lengths.parquet")

    plot_sentence_length_distribution(sentence_lengths, args.out)


if __name__ == "__main__":
    main()
