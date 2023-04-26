import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from selfrel.data.conllu import CoNLLUPlusDataset


def build_dataframe(cc_news: CoNLLUPlusDataset) -> pd.DataFrame:
    dataframe_dictionary: dict[str, list[int]] = {"article_id": [], "sentence_id": [], "sentence_length": []}
    for sentence in tqdm(cc_news, desc="Building Dataframe"):
        dataframe_dictionary["article_id"].append(int(sentence.get_label("article_id").value))
        dataframe_dictionary["sentence_id"].append(int(sentence.get_label("sentence_id").value))
        dataframe_dictionary["sentence_length"].append(len(sentence))

    return pd.DataFrame.from_dict(dataframe_dictionary)


def plot_sentence_distribution(dataframe: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid")
    dist: sns.FacetGrid = sns.displot(
        dataframe,
        x="sentence_length",
        kind="hist",
        stat="percent",
        kde=True,
        binwidth=5,
        aspect=2,
    )
    dist.set(xlabel="Sentence Length", ylabel="Percentage of Dataset")
    dist.axes[0][0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.xlim(0)

    plt.savefig(out / "cc_news_sentence_distribution.pdf", dpi=300)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_or_dataframe",
        type=Path,
        help="The path to the CC-News CoNLL-U Plus dataset or to a previously generated dataframe.",
    )
    parser.add_argument("--out", type=Path, default=Path("."), help="The output directory.")
    args: argparse.Namespace = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    dataframe: pd.DataFrame
    if args.dataset_or_dataframe.suffix == ".json":
        dataframe = pd.read_json(args.dataset_or_dataframe)
    else:
        cc_news: CoNLLUPlusDataset = CoNLLUPlusDataset(args.dataset_or_dataframe, persist=False)
        dataframe = build_dataframe(cc_news)
        dataframe.to_json(args.out / "cc_news_sentence_distribution.json")

    plot_sentence_distribution(dataframe, args.out)


if __name__ == "__main__":
    main()
