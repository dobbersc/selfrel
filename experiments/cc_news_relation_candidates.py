import argparse
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from selfrel.data import CoNLLUPlusDataset
from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter

if TYPE_CHECKING:
    from flair.data import Label, Sentence, Span


def build_relation_candidates_dataframe(cc_news: CoNLLUPlusDataset) -> pd.DataFrame:
    """
    Returns a dataframe containing all valid entity pair permutations (relation candidates) per sentence.
    The permutations are constructed by a cross-product, excluding the identity entity pair.

    Dataframe columns:
    global_sentence_id | article_id | sentence_id |
    head_entity | head_category | head_score |
    tail_entity | tail_category | tail_score
    """
    dataframe_dictionary: dict[str, list[Union[str, int, float]]] = {
        "global_sentence_id": [],
        "article_id": [],
        "sentence_id": [],
        "head_entity": [],
        "head_category": [],
        "head_score": [],
        "tail_entity": [],
        "tail_category": [],
        "tail_score": [],
    }

    sentence: Sentence
    for sentence in tqdm(cc_news, desc="Building Dataframe"):
        global_sentence_id: int = int(sentence.get_label("global_sentence_id").value)
        article_id: int = int(sentence.get_label("article_id").value)
        sentence_id: int = int(sentence.get_label("sentence_id").value)

        head: Span
        tail: Span
        for head, tail in itertools.product(sentence.get_spans("ner"), repeat=2):
            if head is tail:
                continue

            head_label: Label = head.get_label("ner")
            tail_label: Label = tail.get_label("ner")

            dataframe_dictionary["global_sentence_id"].append(global_sentence_id)
            dataframe_dictionary["article_id"].append(article_id)
            dataframe_dictionary["sentence_id"].append(sentence_id)

            dataframe_dictionary["head_entity"].append(head.text)
            dataframe_dictionary["head_category"].append(head_label.value)
            dataframe_dictionary["head_score"].append(head_label.score)

            dataframe_dictionary["tail_entity"].append(tail.text)
            dataframe_dictionary["tail_category"].append(tail_label.value)
            dataframe_dictionary["tail_score"].append(tail_label.score)

    return pd.DataFrame.from_dict(dataframe_dictionary)


def select_conll04_relation_candidates(relation_candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with relation candidates valid for the ConLL04 task.

    Valid relations:
        - WorkFor:    PER -> ORG
        - Kill:       PER -> PER
        - OrgBasedIn: ORG -> LOC
        - LiveIn:     PER -> LOC
        - LocatedIn:  LOC -> LOC
    """
    work_for = (relation_candidates["head_category"] == "PER") & (relation_candidates["tail_category"] == "ORG")
    kill = (relation_candidates["head_category"] == "PER") & (relation_candidates["tail_category"] == "PER")
    org_based_in = (relation_candidates["head_category"] == "ORG") & (relation_candidates["tail_category"] == "LOC")
    live_in = (relation_candidates["head_category"] == "PER") & (relation_candidates["tail_category"] == "LOC")
    located_in = (relation_candidates["head_category"] == "LOC") & (relation_candidates["tail_category"] == "LOC")

    conll04_relations: np.ndarray = np.select(
        [work_for, kill, org_based_in, live_in, located_in],
        ["WorkFor", "Kill", "OrgBasedIn", "LiveIn", "LocatedIn"],
        default=None,
    )

    conll04_relation_candidates = relation_candidates.assign(relation=pd.Series(conll04_relations, copy=False))
    conll04_relation_candidates = conll04_relation_candidates.dropna(subset="relation")

    return conll04_relation_candidates


def plot_conll04_relation_candidates_distribution(
    conllu04_relation_candidates: pd.DataFrame,
    entity_threshold: float = 0.8,
    out: Optional[Path] = None,
) -> None:
    conllu04_relation_candidates = conllu04_relation_candidates[
        (conllu04_relation_candidates["head_score"] >= entity_threshold)
        & (conllu04_relation_candidates["tail_score"] >= entity_threshold)
    ]

    relation_occurrences = (
        conllu04_relation_candidates.groupby(
            ["head_entity", "head_category", "tail_entity", "tail_category", "relation"],
        )
        .size()
        .reset_index(name="count")
    )

    # Filter relations, where the head entity and tail entity are equal
    relation_occurrences = relation_occurrences[
        relation_occurrences["head_entity"] != relation_occurrences["tail_entity"]
    ]

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        relation_occurrences,
        y="relation",
        x="count",
        showfliers=False,
        boxprops={"facecolor": "none"},
        medianprops={"color": "orange"},
        ax=ax,
    )

    sns.stripplot(
        relation_occurrences.sample(n=1000, random_state=1),
        y="relation",
        x="count",
        jitter=0.25,
        alpha=0.25,
        ax=ax,
    )

    ax.set(xlabel="Relation Candidate Occurrence", ylabel="Relation")

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 8, 32, 128, 512])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    sns.despine(ax=ax)
    plt.tight_layout()

    if out is not None:
        out_figure: Path = out / "cc_news_conll04_relation_candidates_distribution.pdf"
        print(f"Exporting Plot to {out_figure}")
        plt.savefig(out_figure, dpi=300)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset_or_dataframe",
        type=Path,
        help=(
            "Option 1: The path to the CC-News CoNLL-U Plus dataset "
            "including named entity annotations (cc-news-ner.conllup).\n"
            "Option 2: The path to a previously generated dataframe (cc_news_relation_candidates.parquet)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="The output directory. Per default, not output files are generated.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    relation_candidates: pd.DataFrame
    if args.dataset_or_dataframe.suffix == ".parquet":
        print(f"Loading Dataframe from {args.dataset_or_dataframe}")
        relation_candidates = pd.read_parquet(args.dataset_or_dataframe)
    else:
        cc_news: CoNLLUPlusDataset = CoNLLUPlusDataset(args.dataset_or_dataframe, persist=False)
        relation_candidates = build_relation_candidates_dataframe(cc_news)

        if args.out is not None:
            print(f"Exporting Dataframe to {args.out}")
            relation_candidates.to_parquet(args.out / "cc_news_relation_candidates.parquet")

    conllu04_relation_candidates: pd.DataFrame = select_conll04_relation_candidates(relation_candidates)
    plot_conll04_relation_candidates_distribution(conllu04_relation_candidates, out=args.out)


if __name__ == "__main__":
    main()
