import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import seaborn as sns
from flair.data import Corpus
from flair.datasets import RE_ENGLISH_CONLL04, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier
from matplotlib import pyplot as plt

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter
from selfrel.utils.inspect_relations import infer_entity_pair_labels


def plot_relationship_distribution(corpus: Corpus, out: Optional[Path] = None) -> None:
    # Prepare data
    entity_pair_labels: Optional[set[tuple[str, str]]] = infer_entity_pair_labels(
        (batch[0] for batch in DataLoader(corpus.train, batch_size=1)),
        relation_label_type="relation",
        entity_label_types="ner",
    )

    model: RelationClassifier = RelationClassifier(
        embeddings=TransformerDocumentEmbeddings(model="distilbert-base-uncased", fine_tune=True),
        label_dictionary=corpus.make_label_dictionary("relation"),
        label_type="relation",
        entity_label_types="ner",
        entity_pair_labels=entity_pair_labels,
        cross_augmentation=True,
        zero_tag_value="No_Relation",
    )

    records: list[tuple[Any, ...]] = []
    for split in ["train", "dev", "test"]:
        dataset = getattr(corpus, split)

        relationship_counter: Counter[str] = Counter(
            marked_sentence.get_label("relation").value
            for marked_sentence in model.transform_dataset(dataset).datapoints
        )

        records.extend((split, relationship, count) for relationship, count in relationship_counter.items())

    relationships: pd.DataFrame = (
        pd.DataFrame.from_records(records, columns=["Split", "Relationship", "Occurrence"])
        .replace(
            {
                "train": "Training",
                "dev": "Validation",
                "test": "Testing",
            }
        )
        .sort_values(["Split", "Relationship"], ignore_index=True)
    )

    relationships = relationships.replace(
        {relationship: relationship.replace("_", "") for relationship in relationships["Relationship"]}
    )

    no_relation_counts: pd.DataFrame = relationships[relationships["Relationship"] == "NoRelation"]
    print(no_relation_counts)
    relationships = relationships[relationships["Relationship"] != "NoRelation"]

    # Plot
    sns.set_theme(style="whitegrid", font_scale=1.75)

    grid: sns.FacetGrid = sns.catplot(
        relationships,
        x="Relationship",
        y="Occurrence",
        col="Split",
        kind="bar",
        col_order=["Training", "Validation", "Testing"],
        height=7,
        aspect=1,
        width=0.75,
    )

    grid.set_xticklabels(rotation=45)

    for ax in grid.axes.flatten():
        ax.set_xlabel("")

    plt.tight_layout()

    if out is not None:
        out_figure: Path = out / "conll04_relationship_distribution.pdf"
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

    conll04: Corpus = RE_ENGLISH_CONLL04()
    plot_relationship_distribution(conll04, out=args.out)


if __name__ == "__main__":
    main()
