import argparse
from pathlib import Path
from typing import Optional

import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter

SELECTED_SEEDS: set[int] = {7}


def plot_results(relation_scores: pd.DataFrame, downsample: float, out: Optional[Path]) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.5)

    grid: sns.FacetGrid = sns.FacetGrid(
        relation_scores,
        col="Class",
        col_wrap=3,
        col_order=[
            "Macro Averaged",
            "Micro Averaged",
            "EMPTY",
            "Kill",
            "LiveIn",
            "EMPTY",
            "LocatedIn",
            "OrgBasedIn",
            "WorkFor",
        ],
        sharex=False,
        height=5,
        aspect=1.2,
    )

    grid.map_dataframe(
        sns.lineplot,
        x="Iteration",
        y="Score",
        hue="Selection Strategy",
        style="Score Type",
        markers=True,
        markersize=10,
        dashes=False,
        linestyle=(0, (1, 2)),
        palette=sns.color_palette()[1:],
    )

    grid.axes[5].legend(*grid.axes[0].get_legend_handles_labels(), loc="center", markerscale=1.75)

    for ax in grid.axes.flatten():
        relationship = ax.get_title().removeprefix("Class = ")
        if relationship in {"Micro Averaged", "Macro Averaged"}:
            ax.set_title(relationship)

        if relationship == "EMPTY":
            # Hide empty plots (hack to align the plots)
            ax.set_title("")
            ax.axis("off")

        # Set grid
        ax.xaxis.grid(False)
        ax.grid(visible=True, which="minor", axis="y", color="gray", linewidth=0.2)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        ax.set_ylim([0.0, 1.04])

    if out is not None:
        out_figure: Path = out / f"multiple_iterations_{downsample:.2f}.pdf"
        print(f"Exporting Plot to {out_figure}")
        plt.savefig(out_figure, dpi=300)

    plt.show()


def plot_results_f1(
    relation_scores: pd.DataFrame, downsample: float, out: Optional[Path], base_scores: pd.DataFrame
) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.5)

    relation_scores = relation_scores[relation_scores["Class"] == "Macro Averaged"]

    grid: sns.FacetGrid = sns.FacetGrid(
        relation_scores,
        col="Score Type",
        col_wrap=3,
        sharex=False,
        height=5,
        aspect=1.1,
    )

    grid.map_dataframe(
        sns.barplot,
        x="Iteration",
        y="Score",
        hue="Selection Strategy",
        palette=sns.color_palette()[1:],
    )

    for ax in grid.axes.flatten():
        relationship = ax.get_title().removeprefix("Score Type = ")
        ax.set_title(relationship)

        # Plot initial model score
        ax.axhline(
            y=base_scores[relationship].iloc(0)[0],
            color=sns.color_palette()[0],
            linestyle="dashed",
            label=f"Initial Model (Z%={int(downsample * 100)}%)",
        )

        # Add bar value labels
        # for bar in ax.containers:
        #     ax.bar_label(bar, label_type="edge")
        #     ax.margins(y=0.1)

        # Set grid
        ax.xaxis.grid(False)
        ax.grid(visible=True, which="minor", axis="y", color="gray", linewidth=0.2)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        ax.set_ylim([0.0, 1.04])

    # axbox = grid.axes[1].get_position()
    # grid.axes[2].legend(loc=(axbox.x0+ 5, axbox.y0 + 0.4), *grid.axes[0].get_legend_handles_labels())
    grid.add_legend(handles=grid.axes[0].get_legend_handles_labels()[0])

    if out is not None:
        out_figure: Path = out / f"multiple_iterations_f1_{downsample:.2f}.pdf"
        print(f"Exporting Plot to {out_figure}")
        plt.savefig(out_figure, dpi=300)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "results",
        type=Path,
        help="The path to the 'results.tsv' file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="The output directory. Per default, no output files are generated.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = pd.read_csv(args.results, sep="\t", comment="#")
    for downsample in [0.01, 0.05, 0.10, 1.00]:
        relation_scores = (
            df[(df["downsample"] == downsample) & (df["iteration"] != 0) & df["seed"].isin(SELECTED_SEEDS)][
                ["class", "selection strategy", "iteration", "precision", "recall", "f1_score"]
            ]
            .rename(
                columns={
                    "class": "Class",
                    "selection strategy": "Selection Strategy",
                    "iteration": "Iteration",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1-Score",
                }
            )
            .replace({relation: relation.replace("_", "") for relation in df["class"].unique()})
            .replace({"micro avg": "Micro Averaged", "macro avg": "Macro Averaged"})
            .replace({"prediction confidence": "Prediction Confidence", "entropy": "Occurrence + Entropy"})
            .fillna(f"Initial Model (Z%={int(downsample * 100)}%)")
            .melt(id_vars=["Class", "Selection Strategy", "Iteration"], value_name="Score", var_name="Score Type")
        )
        relation_scores["Iteration"] = relation_scores["Iteration"].astype(str)

        base_scores = df[
            (df["iteration"] == 0) & (df["downsample"] == downsample) & (df["class"] == "macro avg")
        ].rename(columns={"precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"})

        plot_results_f1(relation_scores, downsample=downsample, out=args.out, base_scores=base_scores)
        plot_results(relation_scores, downsample=downsample, out=args.out)


if __name__ == "__main__":
    main()
