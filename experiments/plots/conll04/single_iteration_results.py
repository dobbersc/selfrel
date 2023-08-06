import argparse
from pathlib import Path
from typing import Optional

import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter

INITIAL_MODEL_SEEDS: set[int] = {8, 42}


def plot_results(df: pd.DataFrame, downsample: float, out: Optional[Path]) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.5)

    relation_scores = (
        df[
            (df["downsample"] == downsample)
            & ((df["iteration"] == 0) | ((~df["seed"].isin(INITIAL_MODEL_SEEDS)) & (df["iteration"] == 1)))
        ][["class", "selection strategy", "precision", "recall", "f1_score"]]
        .rename(
            columns={
                "class": "Class",
                "selection strategy": "Selection Strategy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
            }
        )
        .replace({relation: relation.replace("_", "") for relation in df["class"].unique()})
        .replace({"micro avg": "Micro Averaged", "macro avg": "Macro Averaged"})
        .replace({"prediction confidence": "Prediction Confidence", "entropy": "Occurrence + Entropy"})
        .fillna(f"Initial Model (Z%={int(downsample * 100)}%)")
        .melt(id_vars=["Class", "Selection Strategy"], value_name="Score", var_name="Score Type")
    )

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
        sns.barplot,
        x="Score Type",
        y="Score",
        hue="Selection Strategy",
        errorbar="sd",
        errwidth=2,
        capsize=0.1,
        palette=sns.color_palette(),
    )

    grid.axes[5].legend(*grid.axes[0].get_legend_handles_labels(), loc="center")

    for ax in grid.axes.flatten():
        relationship = ax.get_title().removeprefix("Class = ")
        if relationship in {"Micro Averaged", "Macro Averaged"}:
            ax.set_title(relationship)

        if relationship == "EMPTY":
            # Hide empty plots (hack to align the plots)
            ax.set_title("")
            ax.axis("off")

        # for i, bar in enumerate(ax.patches):
        #    ax.text(
        #        x=bar.get_x() + bar.get_width() / 2.,
        #        y=np.nan_to_num(ax.lines[i*3].get_ydata(), nan=bar.get_height())[1] + 0.02,
        #        s=f"{bar.get_height():.2%}",
        #        fontsize="xx-small",
        #        rotation=45,
        #        color="black",
        #        ha='center',
        #        va='bottom'
        #    )

        # Set minor grid
        ax.grid(visible=True, which="minor", axis="y", color="gray", linewidth=0.2)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        ax.set_xlabel("")
        ax.set_ylim([0.0, 1.04])

    if out is not None:
        out_figure: Path = out / f"single_iteration_{downsample:.2f}.pdf"
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
        plot_results(df, downsample=downsample, out=args.out)


if __name__ == "__main__":
    main()
