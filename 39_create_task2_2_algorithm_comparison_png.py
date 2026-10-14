"""
Create a standalone Task 2.2 algorithm-comparison PNG from the final CSV.

Input:
    task2_2_algorithm_comparison.csv

Output:
    figure_task2_2_algorithm_comparison.png

This script uses the 5-fold grouped CV mean rows, because those are the
comparison values discussed in the report and appendix.
"""

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "task2_2_algorithm_comparison.csv"
OUTPUT_FILE = "figure_task2_2_algorithm_comparison.png"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Keep only the grouped cross-validation summary rows.
    df = df[df["evaluation"] == "5-fold grouped CV mean"].copy()

    # Standardise display names.
    name_map = {
        "Multiple Linear Regression": "Multiple Linear Regression",
        "Random Forest": "Random Forest Regressor",
        "Decision Tree": "Decision Tree Regressor",
        "Gradient Boosting": "Gradient Boosting Regressor",
        "Mean Baseline": "Mean baseline",
    }

    df["label"] = df["model"].map(name_map)

    # Order from best RMSE (lowest) to worst RMSE (highest).
    display_order = [
        "Multiple Linear Regression",
        "Random Forest Regressor",
        "Decision Tree Regressor",
        "Gradient Boosting Regressor",
        "Mean baseline",
    ]

    df["label"] = pd.Categorical(df["label"], categories=display_order, ordered=True)
    df = df.sort_values("label")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    y = list(range(len(df)))
    xmin = df["rmse"].min() - 0.02

    ax.hlines(
        y=y,
        xmin=xmin,
        xmax=df["rmse"],
        linewidth=1.3,
    )

    ax.scatter(
        df["rmse"],
        y,
        s=55,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()

    ax.set_xlabel("5-fold grouped CV RMSE (lower is better)")
    ax.set_title("Task 2.2 regression-model comparison")

    for x, y_pos in zip(df["rmse"], y):
        ax.text(
            x + 0.006,
            y_pos,
            f"{x:.3f}",
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
