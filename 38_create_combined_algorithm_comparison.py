"""
Create Figure 3 for the HIT140 Objective 2 report.

The figure compares 5-fold cross-validated RMSE across regression models
for Task 2.1 and Task 2.2.

Output:
    figure_3_combined_algorithm_comparison.png
"""

import pandas as pd
import matplotlib.pyplot as plt


TASK21_FILE = "task2_1_algorithm_comparison.csv"
TASK22_FILE = "task2_2_algorithm_comparison.csv"
OUTPUT_FILE = "figure_3_combined_algorithm_comparison.png"


def load_task21_results():
    df = pd.read_csv(TASK21_FILE)

    # Use the final Task 2.1 feature set.
    df = df[df["feature_set"] == "D_margin_specific"].copy()

    # Mean baseline CV values come from the final Task 2.1 validation.
    baseline_cv_rmse = 1.358

    rows = [
        ("MLR", float(df.loc[df["algorithm"] == "LinearRegression", "cv_rmse_mean"].iloc[0])),
        ("Decision Tree", float(df.loc[df["algorithm"] == "DecisionTree", "cv_rmse_mean"].iloc[0])),
        ("Random Forest", float(df.loc[df["algorithm"] == "RandomForest", "cv_rmse_mean"].iloc[0])),
        ("Gradient Boosting", float(df.loc[df["algorithm"] == "GradientBoosting", "cv_rmse_mean"].iloc[0])),
        ("Mean baseline", baseline_cv_rmse),
    ]

    return pd.DataFrame(rows, columns=["model", "rmse"])


def load_task22_results():
    df = pd.read_csv(TASK22_FILE)

    df = df[df["evaluation"] == "5-fold grouped CV mean"].copy()

    name_map = {
        "Multiple Linear Regression": "MLR",
        "Decision Tree": "Decision Tree",
        "Random Forest": "Random Forest",
        "Gradient Boosting": "Gradient Boosting",
        "Mean Baseline": "Mean baseline",
    }

    df["model"] = df["model"].map(name_map)

    order = [
        "MLR",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Mean baseline",
    ]

    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")

    return df[["model", "rmse"]].copy()


def draw_panel(ax, df, title):
    # Reverse order so MLR appears at the bottom, matching the report figure.
    df = df.iloc[::-1].reset_index(drop=True)

    y = list(range(len(df)))

    ax.hlines(
        y=y,
        xmin=df["rmse"].min() - 0.03,
        xmax=df["rmse"],
        linewidth=1.2,
    )

    ax.scatter(
        df["rmse"],
        y,
        s=42,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=8)

    ax.set_xlabel("5-fold CV RMSE")
    ax.set_title(title, fontsize=10)

    for x, y_pos in zip(df["rmse"], y):
        ax.text(
            x + 0.012,
            y_pos,
            f"{x:.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(
        df["rmse"].min() - 0.04,
        df["rmse"].max() + 0.08,
    )

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    task21 = load_task21_results()
    task22 = load_task22_results()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.1),
        constrained_layout=True,
    )

    draw_panel(
        axes[0],
        task21,
        "Task 2.1: absolute goal difference",
    )

    draw_panel(
        axes[1],
        task22,
        "Task 2.2: goals scored",
    )

    fig.suptitle(
        "Competing regression algorithms (lower RMSE is better)",
        fontsize=11,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
