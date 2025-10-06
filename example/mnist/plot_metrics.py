"""Utility for plotting MNIST experiment metrics stored under logs/."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MNIST experiment metrics from CSV logs")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/mnist"),
        help="Directory containing metrics CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/mnist/mnist_metrics.png"),
        help="Path to save the generated plot image (PNG)",
    )
    return parser.parse_args()


def _collect_csv_files(logs_dir: Path) -> list[Path]:
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory '{logs_dir}' does not exist")
    csv_files = sorted(p for p in logs_dir.glob("mnist_eval_metrics_*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No MNIST metrics CSV files found in '{logs_dir}'")
    return csv_files


RUN_PATTERN = re.compile(
    r"^mnist_eval_metrics_"
    r"bs(?P<batch_size>[^_]+)_"
    r"testbs(?P<test_batch_size>[^_]+)_"
    r"epochs(?P<epochs>[^_]+)_"
    r"lr(?P<lr>[^_]+)_"
    r"optimizer(?P<optimizer>.+)_"
    r"seed(?P<seed>[^_]+)$"
)


def _parse_run_metadata(stem: str) -> dict[str, str]:
    match = RUN_PATTERN.match(stem)
    if not match:
        return {}
    metadata = match.groupdict()
    metadata["lr"] = metadata["lr"].replace("p", ".")
    return metadata


def _load_metrics(csv_paths: Iterable[Path]) -> list[pd.DataFrame]:
    dataframes: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df_raw = pd.read_csv(csv_path)
        if df_raw.empty:
            continue
        df = df_raw.copy()
        df["run"] = csv_path.stem
        metadata = _parse_run_metadata(csv_path.stem)
        df.attrs["metadata"] = metadata
        for key, value in metadata.items():
            df[key] = value
        for numeric_column, dtype in (("epoch", int), ("avg_loss", float), ("accuracy", float)):
            if numeric_column in df.columns:
                df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce").astype(dtype)
        dataframes.append(df)
    return dataframes


def plot_metrics(dataframes: list[pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if not dataframes:
        raise ValueError("No metrics dataframes provided for plotting.")

    combined = pd.concat(dataframes, ignore_index=True)

    metadata_keys: set[str] = set()
    for df in dataframes:
        metadata_keys.update(df.attrs.get("metadata", {}).keys())

    metadata_keys = {key for key in metadata_keys if key in combined.columns}
    varying_keys = [key for key in metadata_keys if combined[key].nunique(dropna=False) > 1]
    other_keys = [key for key in sorted(varying_keys) if key != "seed"]

    if "seed" in combined.columns:
        combined["seed"] = combined["seed"].astype(str)

    if other_keys:
        group_columns = other_keys
    elif "optimizer" in combined.columns:
        group_columns = ["optimizer"]
    else:
        group_columns = []

    groupby_columns = group_columns + ["epoch"]

    agg_df = (
        combined.groupby(groupby_columns, dropna=False)
        .agg(
            avg_loss_mean=("avg_loss", "mean"),
            avg_loss_min=("avg_loss", "min"),
            avg_loss_max=("avg_loss", "max"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
        )
        .reset_index()
    )

    if not group_columns:
        agg_df["__label__"] = "metrics"
    else:
        agg_df["__label__"] = agg_df.apply(
            lambda row: ", ".join(f"{col}={row[col]}" for col in group_columns), axis=1
        )

    labels = list(agg_df["__label__"].unique())
    palette = sns.color_palette(n_colors=max(len(labels), 1))
    color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(labels)}

    for label, group in agg_df.groupby("__label__"):
        color = color_map[label]
        group_sorted = group.sort_values("epoch")
        ax_loss.plot(
            group_sorted["epoch"],
            group_sorted["avg_loss_mean"],
            label=label,
            color=color,
            linewidth=2,
        )
        ax_loss.fill_between(
            group_sorted["epoch"],
            group_sorted["avg_loss_min"],
            group_sorted["avg_loss_max"],
            color=color,
            alpha=0.2,
        )

        ax_acc.plot(
            group_sorted["epoch"],
            group_sorted["accuracy_mean"],
            color=color,
            linewidth=2,
        )
        ax_acc.fill_between(
            group_sorted["epoch"],
            group_sorted["accuracy_min"],
            group_sorted["accuracy_max"],
            color=color,
            alpha=0.2,
        )

    ax_loss.set_title("Average Loss per Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Epoch")

    handles, legend_labels = ax_loss.get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=min(len(legend_labels), 3))
        legend = ax_loss.get_legend()
        if legend is not None:
            legend.remove()
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_files = _collect_csv_files(args.logs_dir)
    dfs = _load_metrics(csv_files)
    if not dfs:
        raise ValueError("No non-empty MNIST metrics files found to plot.")
    plot_metrics(dfs, args.output)


if __name__ == "__main__":
    main()


