"""Utility for plotting MNIST experiment metrics stored under logs/."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["run"] = csv_path.stem
        df.attrs["metadata"] = _parse_run_metadata(csv_path.stem)
        dataframes.append(df)
    return dataframes


def plot_metrics(dataframes: list[pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    metadata_list = [df.attrs.get("metadata", {}) for df in dataframes]
    value_sets: dict[str, set[str]] = {}
    for meta in metadata_list:
        for key, value in meta.items():
            value_sets.setdefault(key, set()).add(value)
    varying_keys = {key for key, values in value_sets.items() if len(values) > 1}

    def format_label(meta: dict[str, str]) -> str:
        if not varying_keys:
            return meta.get("optimizer", meta.get("run", "run"))
        if len(varying_keys) == 1:
            key = next(iter(varying_keys))
            return meta.get(key, meta.get("run", "run"))
        pairs = [f"{key}={meta.get(key, '?')}" for key in sorted(varying_keys)]
        return ", ".join(pairs)

    for df in dataframes:
        meta = df.attrs.get("metadata", {})
        display_label = format_label(meta)
        sns.lineplot(data=df, x="epoch", y="avg_loss", marker="o", ax=ax_loss, label=display_label)
        sns.lineplot(data=df, x="epoch", y="accuracy", marker="o", ax=ax_acc, label=display_label)

    ax_loss.set_title("Average Loss per Epoch")
    ax_loss.set_ylabel("Loss")

    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Epoch")

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
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


