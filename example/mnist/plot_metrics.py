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
from rich.console import Console
from rich.table import Table


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
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("logs/mnist/mnist_last_epoch_summary.csv"),
        help="Path to save the last-epoch metrics summary (CSV)",
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


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_last_epoch_summary(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metadata_candidates = [
        "run",
        "optimizer",
        "lr",
        "batch_size",
        "test_batch_size",
        "epochs",
        "seed",
    ]

    for df in dataframes:
        if df.empty:
            continue
        if "epoch" in df.columns:
            last_row = df.loc[df["epoch"].idxmax()]
        else:
            last_row = df.iloc[-1]

        record: dict[str, Any] = {}
        for column in metadata_candidates:
            if column in last_row.index:
                record[column] = last_row[column]

        if "epoch" in last_row.index:
            record["epoch"] = last_row["epoch"]

        for column in df.columns:
            if column in record or column in metadata_candidates:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                record[column] = last_row[column]

        if record:
            records.append(record)

    raw_df = pd.DataFrame.from_records(records)
    if raw_df.empty:
        return raw_df

    numeric_columns = [
        column
        for column in raw_df.columns
        if pd.api.types.is_numeric_dtype(raw_df[column]) and column not in {"seed"}
    ]

    metadata_columns = [
        column
        for column in raw_df.columns
        if column not in numeric_columns
    ]

    group_priority = [
        "optimizer",
        "lr",
        "batch_size",
        "test_batch_size",
        "epochs",
    ]
    group_columns = [col for col in group_priority if col in metadata_columns]

    additional_metadata = [
        col
        for col in metadata_columns
        if col not in {"run", "seed"} and col not in group_columns
    ]
    group_columns.extend(additional_metadata)

    temp_group_col = "__group__"
    if not group_columns:
        raw_df[temp_group_col] = 0
        group_columns = [temp_group_col]

    agg_dict: dict[str, list[str]] = {
        column: ["mean", "std"]
        for column in numeric_columns
        if column not in {"seed"}
    }

    grouped = raw_df.groupby(group_columns, dropna=False)
    summary_df = grouped.agg(agg_dict)
    if isinstance(summary_df.columns, pd.MultiIndex):
        summary_df.columns = [f"{col}_{stat}" for col, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    if temp_group_col in summary_df.columns:
        summary_df = summary_df.drop(columns=temp_group_col)

    if "seed" in raw_df.columns and any(col not in {"seed"} for col in metadata_columns):
        summary_df["seed_count"] = grouped["seed"].nunique().values
    elif "seed" in raw_df.columns:
        summary_df["seed_count"] = len(raw_df["seed"].unique())

    ordered_metadata = [
        col
        for col in group_priority + additional_metadata
        if col in summary_df.columns
    ]
    if "seed_count" in summary_df.columns:
        ordered_metadata.append("seed_count")

    metric_columns = [
        col
        for col in summary_df.columns
        if col not in ordered_metadata and col not in {"seed", "run"}
    ]
    key_metric_order = []
    for base in ("avg_loss", "accuracy"):
        for suffix in ("_mean", "_std"):
            name = f"{base}{suffix}"
            if name in metric_columns:
                key_metric_order.append(name)
    remaining_metrics = [col for col in metric_columns if col not in key_metric_order]
    ordered_columns = ordered_metadata + key_metric_order + sorted(remaining_metrics)
    summary_df = summary_df[ordered_columns]

    sort_candidates = [
        col for col in ("optimizer", "lr", "batch_size", "test_batch_size", "epochs")
        if col in summary_df.columns
    ]
    if sort_candidates:
        summary_df = summary_df.sort_values(by=sort_candidates).reset_index(drop=True)
    return summary_df


def _print_rich_summary(summary_df: pd.DataFrame) -> None:
    console = Console()
    if summary_df.empty:
        console.print("[bold yellow]No last-epoch metrics to summarize.[/bold yellow]")
        return

    metric_bases = sorted({col[:-5] for col in summary_df.columns if col.endswith("_mean")})
    std_columns = {col[:-4] for col in summary_df.columns if col.endswith("_std")}

    metadata_columns = [
        col
        for col in summary_df.columns
        if not (col.endswith("_mean") or col.endswith("_std"))
    ]

    table = Table(title="Last Epoch Metrics Summary")
    for column in metadata_columns:
        justify = "right" if pd.api.types.is_numeric_dtype(summary_df[column]) else "left"
        table.add_column(column, justify=justify)

    for base in metric_bases:
        label = f"{base} (+/-)"
        table.add_column(label, justify="right")

    for _, row in summary_df.iterrows():
        cells: list[str] = []
        for column in metadata_columns:
            cells.append(_format_value(row[column]))

        for base in metric_bases:
            mean_value = row.get(f"{base}_mean")
            std_value = row.get(f"{base}_std") if base in std_columns else None
            if pd.isna(mean_value):
                cells.append("-")
            else:
                mean_formatted = _format_value(mean_value)
                if std_value is None or pd.isna(std_value):
                    cells.append(mean_formatted)
                else:
                    std_formatted = _format_value(std_value)
                    cells.append(f"{mean_formatted} +/- {std_formatted}")

        table.add_row(*cells)

    console.print(table)


def _write_summary_csv(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


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
    summary_df = _build_last_epoch_summary(dfs)
    _print_rich_summary(summary_df)
    if not summary_df.empty:
        _write_summary_csv(summary_df, args.summary_csv)


if __name__ == "__main__":
    main()


