from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, TextIO
import csv
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
import flax.struct
import urllib.error
from torchvision import datasets
from torch.utils.data import DataLoader

try:  # Support both package and script execution
    from .optimizers import make_optimizer, optimizer_names, requires_schedule_free_eval
except ImportError:  # pragma: no cover - fallback when run as a script
    from optimizers import make_optimizer, optimizer_names, requires_schedule_free_eval
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


MEAN = 0.1307
STD = 0.3081


def _safe_float_str(value: float) -> str:
    return f"{value}".replace(".", "p")


def _preferred_targets(dataset: Any) -> Any:
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "labels"):
        return dataset.labels
    raise AttributeError("Dataset does not expose targets or labels attributes")


def load_mnist_splits(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        train_ds = datasets.MNIST(str(data_dir), train=True, download=True)
        test_ds = datasets.MNIST(str(data_dir), train=False, download=True)
    except (RuntimeError, urllib.error.URLError) as err:
        raise RuntimeError(
            "Failed to download MNIST. Place the extracted dataset files under "
            f"{data_dir} (e.g. train-images-idx3-ubyte) or run with --data-dir pointing "
            "to an existing download."
        ) from err

    def _process_images(ds: Any) -> np.ndarray:
        images = ds.data.numpy().astype(np.float32) / 255.0
        images = (images - MEAN) / STD
        return np.expand_dims(images, axis=-1)

    def _process_labels(ds: Any) -> np.ndarray:
        return _preferred_targets(ds).numpy().astype(np.int32)

    return (
        _process_images(train_ds),
        _process_labels(train_ds),
        _process_images(test_ds),
        _process_labels(test_ds),
    )


class ArrayDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "image": self._images[idx],
            "label": self._labels[idx],
        }


def collate_batch(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    images = np.stack([item["image"] for item in batch], axis=0)
    labels = np.array([item["label"] for item in batch], dtype=np.int32)
    return {"image": images, "label": labels}


def _coerce_estim_lr(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_estim_lr(opt_state: Any) -> float | None:
    seen: set[int] = set()

    def _search(node: Any) -> float | None:
        if node is None:
            return None
        node_id = id(node)
        if node_id in seen:
            return None
        seen.add(node_id)

        if hasattr(node, "estim_lr"):
            estim_lr = _coerce_estim_lr(getattr(node, "estim_lr"))
            if estim_lr is not None:
                return estim_lr

        if isinstance(node, Mapping):
            for value in node.values():
                result = _search(value)
                if result is not None:
                    return result

        if isinstance(node, tuple):
            if hasattr(node, "_fields"):
                for field in node._fields:
                    result = _search(getattr(node, field))
                    if result is not None:
                        return result
            for value in node:
                result = _search(value)
                if result is not None:
                    return result

        if isinstance(node, list):
            for value in node:
                result = _search(value)
                if result is not None:
                    return result

        if hasattr(node, "__dict__"):
            for value in vars(node).values():
                result = _search(value)
                if result is not None:
                    return result

        if hasattr(node, "_asdict"):
            for value in node._asdict().values():
                result = _search(value)
                if result is not None:
                    return result

        return None

    return _search(opt_state)


@flax.struct.dataclass
class TrainState:
    step: int
    apply_fn: Any = flax.struct.field(pytree_node=False)
    params: Any
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    dropout_rng: jax.Array

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Any,
        params: Any,
        tx: optax.GradientTransformation,
        dropout_rng: jax.Array,
    ) -> "TrainState":
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dropout_rng=dropout_rng,
        )

    def apply_gradients(self, *, grads: Any, dropout_rng: jax.Array) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            dropout_rng=dropout_rng,
        )


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.25)(x, deterministic=not training)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not training)
        x = nn.Dense(features=10)(x)
        return x


def create_train_state(rng: jax.Array, learning_rate: float, model: CNN, *, optimizer_name: str) -> TrainState:
    params_rng, dropout_rng = jax.random.split(rng)
    dummy_batch = jnp.ones((1, 28, 28, 1), dtype=jnp.float32)
    variables = model.init({"params": params_rng, "dropout": dropout_rng}, dummy_batch, training=True)
    params = variables["params"]
    tx = make_optimizer(optimizer_name, learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, dropout_rng=dropout_rng)


@partial(jax.jit, static_argnums=2)
def train_step(state: TrainState, batch: dict[str, np.ndarray], model: CNN) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    images = jnp.asarray(batch["image"], dtype=jnp.float32)
    labels = jnp.asarray(batch["label"], dtype=jnp.int32)
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits = model.apply({"params": params}, images, training=True, rngs={"dropout": dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    metrics = {"loss": loss, "accuracy": accuracy}
    return new_state, metrics


@partial(jax.jit, static_argnums=2)
def eval_step(params: Any, batch: dict[str, np.ndarray], model: CNN) -> tuple[jnp.ndarray, jnp.ndarray]:
    images = jnp.asarray(batch["image"], dtype=jnp.float32)
    labels = jnp.asarray(batch["label"], dtype=jnp.int32)
    logits = model.apply({"params": params}, images, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    correct = jnp.argmax(logits, axis=-1) == labels
    return loss.sum(), correct.sum()


def train_epoch(
    state: TrainState,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    epoch: int,
    args: argparse.Namespace,
    model: CNN,
) -> TrainState:
    train_dataset = ArrayDataset(train_images, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    steps_per_epoch = train_dataset.__len__() // args.batch_size
    train_examples = steps_per_epoch * args.batch_size

    format_loss = lambda value: f"{float(value):.6f}"  # noqa: E731
    format_acc = lambda value: f"{float(value) * 100:.2f}%"  # noqa: E731
    format_estim_lr = lambda value: f"{float(value):.6f}"  # noqa: E731

    progress = Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]}", justify="right"),
        TextColumn("Acc: {task.fields[acc]}", justify="right"),
        TextColumn("Est. LR: {task.fields[estim_lr]}", justify="right"),
    )

    metrics_table = Table.grid()
    metrics_table.add_column("Metric", style="bold cyan")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("Epoch", str(epoch))
    metrics_table.add_row("Loss", "-")
    metrics_table.add_row("Accuracy", "-")
    metrics_table.add_row("Estim LR", "-")

    progress_task = progress.add_task(
        "train",
        total=steps_per_epoch,
        epoch=epoch,
        total_epochs=args.epochs,
        loss="-",
        acc="-",
        estim_lr="-",
    )

    console = Console()

    with Live(
        Panel(
            Group(
                progress,
                Panel(metrics_table, title="Latest Metrics", border_style="green"),
            ),
            title=f"Training Epoch {epoch}",
            border_style="magenta",
        ),
        console=console,
        refresh_per_second=8,
    ):
        for step, batch in enumerate(train_loader, start=1):
            state, metrics = train_step(state, batch, model)
            loss_str = format_loss(metrics["loss"])
            acc_str = format_acc(metrics["accuracy"])
            estim_lr_value = _extract_estim_lr(state.opt_state)
            estim_lr_str = format_estim_lr(estim_lr_value) if estim_lr_value is not None else "-"
            metrics_table.columns[1]._cells[1] = loss_str
            metrics_table.columns[1]._cells[2] = acc_str
            metrics_table.columns[1]._cells[3] = estim_lr_str
            progress.update(
                progress_task,
                advance=1,
                loss=loss_str,
                acc=acc_str,
                estim_lr=estim_lr_str,
            )
            if args.dry_run and step >= 1:
                break
    return state


def evaluate(
    state: TrainState,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    model: CNN,
    *,
    batch_size: int,
    optimizer_name: str,
) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total_examples = test_images.shape[0]
    test_dataset = ArrayDataset(test_images, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    progress = Progress(
        TextColumn("[bold blue]Evaluating[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("Est. LR: {task.fields[estim_lr]}", justify="right"),
    )
    metrics_panel: Panel | None = None
    console = Console()

    estim_lr_value = _extract_estim_lr(state.opt_state)
    estim_lr_str = f"{estim_lr_value:.6f}" if estim_lr_value is not None else "-"

    with Live(console=console, refresh_per_second=8) as live:
        eval_task = progress.add_task("eval", total=math.ceil(total_examples / batch_size), estim_lr=estim_lr_str)
        live.update(Panel(progress, title="Evaluation", border_style="cyan"))
        for batch in test_loader:
            params_for_eval = state.params
            if requires_schedule_free_eval(optimizer_name):
                params_for_eval = optax.contrib.schedule_free_eval_params(state.opt_state, state.params)
            batch_loss, batch_correct = eval_step(params_for_eval, batch, model)
            total_loss += float(batch_loss)
            total_correct += int(batch_correct)
            progress.update(eval_task, advance=1)
            live.update(Panel(progress, title="Evaluation", border_style="cyan"))

    avg_loss = total_loss / total_examples
    accuracy = 100.0 * total_correct / total_examples

    summary_table = Table(title="Evaluation Summary", show_lines=True)
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Average Loss", f"{avg_loss:.4f}")
    summary_table.add_row("Accuracy", f"{total_correct}/{total_examples} ({accuracy:.2f}%)")
    if estim_lr_value is not None:
        summary_table.add_row("Estim LR", estim_lr_str)

    console = Console()
    console.print(summary_table)

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "total_correct": float(total_correct),
        "total_examples": float(total_examples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX/Optax MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="test batch size")
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=optimizer_names(),
        help="optimizer to use (configurable via optimizers.py)",
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="run a single logging step and exit")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed")
    parser.add_argument("--log-interval", type=int, default=200, metavar="N", help="logging interval in steps")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mnist"),
        help="directory (relative to repo by default) to store MNIST files",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="persist trained parameters to mnist_cnn.msgpack",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("mnist_cnn.msgpack"),
        help="output path for serialized parameters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    model = CNN()
    state = create_train_state(rng, args.lr, model, optimizer_name=args.optimizer)

    train_images, train_labels, test_images, test_labels = load_mnist_splits(data_dir)

    logs_path = Path("logs/mnist")
    logs_path.mkdir(parents=True, exist_ok=True)
    metrics_filename = (
        "mnist_eval_metrics_"
        f"bs{args.batch_size}_"
        f"testbs{args.test_batch_size}_"
        f"epochs{args.epochs}_"
        f"lr{_safe_float_str(args.lr)}_"
        f"optimizer{args.optimizer}_"
        f"seed{args.seed}.csv"
    )
    metrics_file = logs_path / metrics_filename
    is_new_file = not metrics_file.exists()

    if metrics_file.exists():
        metrics_file.unlink()
        is_new_file = True

    with metrics_file.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["epoch", "avg_loss", "accuracy", "total_correct", "total_examples"])

        for epoch in range(1, args.epochs + 1):
            state = train_epoch(state, train_images, train_labels, epoch, args, model)
            eval_metrics = evaluate(
                state,
                test_images,
                test_labels,
                model,
                batch_size=args.test_batch_size,
                optimizer_name=args.optimizer,
            )
            writer.writerow(
                [
                    epoch,
                    f"{eval_metrics['avg_loss']:.6f}",
                    f"{eval_metrics['accuracy']:.2f}",
                    int(eval_metrics["total_correct"]),
                    int(eval_metrics["total_examples"]),
                ]
            )
            csvfile.flush()

    if args.save_model:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = serialization.to_bytes(state.params)
        args.model_path.write_bytes(payload)
        print(f"Saved parameters to {args.model_path}")


if __name__ == "__main__":
    main()
