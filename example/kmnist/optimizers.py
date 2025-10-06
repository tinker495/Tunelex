"""Factories for Optax optimizers used by the KMNIST example."""

from __future__ import annotations

from typing import Callable, Dict

import optax

OptimizerFactory = Callable[[float], optax.GradientTransformation]


_OPTIMIZERS: Dict[str, OptimizerFactory] = {
    "adamw": lambda lr: optax.adamw(learning_rate=lr),
    "prodigy": lambda lr: optax.contrib.prodigy(),
    "schedule_free_adamw": lambda lr: optax.contrib.schedule_free_adamw(learning_rate=lr),
    # "schedule_free_sgd": lambda lr: optax.contrib.schedule_free_sgd(learning_rate=lr), # don't measure sgd
    # "sgd": lambda lr: optax.sgd(learning_rate=lr),
    # "momentum": lambda lr: optax.sgd(learning_rate=lr, momentum=0.9),
}


def optimizer_names() -> list[str]:
    """Return the list of supported optimizer identifiers."""

    return sorted(_OPTIMIZERS.keys())


def make_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
    """Build the requested optimizer.

    Args:
        name: Key referencing an optimizer factory defined in ``_OPTIMIZERS``.
        learning_rate: Learning rate forwarded to the factory.

    Raises:
        KeyError: If ``name`` is not registered.

    Returns:
        optax.GradientTransformation: Initialised optimizer instance.
    """

    try:
        factory = _OPTIMIZERS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown optimizer '{name}'. Available: {optimizer_names()}") from exc
    return factory(learning_rate)


def requires_schedule_free_eval(name: str) -> bool:
    """Return True if the optimizer requires schedule-free evaluation parameters."""

    return "schedule_free" in name.lower()



