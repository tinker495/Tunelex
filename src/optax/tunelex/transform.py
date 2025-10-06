from typing import Optional, NamedTuple, Tuple
import chex
import jax
import jax.numpy as jnp
import jax.typing
from optax._src import base
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree

class ProdigyState(NamedTuple):
  """State for the schedule-free Prodigy direction transform."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  grad_sum: base.Updates
  params0: base.Params
  estim_lr: chex.Array
  numerator_weighted: chex.Array
  count: chex.Array


def scale_by_prodigy(
    betas: Tuple[float, float] = (0.9, 0.999),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    safeguard_warmup: bool = False,
) -> base.GradientTransformationExtraArgs:
    r"""Schedule-free compatible Prodigy direction transform."""

    beta1, beta2 = betas
    if beta3 is None:
        beta3 = beta2**0.5

    def init_fn(params: base.Params) -> ProdigyState:
        params_dtype = optax.tree.dtype(params, 'lowest')
        exp_avg = optax.tree.zeros_like(params)
        exp_avg_sq = optax.tree.zeros_like(params)
        grad_sum = optax.tree.zeros_like(params)
        params0 = params
        estim_lr = jnp.asarray(estim_lr0, dtype=params_dtype)
        numerator_weighted = jnp.zeros((), dtype=params_dtype)
        count = jnp.zeros((), jnp.int32)
        return ProdigyState(
            exp_avg,
            exp_avg_sq,
            grad_sum,
            params0,
            estim_lr,
            numerator_weighted,
            count,
        )

    def update_fn(
        updates: base.Updates,
        state: ProdigyState,
        params: Optional[base.Params] = None,
        **extra_args,
    ) -> tuple[base.Updates, ProdigyState]:
        del extra_args
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        count_inc = numerics.safe_increment(state.count)
        estim_lr = state.estim_lr
        grad_sum = state.grad_sum
        params0 = state.params0

        dg = jax.tree.map(lambda g: estim_lr * g, updates)
        param_diff = jax.tree.map(lambda p0, p: p0 - p, params0, params)
        numerator_acum = optax.tree.vdot(updates, param_diff)

        exp_avg = jax.tree.map(
            lambda ea, dgk: beta1 * ea + (1.0 - beta1) * dgk, state.exp_avg, dg
        )
        exp_avg_sq = jax.tree.map(
            lambda eas, dgk: beta2 * eas + (1.0 - beta2) * dgk * dgk,
            state.exp_avg_sq,
            dg,
        )

        if safeguard_warmup:
            grad_sum = jax.tree.map(
                lambda sk, dgk: beta3 * sk + estim_lr * dgk / estim_lr0, grad_sum, dg
            )
        else:
            grad_sum = jax.tree.map(
                lambda sk, dgk: beta3 * sk + dgk / estim_lr0, grad_sum, dg
            )

        numerator_weighted = beta3 * state.numerator_weighted
        numerator_weighted += (estim_lr / estim_lr0) * numerator_acum
        denominator = optax.tree.sum(jax.tree.map(jnp.abs, grad_sum))
        lr_estimate = estim_lr_coef * numerator_weighted / denominator
        estim_lr = jnp.maximum(state.estim_lr, lr_estimate)

        bc = ((1 - beta2**count_inc) ** 0.5) / (1 - beta1**count_inc)
        scaled_lr = jnp.asarray(estim_lr * bc, dtype=estim_lr.dtype)

        updates_with_decay = jax.tree.map(
            lambda ea, eas, p: -weight_decay * scaled_lr * p
            - scaled_lr * ea / (jnp.sqrt(eas) + estim_lr * eps),
            exp_avg,
            exp_avg_sq,
            params,
        )

        new_state = ProdigyState(
            exp_avg,
            exp_avg_sq,
            grad_sum,
            params0,
            estim_lr,
            numerator_weighted,
            count_inc,
        )

        return updates_with_decay, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
