from typing import Optional, NamedTuple, Tuple
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree

class Prodigy_with_schedule_free_State(NamedTuple):
    """State for the schedule-free Prodigy direction transform."""

    # original prodigy states
    exp_avg_sq: base.Updates
    grad_sum: base.Updates
    estim_lr: chex.Array
    params0: base.Params
    numerator_weighted: chex.Array
    count: chex.Array

    # schedule-free states
    b1: chex.Array
    weight_sum: chex.Array
    z: base.Params

def scale_by_prodigy_with_schedule_free(
    learning_rate: base.ScalarOrSchedule,
    betas: Tuple[float, float] = (0.9, 0.999),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    safeguard_warmup: bool = False,
    state_dtype: Optional[jnp.dtype] = None,
    weight_lr_power: float = 2.0,
) -> base.GradientTransformationExtraArgs:
    r"""Schedule-free compatible Prodigy direction transform."""

    beta1, beta2 = betas
    if beta3 is None:
        beta3 = beta2**0.5

    def init_fn(params: base.Params) -> Prodigy_with_schedule_free_State:
        params_dtype = optax.tree.dtype(params, 'lowest')
        exp_avg_sq = optax.tree.zeros_like(params)
        grad_sum = optax.tree.zeros_like(params)
        estim_lr = jnp.asarray(estim_lr0, dtype=params_dtype)
        numerator_weighted = jnp.zeros((), dtype=params_dtype)
        count = jnp.zeros((), jnp.int32)

        # schedule-free states
        b1 = jnp.asarray(beta1, dtype=params_dtype)
        weight_sum = jnp.zeros((), dtype=params_dtype)
        if state_dtype is not None:
            params0 = optax.tree.cast(params, dtype=state_dtype)
            z = optax.tree.cast(params, dtype=state_dtype)
        else:
            params0 = params
            z = params
        return Prodigy_with_schedule_free_State(
            exp_avg_sq,
            grad_sum,
            estim_lr,
            params0,
            numerator_weighted,
            count,
            b1,
            weight_sum,
            z,
        )

    def update_fn(
        updates: base.Updates,
        state: Prodigy_with_schedule_free_State,
        params: Optional[base.Params] = None,
        **extra_args,
    ) -> tuple[base.Updates, Prodigy_with_schedule_free_State]:
        del extra_args
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        count = state.count
        count_inc = numerics.safe_increment(count)
        sched = learning_rate(count_inc) if callable(learning_rate) else learning_rate
        estim_lr = state.estim_lr
        params0 = state.params0
        numerator_weighted = state.numerator_weighted
        b1 = state.b1
        z_old = state.z

        one_minus_beta1 = 1.0 - b1
        beta1_inv = 1.0 / b1
        one_minus_beta2 = 1.0 - beta2
        bc = ((1 - beta2**count_inc) ** 0.5) / (1 - beta1**count_inc)
        dlr = jnp.asarray(estim_lr * sched * bc, dtype=estim_lr.dtype)
        estim_lr_ratio = estim_lr / estim_lr0
        grad_sum_coeff = estim_lr_ratio if safeguard_warmup else dlr / estim_lr0
        weight_decay_dlr = weight_decay * dlr

        param_diff = jax.tree.map(lambda p0, p: p0 - p, params0, params)
        numerator_acum = optax.tree.vdot(updates, param_diff)

        dg = jax.tree.map(lambda g: estim_lr * g, updates)
        exp_avg_sq = jax.tree.map(
            lambda eas, dgk: beta2 * eas + one_minus_beta2 * dgk * dgk,
            state.exp_avg_sq,
            dg,
        )
        grad_sum = jax.tree.map(
            lambda sk, dgk: beta3 * sk + grad_sum_coeff * dgk,
            state.grad_sum,
            dg,
        )
        abs_grad_sum = jax.tree.map(jnp.abs, grad_sum)

        numerator_weighted = beta3 * numerator_weighted
        numerator_weighted += estim_lr_ratio * dlr * numerator_acum
        denominator = optax.tree.sum(abs_grad_sum)
        lr_estimate = estim_lr_coef * numerator_weighted / denominator
        estim_lr = jnp.maximum(state.estim_lr, lr_estimate)
        estim_lr_eps = estim_lr * eps

        z = jax.tree.map(
            lambda zi, pi, eas, dgk: jnp.asarray(
                zi
                - weight_decay_dlr * pi
                - dlr * dgk / (jnp.sqrt(eas) + estim_lr_eps),
                dtype=jnp.asarray(zi).dtype,
            ),
            z_old,
            params,
            exp_avg_sq,
            dg,
        )

        weight = estim_lr**weight_lr_power
        next_total_weight = state.weight_sum + weight
        ck = jnp.where(
            jnp.logical_or(jnp.isnan(weight), jnp.isnan(next_total_weight)),
            jnp.full(weight.shape, jnp.nan),
            jnp.nan_to_num(weight / next_total_weight, nan=0.0, posinf=jnp.inf),
        )

        def _combined_param_update(pi, zi_prev, zi_new):
            prev_x = (pi - one_minus_beta1 * zi_prev) * beta1_inv
            x = (1.0 - ck) * prev_x + ck * zi_new
            new_param = b1 * x + one_minus_beta1 * zi_new
            return new_param - pi

        updates = jax.tree.map(_combined_param_update, params, z_old, z)

        new_state = Prodigy_with_schedule_free_State(
            exp_avg_sq,
            grad_sum,
            estim_lr,
            params0,
            numerator_weighted,
            count_inc,
            b1,
            next_total_weight,
            z,
        )

        return updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
