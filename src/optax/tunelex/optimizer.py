from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from .transform import scale_by_prodigy
import optax

def schedule_free_prodigy(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    safeguard_warmup: bool = False,
    state_dtype: Optional[jax.typing.DTypeLike] = None,
) -> base.GradientTransformationExtraArgs:
    prodigy_transform = scale_by_prodigy(
        betas=(0, betas[1]),
        beta3=beta3,
        eps=eps,
        estim_lr0=estim_lr0,
        estim_lr_coef=estim_lr_coef,
        weight_decay=weight_decay,
        safeguard_warmup=safeguard_warmup,
    )

    return optax.contrib.schedule_free(
        prodigy_transform,
        learning_rate=learning_rate,
        b1=betas[0],
        weight_lr_power=weight_lr_power,
        state_dtype=state_dtype,
    )
