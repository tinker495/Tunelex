from typing import Optional, Tuple

from optax._src import base
from .transform import scale_by_prodigy_with_schedule_free

def schedule_free_prodigy(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: Tuple[float, float] = (0.95, 0.99),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    safeguard_warmup: bool = False,
) -> base.GradientTransformationExtraArgs:
    return scale_by_prodigy_with_schedule_free(
        learning_rate=learning_rate,
        betas=betas,
        beta3=beta3,
        eps=eps,
        estim_lr0=estim_lr0,
        estim_lr_coef=estim_lr_coef,
        weight_decay=weight_decay,
        safeguard_warmup=safeguard_warmup,
        weight_lr_power=weight_lr_power,
    )