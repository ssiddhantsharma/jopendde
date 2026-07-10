"""Context manager that skips OpenDDE's slow scipy-truncnorm weight init.

`OpenDDE(configs)` spends ~95s drawing 656M params from
`scipy.stats.truncnorm.rvs` (via `opendde.model.triangular.layers.
trunc_normal_init_`), every one of which is immediately overwritten by
`load_state_dict`. Inside `with fast_init():` that initializer is swapped for
torch's native `trunc_normal_` (same truncated-normal distribution, fast C++),
cutting construction to a few seconds; the original is restored on exit.

Importing this module is torch-free -- torch/opendde are imported lazily only
when the context manager is entered -- so it doesn't break jopendde's
torch-free-import guarantee.

    from jopendde.fast_init import fast_init
    with fast_init():
        runner = InferenceRunner(configs)   # or OpenDDE(configs)
"""

from __future__ import annotations

import contextlib
import math

_TRUNC_STD = 0.8796256610342398  # scipy truncnorm(a=-2, b=2).std()


@contextlib.contextmanager
def fast_init():
    import torch
    from opendde.model.modules import primitives as P
    from opendde.model.triangular import layers as L

    def fast(weights, scale: float = 1.0, fan: str = "fan_in") -> None:
        f = L._calculate_fan(weights.shape, fan)
        std = math.sqrt(scale / max(1, f)) / _TRUNC_STD
        with torch.no_grad():
            torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-2 * std, b=2 * std)

    orig_L = L.trunc_normal_init_
    orig_P = P.trunc_normal_init_  # the by-name binding imported into primitives
    L.trunc_normal_init_ = fast    # covers lecun_/he_normal_init_ (they call it internally)
    P.trunc_normal_init_ = fast
    try:
        yield
    finally:
        L.trunc_normal_init_ = orig_L
        P.trunc_normal_init_ = orig_P
