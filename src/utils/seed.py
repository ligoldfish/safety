from __future__ import annotations

import random

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
