from dataclasses import dataclass

import torch
from nnm.cache import Cache


@dataclass(eq=False)
class Session:
    cache: Cache
    policy_version: int
    last_logits: torch.Tensor | None = None
    closed: bool = False

    def close(self):
        self.cache.clear()
        self.last_logits = None
        self.closed = True

    def reset_cache(self):
        self.cache.clear()
        self.last_logits = None
