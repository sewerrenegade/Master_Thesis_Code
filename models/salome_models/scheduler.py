from typing import List, Optional

import numpy as np

CLIP_LAM_MAX = 1e4



class ConstantScheduler:
    def __init__(
        self,
        lam: float,
    ) -> None:

        self.lam = lam

    def __call__(self, loss: float) -> float:

        return self.lam
