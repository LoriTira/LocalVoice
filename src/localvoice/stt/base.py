from typing import Protocol

import numpy as np


class STTEngine(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
