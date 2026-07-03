from collections.abc import Iterator
from typing import Protocol

import numpy as np


class TTSEngine(Protocol):
    def load(self) -> None: ...

    def synthesize(self, text: str) -> Iterator[np.ndarray]: ...
