from collections.abc import Iterator

import numpy as np

from localvoice.config import TtsConfig


class KokoroMlxEngine:
    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg
        self._model = None

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self._cfg.model)
        for _ in self.synthesize("Ready."):
            pass

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for segment in self._model.generate(
            text=text, voice=self._cfg.voice, speed=self._cfg.speed, lang_code="a"
        ):
            audio = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
            if audio.size:
                yield audio
