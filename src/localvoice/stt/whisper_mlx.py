import numpy as np

from localvoice.config import SttConfig


class WhisperMlxEngine:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg

    def load(self) -> None:
        self.transcribe(np.zeros(3200, np.float32), 16000)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        assert sample_rate == 16000, "whisper path expects 16 kHz mono"
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._cfg.model)
        return str(result["text"]).strip()
