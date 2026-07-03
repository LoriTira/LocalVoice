import numpy as np

from localvoice.config import SttConfig


class WhisperMlxEngine:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg

    def load(self) -> None:
        self.transcribe(np.zeros(3200, np.float32), 16000)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        if sample_rate != 16000:
            raise ValueError(f"whisper path expects 16 kHz mono audio, got {sample_rate} Hz")
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._cfg.model)
        if "text" not in result:
            raise RuntimeError(f"unexpected mlx-whisper result shape: {sorted(result)}")
        return str(result["text"]).strip()
