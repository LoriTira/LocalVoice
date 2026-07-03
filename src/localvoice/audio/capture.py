import math
import threading

import numpy as np

from localvoice.config import AudioConfig


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))))


class GatedBuffer:
    def __init__(self, max_seconds: float = 300.0, sr: int = 16000) -> None:
        self._max_samples = int(max_seconds * sr)
        self._lock = threading.Lock()
        self._armed = False
        self._frames: list[np.ndarray] = []
        self._total = 0

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._frames = []
            self._total = 0

    def disarm(self) -> np.ndarray:
        with self._lock:
            self._armed = False
            frames, self._frames = self._frames, []
            self._total = 0
        if not frames:
            return np.zeros(0, np.float32)
        return np.concatenate(frames).astype(np.float32)

    def discard(self) -> None:
        with self._lock:
            self._armed = False
            self._frames = []
            self._total = 0

    def write(self, frames: np.ndarray) -> None:
        with self._lock:
            if not self._armed or self._total >= self._max_samples:
                return
            room = self._max_samples - self._total
            chunk = frames[:room]
            self._frames.append(np.asarray(chunk, np.float32))
            self._total += len(chunk)


class MicCapture:
    def __init__(self, cfg: AudioConfig) -> None:
        self.buffer = GatedBuffer()
        self._cfg = cfg
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        device = self._cfg.input_device or None

        def callback(indata, _frames, _time, _status):
            self.buffer.write(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", device=device, callback=callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def arm(self) -> None:
        self.buffer.arm()

    def disarm(self) -> np.ndarray:
        return self.buffer.disarm()

    def discard(self) -> None:
        self.buffer.discard()
