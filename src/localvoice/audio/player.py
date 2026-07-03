import threading
from collections import deque
from collections.abc import Callable

import numpy as np

from localvoice.config import AudioConfig

_RESPONSE, _RAW = 0, 1


class PlaybackQueue:
    def __init__(self, on_response_finished: Callable[[], None]) -> None:
        self._on_finished = on_response_finished
        self._lock = threading.Lock()
        self._chunks: deque[tuple[np.ndarray, int, int | None]] = deque()  # (samples, kind, tag)
        self._spoken: set[int] = set()
        self._ended = False
        self._fired = False

    def submit(self, samples: np.ndarray, tag: int) -> None:
        with self._lock:
            self._chunks.append((np.asarray(samples, np.float32), _RESPONSE, tag))

    def submit_raw(self, samples: np.ndarray) -> None:
        with self._lock:
            self._chunks.append((np.asarray(samples, np.float32), _RAW, None))

    def mark_end(self) -> None:
        with self._lock:
            self._ended = True

    def flush(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._ended = False
            self._fired = False
            self._spoken = set()

    def spoken_tags(self) -> set[int]:
        with self._lock:
            return set(self._spoken)

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        fire = False
        with self._lock:
            i = 0
            while i < n and self._chunks:
                samples, kind, tag = self._chunks[0]
                if kind == _RESPONSE and tag is not None:
                    self._spoken.add(tag)
                take = min(n - i, len(samples))
                out[i : i + take] = samples[:take]
                if take == len(samples):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = (samples[take:], kind, tag)
                i += take
            response_left = any(k == _RESPONSE for _, k, _ in self._chunks)
            if self._ended and not response_left and not self._fired:
                self._fired = True
                fire = True
        if fire:
            self._on_finished()
        return out


class AudioPlayer:
    def __init__(self, cfg: AudioConfig, on_response_finished: Callable[[], None]) -> None:
        self.queue = PlaybackQueue(on_response_finished)
        self._cfg = cfg
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        device = self._cfg.output_device or None

        def callback(outdata, frames, _time, _status):
            outdata[:, 0] = self.queue.pull(frames)

        self._stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype="float32", device=device, callback=callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def submit(self, samples: np.ndarray, tag: int) -> None:
        self.queue.submit(samples, tag)

    def submit_raw(self, samples: np.ndarray) -> None:
        self.queue.submit_raw(samples)

    def mark_end(self) -> None:
        self.queue.mark_end()

    def flush(self) -> None:
        self.queue.flush()

    def spoken_tags(self) -> set[int]:
        return self.queue.spoken_tags()
