import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from localvoice.config import AudioConfig

_RESPONSE, _RAW = 0, 1


class PlaybackQueue:
    def __init__(
        self, on_response_finished: Callable[[], None], rebuffer_samples: int = 7200
    ) -> None:
        self._on_finished = on_response_finished
        # Anti-stutter gate: when response audio runs dry mid-response, hold output
        # until this many samples are queued (or mark_end) so a slow producer causes
        # one clean pause instead of rapid sliver/gap alternation.
        self._rebuffer = rebuffer_samples
        self._gate_open = False
        self._lock = threading.Lock()
        self._chunks: deque[tuple[np.ndarray, int, int | None]] = deque()  # (samples, kind, tag)
        self._spoken: set[int] = set()
        self._ended = False
        self._fired = False

    def set_rebuffer(self, samples: int) -> None:
        with self._lock:
            self._rebuffer = samples

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
            self._gate_open = False
            self._spoken = set()

    def spoken_tags(self) -> set[int]:
        with self._lock:
            return set(self._spoken)

    def _fill_locked(self, out: np.ndarray) -> int:
        n = len(out)
        i = 0
        while i < n and self._chunks:
            samples, kind, tag = self._chunks[0]
            if kind == _RESPONSE and not self._gate_open:
                queued = sum(len(s) for s, k, _ in self._chunks if k == _RESPONSE)
                if queued >= self._rebuffer or self._ended:
                    self._gate_open = True
                else:
                    break  # hold response audio until enough is buffered
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
        if not response_left and not self._ended:
            self._gate_open = False  # ran dry mid-response: re-arm the gate
        if self._ended and not response_left and not self._fired:
            self._fired = True
            return -i - 1  # sentinel: fire on_response_finished outside the lock
        return i

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self._lock:
            consumed = self._fill_locked(out)
        if consumed < 0:
            self._on_finished()
        return out

    def pull_or_none(self, n: int) -> np.ndarray | None:
        """Feeder-thread variant: None when nothing is consumable (idle or gated),
        so the caller writes nothing and the device buffer stays shallow —
        keeping earcon and speech onset latency low. Zeros inside consumed
        audio (natural pauses) are still delivered, preserving tempo."""
        out = np.zeros(n, np.float32)
        with self._lock:
            consumed = self._fill_locked(out)
        if consumed < 0:
            self._on_finished()
            consumed = -consumed - 1
        return out if consumed > 0 else None


_FEED_BLOCK = 1024  # ~43 ms per write
_DEVICE_BUFFER_S = 0.35  # device-side buffering; must exceed the worst MLX GIL stall


class AudioPlayer:
    def __init__(self, cfg: AudioConfig, on_response_finished: Callable[[], None]) -> None:
        self.queue = PlaybackQueue(
            on_response_finished, rebuffer_samples=int(24000 * cfg.rebuffer_ms / 1000)
        )
        self._cfg = cfg
        self._stream = None
        self._feeder: threading.Thread | None = None
        self._running = False
        self._restart_lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd

        self.queue.set_rebuffer(int(24000 * self._cfg.rebuffer_ms / 1000))
        device = self._cfg.output_device or None
        # Write-mode stream with a deep device-side buffer instead of a Python
        # callback: MLX holds the GIL in bursts up to ~300 ms (prefill/encode), so a
        # Python render callback misses deadlines and PortAudio replays stale
        # buffers — the audible "brrr". CoreAudio drains this buffer in C, immune
        # to the GIL; a stalled feeder just means the buffer runs down silently.
        self._stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype="float32", device=device, latency=_DEVICE_BUFFER_S
        )
        self._stream.start()
        self._running = True
        self._feeder = threading.Thread(target=self._feed, daemon=True, name="audio-feeder")
        self._feeder.start()

    def _feed(self) -> None:
        import sounddevice as sd

        while self._running:
            block = self.queue.pull_or_none(_FEED_BLOCK)
            if block is None:  # idle or gated: keep the device buffer shallow
                time.sleep(0.01)
                continue
            try:
                self._stream.write(block)  # blocks when the device buffer is full
            except (sd.PortAudioError, AttributeError):
                time.sleep(0.005)  # flush()/stop() aborted the stream; retry or exit loop

    def stop(self) -> None:
        self._running = False
        stream = self._stream
        if stream is not None:
            stream.abort()  # unblock a feeder waiting in write()
        if getattr(self, "_feeder", None) is not None:
            self._feeder.join(timeout=1.0)
            self._feeder = None
        if stream is not None:
            stream.close()
            self._stream = None

    def submit(self, samples: np.ndarray, tag: int) -> None:
        self.queue.submit(samples, tag)

    def submit_raw(self, samples: np.ndarray) -> None:
        self.queue.submit_raw(samples)

    def mark_end(self) -> None:
        self.queue.mark_end()

    def flush(self) -> None:
        self.queue.flush()
        stream = self._stream
        if stream is not None:
            with self._restart_lock:
                stream.abort()  # discard already-written audio for instant barge-in
                stream.start()

    def spoken_tags(self) -> set[int]:
        return self.queue.spoken_tags()
