import numpy as np


def tone(freq_hz: float, ms: int, sr: int = 24000, amp: float = 0.15) -> np.ndarray:
    n = int(sr * ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    wave = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    fade = min(int(sr * 0.003), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave


EARCONS = {"start": tone(880, 60), "stop": tone(660, 50), "cancel": tone(220, 90)}
