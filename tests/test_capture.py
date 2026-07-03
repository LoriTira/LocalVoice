import numpy as np

from localvoice.audio.capture import GatedBuffer, rms


def test_ignores_frames_when_not_armed():
    b = GatedBuffer()
    b.write(np.ones(160, np.float32))
    b.arm()
    b.write(np.full(160, 0.5, np.float32))
    audio = b.disarm()
    assert len(audio) == 160 and np.allclose(audio, 0.5)


def test_arm_clears_previous_audio():
    b = GatedBuffer()
    b.arm()
    b.write(np.ones(10, np.float32))
    b.arm()
    b.write(np.full(5, 0.2, np.float32))
    assert len(b.disarm()) == 5


def test_discard_empties():
    b = GatedBuffer()
    b.arm()
    b.write(np.ones(10, np.float32))
    b.discard()
    b.arm()
    assert len(b.disarm()) == 0


def test_caps_at_max_seconds():
    b = GatedBuffer(max_seconds=0.01, sr=16000)  # 160 samples cap
    b.arm()
    for _ in range(5):
        b.write(np.ones(100, np.float32))
    assert len(b.disarm()) == 160


def test_rms():
    assert rms(np.zeros(100, np.float32)) == 0.0
    assert abs(rms(np.full(100, 0.5, np.float32)) - 0.5) < 1e-6
    assert rms(np.array([], np.float32)) == 0.0


def test_on_level_fires_only_while_armed():
    levels: list[float] = []
    b = GatedBuffer(on_level=levels.append)
    b.write(np.full(160, 0.5, np.float32))
    assert levels == []
    b.arm()
    b.write(np.full(160, 0.5, np.float32))
    assert len(levels) == 1 and abs(levels[0] - 0.5) < 1e-6
    b.disarm()
    b.write(np.full(160, 0.5, np.float32))
    assert len(levels) == 1
