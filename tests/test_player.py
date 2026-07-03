import numpy as np

from localvoice.audio.earcons import EARCONS, tone
from localvoice.audio.player import PlaybackQueue


def test_tone_shape_and_fade():
    t = tone(880, 60, sr=24000)
    assert t.dtype == np.float32 and t.ndim == 1
    assert len(t) == int(24000 * 0.060)
    assert abs(t[0]) < 1e-4 and abs(t[-1]) < 1e-4
    assert 0.05 < np.abs(t).max() <= 0.2
    assert set(EARCONS) == {"start", "stop", "cancel"}


def make(fired: list) -> PlaybackQueue:
    return PlaybackQueue(on_response_finished=lambda: fired.append(True))


def test_pull_pads_with_zeros_when_empty():
    q = make([])
    out = q.pull(64)
    assert out.shape == (64,) and not out.any()


def test_spoken_tags_track_partial_consumption():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.submit(np.ones(100, np.float32), tag=1)
    q.pull(120)  # all of tag 0, some of tag 1
    assert q.spoken_tags() == {0, 1}
    q2 = make([])
    q2.submit(np.ones(100, np.float32), tag=0)
    q2.submit(np.ones(100, np.float32), tag=1)
    q2.pull(80)
    assert q2.spoken_tags() == {0}


def test_response_finished_fires_once_after_mark_end():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.pull(200)
    assert fired == []  # not marked yet
    q.mark_end()
    q.pull(10)
    q.pull(10)
    assert fired == [True]


def test_flush_drops_audio_and_resets():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.mark_end()
    q.flush()
    assert not q.pull(50).any()
    assert fired == []  # flushed response never "finishes"
    assert q.spoken_tags() == set()


def test_earcon_audio_plays_but_is_not_accounted():
    q = make([])
    q.submit_raw(np.full(50, 0.5, np.float32))
    out = q.pull(50)
    assert out.any()
    assert q.spoken_tags() == set()


def test_earcon_and_response_play_in_submit_order():
    q = make([])
    q.submit_raw(np.full(10, 0.5, np.float32))
    q.submit(np.full(10, 0.9, np.float32), tag=0)
    first = q.pull(10)
    second = q.pull(10)
    assert np.allclose(first, 0.5) and np.allclose(second, 0.9)
