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


def make(fired: list, rebuffer: int = 0) -> PlaybackQueue:
    # rebuffer=0 disables the anti-stutter gate so accounting/drain tests stay focused;
    # gate behavior is covered by the test_gate_* cases below with rebuffer=300.
    return PlaybackQueue(on_response_finished=lambda: fired.append(True), rebuffer_samples=rebuffer)


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


def test_gate_holds_small_response_chunk_until_threshold():
    q = make([], rebuffer=300)
    q.submit(np.ones(100, np.float32), tag=0)  # below 300-sample test threshold
    assert not q.pull(50).any()  # gated: silence, chunk not consumed
    assert q.spoken_tags() == set()
    q.submit(np.ones(250, np.float32), tag=1)  # 350 queued >= threshold
    assert q.pull(50).any()
    assert 0 in q.spoken_tags()


def test_gate_opens_on_mark_end_for_short_final_audio():
    q = make([], rebuffer=300)
    q.submit(np.ones(100, np.float32), tag=0)
    assert not q.pull(50).any()
    q.mark_end()  # short answer: must play even below threshold
    assert q.pull(50).any()


def test_gate_reengages_after_midresponse_drain():
    fired: list = []
    q = make(fired, rebuffer=300)
    q.submit(np.ones(400, np.float32), tag=0)
    q.pull(400)  # consume fully; response not ended -> gate re-arms
    q.submit(np.ones(100, np.float32), tag=1)  # dribble below threshold
    assert not q.pull(50).any()  # held: no stutter sliver
    q.mark_end()
    assert q.pull(50).any()
    assert fired == []  # not drained yet
    q.pull(200)
    assert fired == [True]


def test_earcons_bypass_gate():
    q = make([], rebuffer=300)
    q.submit_raw(np.full(50, 0.5, np.float32))
    q.submit(np.ones(100, np.float32), tag=0)  # gated response behind the earcon
    out = q.pull(80)
    assert out[:50].any()  # earcon plays immediately
    assert not out[50:].any()  # response still held


def test_flush_rearms_gate():
    q = make([], rebuffer=300)
    q.submit(np.ones(400, np.float32), tag=0)
    q.pull(10)  # gate opened
    q.flush()
    q.submit(np.ones(100, np.float32), tag=1)
    assert not q.pull(50).any()  # gated again after flush


def test_pull_or_none_reports_idle_and_gated():
    q = make([], rebuffer=300)
    assert q.pull_or_none(64) is None  # idle
    q.submit(np.ones(100, np.float32), tag=0)
    assert q.pull_or_none(64) is None  # gated: nothing consumable
    q.mark_end()
    block = q.pull_or_none(64)
    assert block is not None and block.any()


def test_set_rebuffer_rearms_gate_live():
    """set_rebuffer must re-arm gating on a live queue: a queue built with
    gating disabled (rebuffer=0) that later gets set_rebuffer(300) should
    gate a below-threshold submit exactly as if it had been constructed
    with rebuffer=300 -- proving the setter, not just the constructor,
    controls the threshold used by _fill_locked."""
    q = make([], rebuffer=0)  # gate-disabled at construction
    q.set_rebuffer(300)
    q.submit(np.ones(100, np.float32), tag=0)  # below the newly-set 300 threshold
    assert not q.pull(50).any()  # gated: proves set_rebuffer took effect live
    assert q.spoken_tags() == set()
    q.mark_end()
    assert q.pull(50).any()  # short final answer still plays via mark_end


def test_pull_or_none_fires_drain_once():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(30, np.float32), tag=0)
    q.mark_end()
    first = q.pull_or_none(64)  # consumes all 30, zero-padded, fires
    assert first is not None and first[:30].any() and not first[30:].any()
    assert fired == [True]
    assert q.pull_or_none(64) is None
    assert fired == [True]
