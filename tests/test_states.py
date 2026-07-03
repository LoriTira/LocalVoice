import pytest

from localvoice.events import Action as A
from localvoice.events import Event
from localvoice.events import EventType as E
from localvoice.events import State as S
from localvoice.states import transition


def ev(t: E, held_ms: int = 0) -> Event:
    return Event(type=t, held_ms=held_ms)


BARGE_IN = (A.CANCEL_PIPELINE, A.TRUNCATE_HISTORY, A.FLUSH_AUDIO, A.EARCON_START, A.START_CAPTURE)
CANCEL = (A.CANCEL_PIPELINE, A.TRUNCATE_HISTORY, A.FLUSH_AUDIO, A.EARCON_CANCEL)

CASES = [
    (S.IDLE, ev(E.PTT_DOWN), S.LISTENING, (A.EARCON_START, A.START_CAPTURE)),
    (S.IDLE, ev(E.PTT_UP, 500), S.IDLE, ()),
    (S.IDLE, ev(E.ESC), S.IDLE, ()),
    (S.IDLE, ev(E.RESPONSE_FINISHED), S.IDLE, ()),
    (S.LISTENING, ev(E.PTT_UP, 500), S.PROCESSING, (A.EARCON_STOP, A.STOP_CAPTURE_AND_RUN)),
    (S.LISTENING, ev(E.PTT_UP, 80), S.IDLE, (A.DISCARD_CAPTURE,)),
    (S.LISTENING, ev(E.ESC), S.IDLE, (A.DISCARD_CAPTURE, A.EARCON_CANCEL)),
    (S.LISTENING, ev(E.PTT_DOWN), S.LISTENING, ()),
    (S.PROCESSING, ev(E.FIRST_AUDIO), S.SPEAKING, ()),
    (S.PROCESSING, ev(E.PTT_DOWN), S.LISTENING, BARGE_IN),
    (S.PROCESSING, ev(E.ESC), S.IDLE, CANCEL),
    (S.PROCESSING, ev(E.RESPONSE_FINISHED), S.IDLE, (A.COMMIT_TURN,)),
    (
        S.PROCESSING,
        ev(E.PIPELINE_ERROR),
        S.IDLE,
        (A.TRUNCATE_HISTORY, A.FLUSH_AUDIO, A.REPORT_ERROR),
    ),
    (S.SPEAKING, ev(E.PTT_DOWN), S.LISTENING, BARGE_IN),
    (S.SPEAKING, ev(E.ESC), S.IDLE, CANCEL),
    (S.SPEAKING, ev(E.RESPONSE_FINISHED), S.IDLE, (A.COMMIT_TURN,)),
    (S.SPEAKING, ev(E.PIPELINE_ERROR), S.IDLE, (A.TRUNCATE_HISTORY, A.FLUSH_AUDIO, A.REPORT_ERROR)),
    (S.SPEAKING, ev(E.FIRST_AUDIO), S.SPEAKING, ()),
]


@pytest.mark.parametrize("state,event,next_state,actions", CASES)
def test_transition(state, event, next_state, actions):
    assert transition(state, event) == (next_state, actions)


def test_debounce_is_configurable():
    assert transition(S.LISTENING, ev(E.PTT_UP, 150), debounce_ms=200)[0] == S.IDLE
    assert transition(S.LISTENING, ev(E.PTT_UP, 150), debounce_ms=100)[0] == S.PROCESSING


def test_every_state_event_pair_is_total():
    for s in S:
        for t in E:
            next_state, actions = transition(s, Event(type=t, held_ms=500))
            assert isinstance(next_state, S) and isinstance(actions, tuple)
