from localvoice.events import Action as A
from localvoice.events import Event
from localvoice.events import EventType as E
from localvoice.events import State as S

_BARGE_IN = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_START, A.START_CAPTURE)
_CANCEL = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_CANCEL)
_ERROR = (A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.REPORT_ERROR)

_TABLE: dict[tuple[S, E], tuple[S, tuple[A, ...]]] = {
    (S.IDLE, E.PTT_DOWN): (S.LISTENING, (A.EARCON_START, A.START_CAPTURE)),
    (S.LISTENING, E.ESC): (S.IDLE, (A.DISCARD_CAPTURE, A.EARCON_CANCEL)),
    (S.PROCESSING, E.FIRST_AUDIO): (S.SPEAKING, ()),
    (S.PROCESSING, E.PTT_DOWN): (S.LISTENING, _BARGE_IN),
    (S.PROCESSING, E.ESC): (S.IDLE, _CANCEL),
    (S.PROCESSING, E.RESPONSE_FINISHED): (S.IDLE, (A.COMMIT_TURN,)),
    (S.PROCESSING, E.PIPELINE_ERROR): (S.IDLE, _ERROR),
    (S.SPEAKING, E.PTT_DOWN): (S.LISTENING, _BARGE_IN),
    (S.SPEAKING, E.ESC): (S.IDLE, _CANCEL),
    (S.SPEAKING, E.RESPONSE_FINISHED): (S.IDLE, (A.COMMIT_TURN,)),
    (S.SPEAKING, E.PIPELINE_ERROR): (S.IDLE, _ERROR),
}


def transition(state: S, event: Event, debounce_ms: int = 120) -> tuple[S, tuple[A, ...]]:
    if state is S.LISTENING and event.type is E.PTT_UP:
        if event.held_ms < debounce_ms:
            return S.IDLE, (A.DISCARD_CAPTURE,)
        return S.PROCESSING, (A.EARCON_STOP, A.STOP_CAPTURE_AND_RUN)
    return _TABLE.get((state, event.type), (state, ()))
