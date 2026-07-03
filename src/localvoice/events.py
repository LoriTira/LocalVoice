from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class EventType(Enum):
    PTT_DOWN = auto()
    PTT_UP = auto()
    ESC = auto()
    FIRST_AUDIO = auto()
    RESPONSE_FINISHED = auto()
    PIPELINE_ERROR = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True)
class Event:
    type: EventType
    held_ms: int = 0
    message: str = ""
    gen: int = -1


class Action(Enum):
    START_CAPTURE = auto()
    STOP_CAPTURE_AND_RUN = auto()
    DISCARD_CAPTURE = auto()
    CANCEL_PIPELINE = auto()
    FLUSH_AUDIO = auto()
    TRUNCATE_HISTORY = auto()
    COMMIT_TURN = auto()
    REPORT_ERROR = auto()
    EARCON_START = auto()
    EARCON_STOP = auto()
    EARCON_CANCEL = auto()
