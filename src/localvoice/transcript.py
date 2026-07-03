class Transcript:
    # Thread safety: the pipeline thread calls begin_turn/add_clause while the
    # orchestrator thread calls truncate_commit/commit/abort_pending. This is
    # safe under the GIL with benign interleavings, but it is NOT lock-protected
    # — do not add cross-thread iteration over the pending/committed lists
    # without revisiting this.
    def __init__(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._committed: list[dict] = []
        self._pending_user: str | None = None
        self._pending_clauses: list[str] = []

    def begin_turn(self, user_text: str) -> None:
        self._pending_user = user_text
        self._pending_clauses = []

    def add_clause(self, text: str) -> int:
        self._pending_clauses.append(text)
        return len(self._pending_clauses) - 1

    def messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self._system}, *self._committed]
        if self._pending_user is not None:
            msgs.append({"role": "user", "content": self._pending_user})
        return msgs

    def commit(self) -> None:
        self._finish(" ".join(self._pending_clauses))

    def truncate_commit(self, spoken_tags: set[int]) -> None:
        if self._pending_user is None:
            return
        spoken = [c for i, c in enumerate(self._pending_clauses) if i in spoken_tags]
        cut = len(spoken) < len(self._pending_clauses)
        text = " ".join(spoken)
        self._finish((text + " ..." if text else "...") if cut else text)

    def abort_pending(self) -> None:
        self._pending_user = None
        self._pending_clauses = []

    def history(self) -> list[dict]:
        return list(self._committed)

    def _finish(self, assistant_text: str) -> None:
        if self._pending_user is None:
            return
        self._committed.append({"role": "user", "content": self._pending_user})
        self._committed.append({"role": "assistant", "content": assistant_text})
        self.abort_pending()
