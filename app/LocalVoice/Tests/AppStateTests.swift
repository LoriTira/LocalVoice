import XCTest
@testable import LocalVoice

@MainActor
final class AppStateTests: XCTestCase {
    // MARK: - Startup ordering

    /// `.spawned` → connecting; `.engine(.ready)` → stores config/schema, ready,
    /// protocolMismatch false for version 1; `.engine(.enginesReady)` → engines_ready.
    func testFullStartupOrdering() {
        let state = AppState()
        XCTAssertEqual(state.connection, "connecting") // initial value, before any event

        state.reduce(.spawned)
        XCTAssertEqual(state.connection, "connecting")

        let config: JSONValue = .object(["llm": .object(["think": .bool(false)])])
        let schema = [makeSchemaField(key: "llm.think")]
        state.reduce(.engine(.ready(version: 1, config: config, schema: schema)))
        XCTAssertEqual(state.connection, "ready")
        XCTAssertEqual(state.config, config)
        XCTAssertEqual(state.schema, schema)
        XCTAssertFalse(state.protocolMismatch)

        state.reduce(.engine(.enginesReady))
        XCTAssertEqual(state.connection, "engines_ready")
    }

    /// `.exited` → dead; `.respawning` → connecting again, mirroring a fresh boot attempt.
    func testExitedThenRespawningTransitionsConnection() {
        let state = AppState()
        state.reduce(.spawned)
        state.reduce(.engine(.ready(version: 1, config: .null, schema: [])))
        state.reduce(.engine(.enginesReady))
        XCTAssertEqual(state.connection, "engines_ready")

        state.reduce(.exited(code: 1))
        XCTAssertEqual(state.connection, "dead")

        state.reduce(.respawning(attempt: 1, delaySeconds: 1))
        XCTAssertEqual(state.connection, "connecting")
    }

    /// A `ready` carrying a version other than 1 is a protocol mismatch, surfaced as a flag
    /// rather than a crash — the app can still render a "please update" banner from this.
    func testVersionTwoReadySetsProtocolMismatch() {
        let state = AppState()
        state.reduce(.engine(.ready(version: 2, config: .null, schema: [])))
        XCTAssertTrue(state.protocolMismatch)
        XCTAssertEqual(state.connection, "ready") // connection still transitions; mismatch is a separate flag
    }

    /// The nominal version=1 path must leave protocolMismatch false, including after a prior
    /// mismatched boot recovers (defensive: mismatch isn't sticky across a fresh `ready`).
    func testVersionOneReadyClearsProtocolMismatch() {
        let state = AppState()
        state.reduce(.engine(.ready(version: 2, config: .null, schema: [])))
        XCTAssertTrue(state.protocolMismatch)

        state.reduce(.engine(.ready(version: 1, config: .null, schema: [])))
        XCTAssertFalse(state.protocolMismatch)
    }

    // MARK: - Turn assembly

    /// `.userText` appends a user turn AND an empty assistant turn in one shot — the two-turn
    /// pair the brief specifies, so the UI has an assistant bubble to stream clauses into
    /// immediately, before any clause has arrived.
    func testUserTextAppendsUserTurnAndEmptyAssistantTurn() {
        let state = AppState()
        state.reduce(.engine(.userText("What's the capital of Australia?")))

        XCTAssertEqual(state.turns.count, 2)
        XCTAssertEqual(state.turns[0].role, "user")
        XCTAssertEqual(state.turns[0].text, "What's the capital of Australia?")
        XCTAssertEqual(state.turns[1].role, "assistant")
        XCTAssertEqual(state.turns[1].text, "")
    }

    /// Two `assistantClause`s space-join into the trailing assistant turn's text —
    /// the exact example string from the brief.
    func testTwoClauseTurnAssemblesIntoOneAssistantTurnText() {
        let state = AppState()
        state.reduce(.engine(.userText("hi")))
        state.reduce(.engine(.assistantClause("Hello there.")))
        state.reduce(.engine(.assistantClause("More words.")))

        XCTAssertEqual(state.turns.count, 2)
        XCTAssertEqual(state.turns.last?.text, "Hello there. More words.")
    }

    /// Defensive case: an `assistantClause` with no turns at all yet (no preceding `userText`)
    /// must not crash and must create an assistant turn to hold it.
    func testAssistantClauseWithNoExistingTurnsCreatesOne() {
        let state = AppState()
        state.reduce(.engine(.assistantClause("orphan clause")))

        XCTAssertEqual(state.turns.count, 1)
        XCTAssertEqual(state.turns[0].role, "assistant")
        XCTAssertEqual(state.turns[0].text, "orphan clause")
    }

    /// `.reasoning` sets/appends the last assistant turn's `reasoning` field, independent of
    /// (and without disturbing) the spoken `text`. Each `reasoning` event is one whole think
    /// block (the engine emits one per completed block, not per token), so two of them —
    /// e.g. reasoning from two tool rounds — are separated by a blank line, not run together.
    func testReasoningAttachesToLastAssistantTurnWithBlankLineBetweenBlocks() {
        let state = AppState()
        state.reduce(.engine(.userText("what's the capital of Australia?")))
        state.reduce(.engine(.reasoning("First I should recall Australian geography.")))
        state.reduce(.engine(.reasoning("It is Canberra, not Sydney.")))
        state.reduce(.engine(.assistantClause("Canberra.")))

        XCTAssertEqual(state.turns.count, 2)
        let assistantTurn = state.turns[1]
        XCTAssertEqual(
            assistantTurn.reasoning,
            "First I should recall Australian geography.\n\nIt is Canberra, not Sydney.",
            "multi-round reasoning blocks are joined with a blank line, not concatenated bare"
        )
        XCTAssertEqual(assistantTurn.text, "Canberra.")
    }

    /// The very first reasoning block on a turn is stored verbatim — the blank-line separator
    /// is only inserted *between* blocks, never prepended to the first one.
    func testFirstReasoningBlockHasNoLeadingSeparator() {
        let state = AppState()
        state.reduce(.engine(.userText("hi")))
        state.reduce(.engine(.reasoning("Only one thought here.")))

        XCTAssertEqual(state.turns.last?.reasoning, "Only one thought here.")
    }

    /// Barge-in: a fresh `userText` arriving while an assistant turn is mid-stream starts a new
    /// user/assistant pair; the prior assistant turn's text is left exactly as it was, not
    /// merged, cleared, or retroactively marked.
    func testBargeInStartsFreshPairAndLeavesPriorAssistantTextAsWas() {
        let state = AppState()
        state.reduce(.engine(.userText("first question")))
        state.reduce(.engine(.assistantClause("partial answer")))

        state.reduce(.engine(.userText("second question, interrupting")))

        XCTAssertEqual(state.turns.count, 4)
        XCTAssertEqual(state.turns[0].role, "user")
        XCTAssertEqual(state.turns[0].text, "first question")
        XCTAssertEqual(state.turns[1].role, "assistant")
        XCTAssertEqual(state.turns[1].text, "partial answer") // untouched by the barge-in
        XCTAssertEqual(state.turns[2].role, "user")
        XCTAssertEqual(state.turns[2].text, "second question, interrupting")
        XCTAssertEqual(state.turns[3].role, "assistant")
        XCTAssertEqual(state.turns[3].text, "") // fresh empty pair for the new turn
    }

    /// Turn ids are a monotonically increasing counter across the whole session, not reset
    /// per pair or derived from array position.
    func testTurnIdsAreMonotonicallyIncreasing() {
        let state = AppState()
        state.reduce(.engine(.userText("one")))
        state.reduce(.engine(.userText("two")))

        let ids = state.turns.map(\.id)
        XCTAssertEqual(ids, ids.sorted())
        XCTAssertEqual(Set(ids).count, ids.count, "ids must be unique, got \(ids)")
        XCTAssertEqual(ids.count, 4)
    }

    // MARK: - Assistant state / level

    /// `.state` mirrors directly into `assistantState`, and repeats are idempotent
    /// (level-triggered per docs/gui.md, not edge-triggered) — applying the same value twice
    /// must not throw or change anything else.
    func testStateMirrorsAndRepeatsAreIdempotent() {
        let state = AppState()
        state.reduce(.engine(.state("listening")))
        XCTAssertEqual(state.assistantState, "listening")

        state.reduce(.engine(.state("listening"))) // repeat, level-triggered
        XCTAssertEqual(state.assistantState, "listening")

        state.reduce(.engine(.state("processing")))
        XCTAssertEqual(state.assistantState, "processing")
    }

    /// `.level` sets micLevel while listening; a `.state` transition away from "listening"
    /// resets micLevel to 0 (the mic isn't armed outside that state, per docs/gui.md).
    func testLevelResetsOnStateChangeAwayFromListening() {
        let state = AppState()
        state.reduce(.engine(.state("listening")))
        state.reduce(.engine(.level(0.42)))
        XCTAssertEqual(state.micLevel, 0.42)

        state.reduce(.engine(.state("processing")))
        XCTAssertEqual(state.micLevel, 0)
    }

    /// A repeated `.state("listening")` (idempotent level-trigger) must NOT reset micLevel —
    /// only an actual change away from "listening" does.
    func testLevelNotResetByRepeatedListeningState() {
        let state = AppState()
        state.reduce(.engine(.state("listening")))
        state.reduce(.engine(.level(0.7)))

        state.reduce(.engine(.state("listening"))) // same value repeated
        XCTAssertEqual(state.micLevel, 0.7, "repeated same-value state must not reset level")
    }

    /// Entering "listening" itself must not clobber a level that arrives after it (ordering
    /// sanity — level events only make sense while listening per the protocol, but the reducer
    /// should not proactively zero on entry, only on exit).
    func testLevelSurvivesAcrossMultipleLevelEventsWhileListening() {
        let state = AppState()
        state.reduce(.engine(.state("listening")))
        state.reduce(.engine(.level(0.1)))
        state.reduce(.engine(.level(0.9)))
        XCTAssertEqual(state.micLevel, 0.9)
    }

    // MARK: - Latency

    func testTurnDoneSetsLastLatency() {
        let state = AppState()
        let latency = Latency(stt: 0.18, ttft: 0.24, firstClause: 0.31, ttsFirst: 0.09, total: 0.82)
        state.reduce(.engine(.turnDone(latency)))
        XCTAssertEqual(state.lastLatency, latency)
    }

    // MARK: - Load progress

    /// `.loadProgress` tracks in-flight phases keyed by engine name; the key is removed once
    /// that engine reports phase "done", per the brief ("remove key when p == done").
    func testLoadProgressTracksThenRemovesOnDone() {
        let state = AppState()
        state.reduce(.engine(.loadProgress(engine: "stt", phase: "start", seconds: nil)))
        XCTAssertEqual(state.loading["stt"], "start")

        state.reduce(.engine(.loadProgress(engine: "llm", phase: "start", seconds: nil)))
        XCTAssertEqual(state.loading["stt"], "start")
        XCTAssertEqual(state.loading["llm"], "start")

        state.reduce(.engine(.loadProgress(engine: "stt", phase: "done", seconds: 1.2)))
        XCTAssertNil(state.loading["stt"], "done must remove the key, not just update it")
        XCTAssertEqual(state.loading["llm"], "start")
    }

    // MARK: - Downloads

    /// `.downloadProgress` tracks pct by repo while in flight; `done: true` removes the key
    /// entirely rather than leaving a stale 100%-and-done entry sitting in the dictionary.
    func testDownloadProgressAddsThenRemovesOnDone() {
        let state = AppState()
        state.reduce(.engine(.downloadProgress(repo: "mlx-community/whisper-tiny", pct: 47.3, done: false)))
        XCTAssertEqual(state.downloads["mlx-community/whisper-tiny"] ?? nil, 47.3)

        state.reduce(.engine(.downloadProgress(repo: "mlx-community/whisper-tiny", pct: 100.0, done: true)))
        XCTAssertNil(state.downloads["mlx-community/whisper-tiny"], "done must remove the key")
    }

    /// `pct: nil` (unknown size) is a valid indeterminate in-flight state, distinct from the
    /// key being absent entirely — `downloads` is `[String: Double?]` specifically so a caller
    /// can tell "no download" apart from "download in progress, size unknown."
    func testDownloadProgressWithNilPctIsIndeterminateNotAbsent() {
        let state = AppState()
        state.reduce(.engine(.downloadProgress(repo: "mlx-community/whisper-tiny", pct: nil, done: false)))

        XCTAssertTrue(state.downloads.keys.contains("mlx-community/whisper-tiny"))
        XCTAssertEqual(state.downloads["mlx-community/whisper-tiny"] ?? .some(-1), .none)
    }

    /// Two concurrent downloads track independently by repo key.
    func testMultipleConcurrentDownloadsTrackIndependently() {
        let state = AppState()
        state.reduce(.engine(.downloadProgress(repo: "repo/a", pct: 10, done: false)))
        state.reduce(.engine(.downloadProgress(repo: "repo/b", pct: 20, done: false)))
        XCTAssertEqual(state.downloads["repo/a"] ?? nil, 10)
        XCTAssertEqual(state.downloads["repo/b"] ?? nil, 20)

        state.reduce(.engine(.downloadProgress(repo: "repo/a", pct: 100, done: true)))
        XCTAssertNil(state.downloads["repo/a"])
        XCTAssertEqual(state.downloads["repo/b"] ?? nil, 20)
    }

    // MARK: - Models / config / error

    func testModelsEventReplacesModelsList() {
        let state = AppState()
        let models = [InstalledModel(id: "a/b", path: "/x", kind: "mlx", sizeGb: 18.6)]
        state.reduce(.engine(.models(models)))
        XCTAssertEqual(state.models, models)
    }

    /// `.configApplied` updates `config` and clears any standing banner — a successful config
    /// apply is evidence the engine recovered, so a stale error shouldn't linger on screen.
    func testConfigAppliedUpdatesConfigAndClearsBanner() {
        let state = AppState()
        state.reduce(.engine(.error("engines still loading")))
        XCTAssertEqual(state.banner, "engines still loading")

        let newConfig: JSONValue = .object(["llm": .object(["think": .bool(true)])])
        state.reduce(.engine(.configApplied(config: newConfig, reloaded: ["llm"])))
        XCTAssertEqual(state.config, newConfig)
        XCTAssertNil(state.banner)
    }

    /// `.error` sets the banner to the message verbatim.
    func testErrorSetsBanner() {
        let state = AppState()
        state.reduce(.engine(.error("bad json: unexpected end of input")))
        XCTAssertEqual(state.banner, "bad json: unexpected end of input")
    }

    // MARK: - Tool activity

    /// `.toolCall` sets `toolActivity` to its `summary`; a subsequent
    /// `.toolResult(ok: true)` does NOT clear it immediately — it updates to
    /// the result's own summary and stays visible until `turnDone`, per the
    /// brief's Produces contract ("cleared on turnDone... after the turn
    /// continues", not cleared the instant a successful result arrives).
    func testToolCallSetsActivityThenOkResultUpdatesAndTurnDoneClears() {
        let state = AppState()
        state.reduce(.engine(.toolCall(name: "web_search", summary: "calling web_search")))
        XCTAssertEqual(state.toolActivity, "calling web_search")

        state.reduce(.engine(.toolResult(name: "web_search", ok: true, summary: "found 5 results")))
        XCTAssertEqual(state.toolActivity, "found 5 results", "ok:true keeps the LAST summary visible until turnDone")

        let latency = Latency(stt: 0.1, ttft: 0.1, firstClause: 0.1, ttsFirst: 0.1, total: 0.4)
        state.reduce(.engine(.turnDone(latency)))
        XCTAssertNil(state.toolActivity, "turnDone clears toolActivity")
    }

    /// A failed tool result shows its own summary (so the failure reads out
    /// to the user) rather than being cleared or left at the prior call's
    /// summary; entering the "idle" state clears it in all cases.
    func testToolResultFailureShowsSummaryAndIdleStateClears() {
        let state = AppState()
        state.reduce(.engine(.toolCall(name: "web_search", summary: "calling web_search")))
        state.reduce(.engine(.toolResult(name: "web_search", ok: false, summary: "search failed: network unreachable")))
        XCTAssertEqual(state.toolActivity, "search failed: network unreachable")

        state.reduce(.engine(.state("idle")))
        XCTAssertNil(state.toolActivity, "entering idle clears toolActivity in all cases")
    }

    /// `state == "idle"` clears `toolActivity` even without an intervening
    /// `turnDone` (e.g. after an aborted/failed turn) — pinned separately
    /// from the `turnDone` clear path since the brief specifies both as
    /// independent clearing triggers.
    func testIdleStateClearsToolActivityEvenWithoutTurnDone() {
        let state = AppState()
        state.reduce(.engine(.toolCall(name: "web_search", summary: "Calling web_search")))
        XCTAssertEqual(state.toolActivity, "Calling web_search")

        state.reduce(.engine(.state("idle")))
        XCTAssertNil(state.toolActivity)
    }

    /// Barge-in path: a turn goes PROCESSING/SPEAKING -> LISTENING without ever
    /// passing through "idle". Entering "listening" must clear a standing tool
    /// chip, otherwise a stale "Calling web_search" would freeze on screen
    /// through the entire next turn.
    func testListeningStateClearsToolActivityOnBargeIn() {
        let state = AppState()
        state.reduce(.engine(.state("speaking")))
        state.reduce(.engine(.toolCall(name: "web_search", summary: "Calling web_search")))
        XCTAssertEqual(state.toolActivity, "Calling web_search")

        state.reduce(.engine(.state("listening"))) // barge-in, no idle in between
        XCTAssertNil(state.toolActivity, "entering listening (barge-in) clears the stale tool chip")
    }

    // MARK: - Helpers

    private func makeSchemaField(key: String) -> SchemaField {
        SchemaField(
            key: key,
            type: "bool",
            section: "Language model",
            label: "Thinking mode",
            help: "Silent reasoning; shown, never spoken.",
            widget: "toggle",
            value: .bool(false),
            default: .bool(false),
            minimum: nil,
            maximum: nil,
            step: nil
        )
    }
}
