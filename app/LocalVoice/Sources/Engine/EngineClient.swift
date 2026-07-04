import Foundation
import os

/// An event `EngineClient` yields on its `events` stream: either a decoded
/// protocol event from the engine, or a client-level lifecycle event about
/// the subprocess itself.
enum ClientEvent: Equatable {
    case engine(EngineEvent)
    case spawned
    case exited(code: Int32)
    case respawning(attempt: Int, delaySeconds: Double)
}

/// Owns the engine subprocess: spawns it per `LaunchMode`, decodes its
/// stdout protocol stream into `ClientEvent`s, forwards stderr to the
/// system logger, and respawns on unexpected exit with capped exponential
/// backoff. One `EngineClient` manages at most one live subprocess at a
/// time; `events` is meant for a single consumer (typically `AppState`'s
/// reduce loop).
actor EngineClient {
    private let mode: LaunchMode
    private let autoRespawn: Bool

    private let continuation: AsyncStream<ClientEvent>.Continuation
    nonisolated let events: AsyncStream<ClientEvent>

    private var process: Process?
    private var stdinHandle: FileHandle?

    /// True once `stop()` has been called — suppresses respawn and further
    /// `send()`s even if a stray exit notification arrives afterward.
    private var isStopping = false

    /// True once `continuation.finish()` has been called — guards against a
    /// double-finish (e.g. `stop()` racing a final no-respawn death).
    private var didFinishStream = false

    /// Count of stdout lines that failed `EngineEvent.decode` — exposed for
    /// tests; not surfaced on the event stream since a garbled line isn't
    /// actionable by the UI.
    private(set) var droppedLines = 0

    /// Reset to 0 when the engine reports ready (a healthy boot); fast-flapping
    /// processes that die before ready keep escalating. Drives the respawn
    /// backoff schedule (1, 2, 4, 8, 8, … seconds).
    private var respawnAttempt = 0

    /// Monotonic id bumped on every teardown (`stop()` / `restart()`) so a
    /// late exit callback from an *old* process is ignored. Each spawn's
    /// `terminationHandler`/readability closures capture the generation live
    /// at exit time; `handleExit` and `handleStdout` no-op unless it still
    /// matches. This is what keeps an OLD process's EOF/termination — which
    /// can land *after* `restart()` has already reset `isStopping = false`
    /// and spawned the NEW process — from being misread as the new one's
    /// death and triggering a spurious respawn.
    private var generation = 0

    /// The in-flight backoff `Task` from `scheduleRespawnIfNeeded`, if any.
    /// Held so a teardown can cancel a pending respawn before it fires,
    /// preventing a double-spawn when `restart()` (or `stop()`) races the
    /// backoff window.
    private var respawnTask: Task<Void, Never>?

    /// Monotonic id for each scheduled backoff. The backoff `Task` captures
    /// its own token (a plain `Sendable Int`, unlike a self-referential
    /// `Task` capture which trips Swift 6 strict-concurrency) and only spawns
    /// if it's still the current one when it wakes — so a cancelled backoff
    /// that fires after a newer one was scheduled bows out instead of
    /// clobbering it.
    private var respawnToken = 0

    /// Buffers a stdout chunk that hasn't yet seen a trailing newline —
    /// `Pipe` delivers arbitrary byte chunks, not lines.
    private var stdoutBuffer = Data()

    private static let logger = Logger(subsystem: "dev.localvoice", category: "engine")

    init(mode: LaunchMode, autoRespawn: Bool = true) {
        self.mode = mode
        self.autoRespawn = autoRespawn
        var continuation: AsyncStream<ClientEvent>.Continuation!
        self.events = AsyncStream(bufferingPolicy: .unbounded) { continuation = $0 }
        self.continuation = continuation
    }

    func start() {
        isStopping = false
        spawn()
    }

    /// Finishes `events`, guarded so a second call (e.g. `stop()` racing a
    /// final no-respawn death) is a no-op.
    private func finishStream() {
        guard !didFinishStream else { return }
        didFinishStream = true
        continuation.finish()
    }

    /// Writes `command`'s encoded line + newline to the child's stdin.
    /// Silently does nothing if no process is currently running (not yet
    /// spawned, or dead awaiting respawn) — the UI is expected to gate
    /// interactive controls on connection state, so this is a defensive
    /// no-op rather than a reportable error.
    func send(_ command: EngineCommand) {
        guard let stdinHandle, process?.isRunning == true else { return }
        var line = command.encodedLine()
        line.append("\n")
        guard let data = line.data(using: .utf8) else { return }
        try? stdinHandle.write(contentsOf: data)
    }

    /// Asks the engine to shut down gracefully, gives it 3 seconds, then
    /// SIGKILLs if it hasn't exited. No respawn happens after `stop()`,
    /// including for a process that dies mid-grace-period on its own.
    /// Finishes the `events` continuation — a consumer's `for await` loop
    /// exits. Used for app quit; use `restart()` to cycle the engine while
    /// keeping the stream (and its single consumer) alive.
    func stop() async {
        isStopping = true
        await teardownChild()
        finishStream()
    }

    /// Cycles the engine subprocess in place: tears the child down through
    /// the same graceful path as `stop()` (shutdown command → 3s grace →
    /// SIGKILL fallback → clear handlers/pipes → cancel any pending respawn),
    /// but WITHOUT finishing the `events` continuation, then runs the normal
    /// `start()` spawn path so the new process's events land on the *same*
    /// stream the app's one consumer is already awaiting. This is what
    /// Setup's "Restart Engine" needs: `stop()` + `start()` would finish the
    /// stream on the way down, permanently severing the app's lone
    /// `for await event in client.events` loop so nothing after the restart
    /// (transcript, orb, footer) would ever update again.
    func restart() async {
        // Bump generation and set `isStopping` up front: any exit callback
        // from the outgoing process now belongs to a stale generation and
        // is suppressed by both `isStopping` (during teardown) and the
        // generation guard (after `start()` clears `isStopping` below).
        isStopping = true
        await teardownChild()
        // `start()` resets `isStopping = false` and spawns, incrementing the
        // generation again so the fresh process gets a live callback id.
        start()
    }

    /// Shared teardown for `stop()`/`restart()`: cancels any pending respawn,
    /// then gracefully stops the live child (shutdown → grace → SIGKILL) and
    /// clears process/pipe state. Assumes `isStopping` is already `true` so a
    /// racing exit callback is suppressed. Advances `generation` so any
    /// in-flight exit callback for the outgoing process is stale even after a
    /// subsequent `start()` flips `isStopping` back to `false`. Does not touch
    /// the continuation — the caller decides whether to finish it.
    private func teardownChild() async {
        generation += 1
        // Invalidate any pending backoff both ways: cancel wakes it early, and
        // bumping the token makes it a stale no-op even if it wakes after a
        // subsequent `start()` has cleared `isStopping` (the `restart()` case).
        respawnToken += 1
        respawnTask?.cancel()
        respawnTask = nil

        guard let process, process.isRunning else {
            self.process = nil
            stdinHandle = nil
            return
        }

        send(.shutdown)

        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while process.isRunning, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }

        self.process = nil
        stdinHandle = nil
    }

    // MARK: - Process lifecycle

    private func spawn() {
        let spec = mode.launchSpec

        // This spawn's callback generation: teardown bumped `generation` on
        // the way down, so the outgoing process's still-in-flight callbacks
        // no longer match. Capturing the *new* value here (rather than
        // reading `self.generation` live from the closures) is what lets
        // `handleExit`/`handleStdout` distinguish this process's callbacks
        // from an old one's.
        generation += 1
        let spawnGeneration = generation

        let process = Process()
        process.executableURL = spec.executable
        process.arguments = spec.arguments
        process.currentDirectoryURL = spec.workingDirectory

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        stdoutBuffer.removeAll()

        // An empty `availableData` read signals EOF on the pipe (the child
        // closed its end, typically by exiting). `readabilityHandler` keeps
        // re-invoking immediately for a readable-but-EOF'd FD, so the
        // handler must be cleared right here rather than waiting on the
        // separate `terminationHandler` hop — otherwise this busy-loops
        // between EOF and that hop landing.
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            Task { await self?.handleStdout(data, generation: spawnGeneration) }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            if let text = String(data: data, encoding: .utf8), !text.isEmpty {
                Self.logger.error("\(text, privacy: .public)")
            }
        }
        process.terminationHandler = { [weak self] proc in
            let code = proc.terminationStatus
            Task { await self?.handleExit(code: code, generation: spawnGeneration) }
        }

        do {
            try process.run()
        } catch {
            Self.logger.error("failed to launch engine: \(error, privacy: .public)")
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
            scheduleRespawnIfNeeded()
            return
        }

        self.process = process
        self.stdinHandle = stdinPipe.fileHandleForWriting
        continuation.yield(.spawned)
    }

    private func handleStdout(_ data: Data, generation: Int) {
        // Drop stdout from a superseded process: a chunk read from the old
        // pipe can still be in flight to this actor when a teardown +
        // re-spawn has already happened, and appending it to the fresh
        // `stdoutBuffer` would splice one process's bytes into another's.
        guard generation == self.generation else { return }

        stdoutBuffer.append(data)

        // Split on newline, keeping any trailing partial line buffered for
        // the next chunk.
        while let newlineIndex = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer[stdoutBuffer.startIndex..<newlineIndex]
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineIndex)

            guard let line = String(data: lineData, encoding: .utf8), !line.isEmpty else { continue }
            if let event = EngineEvent.decode(line: line) {
                if case .ready = event {
                    respawnAttempt = 0
                }
                continuation.yield(.engine(event))
            } else {
                droppedLines += 1
            }
        }
    }

    private func handleExit(code: Int32, generation: Int) {
        // Ignore an exit callback from a superseded process. `restart()` bumps
        // `generation` and then clears `isStopping` via `start()`, so the
        // `isStopping` check alone can't catch an old termination that lands
        // in that window — but a stale generation always does. `teardownChild`
        // already SIGKILLed and cleared the old process synchronously, so
        // there's nothing here to clean up for it either.
        guard generation == self.generation else { return }

        // Tear down pipe handlers so they don't fire again for this process.
        if let stdout = process?.standardOutput as? Pipe {
            stdout.fileHandleForReading.readabilityHandler = nil
        }
        if let stderr = process?.standardError as? Pipe {
            stderr.fileHandleForReading.readabilityHandler = nil
        }

        let wasStopping = isStopping
        process = nil
        stdinHandle = nil

        guard !wasStopping else { return }

        continuation.yield(.exited(code: code))
        scheduleRespawnIfNeeded()
    }

    private func scheduleRespawnIfNeeded() {
        guard autoRespawn, !isStopping else {
            finishStream()
            return
        }

        respawnAttempt += 1
        let delay = Self.backoffDelay(forAttempt: respawnAttempt)
        continuation.yield(.respawning(attempt: respawnAttempt, delaySeconds: delay))

        // Held so a teardown mid-backoff (`stop()`/`restart()`) can cancel
        // this before it fires — otherwise a `restart()` landing during the
        // backoff window would spawn once here and once from `start()`. The
        // token distinguishes this schedule from any later one, so a
        // cancelled backoff that wakes *after* a fresh schedule was installed
        // doesn't spawn a stale second process.
        respawnToken += 1
        let token = respawnToken
        respawnTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            await self?.respawnAfterBackoff(token: token)
        }
    }

    private func respawnAfterBackoff(token: Int) {
        // Only the current backoff spawns; a superseded/cancelled one bows out.
        guard token == respawnToken, !isStopping else { return }
        respawnTask = nil
        spawn()
    }

    /// 1, 2, 4, 8, 8, 8, … seconds — doubles each attempt up to a cap of 8s
    /// so a persistently crashing engine doesn't back off indefinitely.
    private static func backoffDelay(forAttempt attempt: Int) -> Double {
        min(8.0, pow(2.0, Double(attempt - 1)))
    }
}
