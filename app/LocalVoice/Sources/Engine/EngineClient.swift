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
    func stop() async {
        isStopping = true
        guard let process, process.isRunning else {
            self.process = nil
            stdinHandle = nil
            finishStream()
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
        finishStream()
    }

    // MARK: - Process lifecycle

    private func spawn() {
        let spec = mode.launchSpec

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
            Task { await self?.handleStdout(data) }
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
            Task { await self?.handleExit(code: code) }
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

    private func handleStdout(_ data: Data) {
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

    private func handleExit(code: Int32) {
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

        Task {
            try? await Task.sleep(for: .seconds(delay))
            await self.respawnAfterBackoff()
        }
    }

    private func respawnAfterBackoff() {
        guard !isStopping else { return }
        spawn()
    }

    /// 1, 2, 4, 8, 8, 8, … seconds — doubles each attempt up to a cap of 8s
    /// so a persistently crashing engine doesn't back off indefinitely.
    private static func backoffDelay(forAttempt attempt: Int) -> Double {
        min(8.0, pow(2.0, Double(attempt - 1)))
    }
}
