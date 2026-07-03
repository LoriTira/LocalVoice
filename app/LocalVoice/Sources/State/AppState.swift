import Foundation
import Observation

/// One turn of the conversation transcript — either what the user said (STT
/// output) or the assistant's reply, assembled clause-by-clause as it
/// streams in. `id` is assigned by `AppState`'s monotonic counter, not
/// derived from array position, so a turn keeps a stable identity across
/// mutation (SwiftUI `List`/`ForEach` diffing) and even after array
/// reordering, were that ever to happen.
struct Turn: Equatable, Identifiable {
    let id: Int
    var role: String // "user" | "assistant"
    var text: String
    var reasoning: String?
}

/// The single source of truth for everything the SwiftUI views render,
/// rebuilt turn-by-turn from the engine's protocol stream via `reduce(_:)`.
/// One `AppState` is meant to be fed by exactly one `EngineClient.events`
/// consumer (typically a long-lived `for await` loop owned by the app
/// shell) — `reduce` itself does not read the stream; it is a pure
/// event-in, state-out step invoked once per `ClientEvent`, so it stays
/// trivially testable without any async machinery.
///
/// `@Observable` (not `ObservableObject`) so SwiftUI views that read a
/// specific field only re-render on changes to that field, per the
/// property-level dependency tracking `@Observable` provides.
@MainActor
@Observable
final class AppState {
    var connection: String = "connecting" // connecting | ready | engines_ready | dead
    var assistantState: String = "idle" // mirror of protocol state values
    var turns: [Turn] = []
    var micLevel: Double = 0
    var lastLatency: Latency?
    var config: JSONValue = .null
    var schema: [SchemaField] = []
    var models: [InstalledModel] = []
    var downloads: [String: Double?] = [:] // repo -> pct (nil = indeterminate); removed when done
    var loading: [String: String] = [:] // engine -> phase, while a load/reload runs
    var banner: String? // latest error message; nil when dismissed
    var protocolMismatch: Bool = false

    /// Backs `Turn.id` — monotonically increasing across the whole session
    /// (never reset per pair, never derived from `turns.count`), so ids stay
    /// unique even as turns are appended and (in principle) later removed.
    private var nextTurnId = 0

    func reduce(_ event: ClientEvent) {
        switch event {
        case .spawned:
            connection = "connecting"
        case .exited:
            connection = "dead"
        case .respawning:
            connection = "connecting"
        case .engine(let engineEvent):
            reduce(engineEvent)
        }
    }

    private func reduce(_ event: EngineEvent) {
        switch event {
        case let .ready(version, config, schema):
            self.config = config
            self.schema = schema
            connection = "ready"
            protocolMismatch = (version != 1)

        case .enginesReady:
            connection = "engines_ready"

        case let .state(s):
            // Level-triggered per docs/gui.md: repeats of the same value are
            // idempotent, but micLevel only resets when the state actually
            // changes away from "listening" (the mic isn't armed outside
            // that state) — resetting on every repeat would zero a level
            // reading that arrived earlier in the same "listening" run.
            let changedAwayFromListening = assistantState == "listening" && s != "listening"
            assistantState = s
            if changedAwayFromListening {
                micLevel = 0
            }

        case let .userText(text):
            appendTurn(role: "user", text: text)
            appendTurn(role: "assistant", text: "")

        case let .assistantClause(text):
            if let last = turns.indices.last, turns[last].role == "assistant" {
                let existing = turns[last].text
                turns[last].text = existing.isEmpty ? text : existing + " " + text
            } else {
                appendTurn(role: "assistant", text: text)
            }

        case let .reasoning(text):
            if let last = turns.indices.last, turns[last].role == "assistant" {
                if let existing = turns[last].reasoning, !existing.isEmpty {
                    turns[last].reasoning = existing + text
                } else {
                    turns[last].reasoning = text
                }
            } else {
                var turn = makeTurn(role: "assistant", text: "")
                turn.reasoning = text
                turns.append(turn)
            }

        case let .turnDone(latency):
            lastLatency = latency

        case let .level(v):
            micLevel = v

        case let .loadProgress(engine, phase, _):
            if phase == "done" {
                loading.removeValue(forKey: engine)
            } else {
                loading[engine] = phase
            }

        case let .downloadProgress(repo, pct, done):
            if done {
                downloads.removeValue(forKey: repo)
            } else {
                downloads[repo] = pct
            }

        case let .models(m):
            models = m

        case let .configApplied(c, _):
            config = c
            banner = nil

        case let .error(message):
            banner = message
        }
    }

    // MARK: - Turn helpers

    private func makeTurn(role: String, text: String) -> Turn {
        defer { nextTurnId += 1 }
        return Turn(id: nextTurnId, role: role, text: text, reasoning: nil)
    }

    private func appendTurn(role: String, text: String) {
        turns.append(makeTurn(role: role, text: text))
    }
}
