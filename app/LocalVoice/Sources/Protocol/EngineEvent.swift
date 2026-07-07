import Foundation

/// One descriptor from `ready.schema` / `config_applied`'s implied schema —
/// see `docs/gui.md` § "Derived settings schema". Field types stay as
/// `JSONValue` for `value`/`default` since the underlying config field can
/// be bool/int/float/str.
struct SchemaField: Codable, Equatable {
    let key: String
    let type: String
    let section: String
    let label: String
    let help: String
    let widget: String
    let value: JSONValue
    let `default`: JSONValue
    let minimum: Double?
    let maximum: Double?
    let step: Double?
}

/// One entry from a `models` event's `installed` list.
struct InstalledModel: Codable, Equatable, Identifiable {
    let id: String
    let path: String
    let kind: String
    let sizeGb: Double

    enum CodingKeys: String, CodingKey {
        case id, path, kind
        case sizeGb = "size_gb"
    }
}

/// `turn_done.latency` — all durations in seconds, all measured
/// pipeline-internally; see `docs/gui.md` for the precise definition of
/// each key.
struct Latency: Codable, Equatable {
    let stt: Double
    let ttft: Double
    let firstClause: Double
    let ttsFirst: Double
    let total: Double

    enum CodingKeys: String, CodingKey {
        case stt, ttft, total
        case firstClause = "first_clause"
        case ttsFirst = "tts_first"
    }
}

/// A decoded engine-to-app protocol event (`docs/gui.md` § Events).
///
/// Not directly `Decodable` — the wire format is a JSON object
/// discriminated by an `"event"` string key, with each event's payload
/// shape unrelated to the others, which doesn't map onto Swift's
/// enum-with-associated-values `Codable` synthesis. `decode(line:)` is the
/// single entry point: it parses the line as JSON, reads `"event"`, and
/// dispatches to the matching case, returning `nil` for anything it
/// doesn't recognize (unknown event name, non-object top level, or
/// unparsable JSON) so the protocol can grow without breaking older
/// clients.
enum EngineEvent: Equatable {
    case ready(version: Int, config: JSONValue, schema: [SchemaField])
    case state(String)
    case userText(String)
    case assistantClause(String)
    case reasoning(String)
    case toolCall(name: String, summary: String)
    case toolResult(name: String, ok: Bool, summary: String)
    case turnDone(Latency)
    case level(Double)
    case loadProgress(engine: String, phase: String, seconds: Double?)
    case enginesReady
    case downloadProgress(repo: String, pct: Double?, done: Bool)
    case models([InstalledModel])
    case configApplied(config: JSONValue, reloaded: [String])
    case error(String)
}

extension EngineEvent {
    /// Parses one line of the engine's stdout protocol stream. Returns
    /// `nil` on malformed JSON, a non-object top level (e.g. `[1, 2]`), or
    /// an `"event"` name this client build doesn't know about yet —
    /// forward compatibility with a protocol that may grow new event
    /// types, per `docs/gui.md`.
    static func decode(line: String) -> EngineEvent? {
        guard let data = line.data(using: .utf8) else { return nil }
        guard let value = try? JSONDecoder().decode(JSONValue.self, from: data) else { return nil }
        guard case let .object(root) = value else { return nil }
        guard case let .string(eventName)? = root["event"] else { return nil }

        switch eventName {
        case "ready":
            guard case let .int(version)? = root["version"],
                  let config = root["config"],
                  let schemaValue = root["schema"],
                  let schema = decodeSchema(schemaValue)
            else { return nil }
            return .ready(version: version, config: config, schema: schema)

        case "state":
            guard case let .string(state)? = root["state"] else { return nil }
            return .state(state)

        case "user_text":
            guard case let .string(text)? = root["text"] else { return nil }
            return .userText(text)

        case "assistant_clause":
            guard case let .string(text)? = root["text"] else { return nil }
            return .assistantClause(text)

        case "reasoning":
            guard case let .string(text)? = root["text"] else { return nil }
            return .reasoning(text)

        case "tool_call":
            guard case let .string(name)? = root["name"],
                  case let .string(summary)? = root["summary"]
            else { return nil }
            return .toolCall(name: name, summary: summary)

        case "tool_result":
            guard case let .string(name)? = root["name"],
                  case let .bool(ok)? = root["ok"],
                  case let .string(summary)? = root["summary"]
            else { return nil }
            return .toolResult(name: name, ok: ok, summary: summary)

        case "turn_done":
            guard let latencyValue = root["latency"],
                  let latency = decodeValue(latencyValue, as: Latency.self)
            else { return nil }
            return .turnDone(latency)

        case "level":
            guard let rms = numberAsDouble(root["rms"]) else { return nil }
            return .level(rms)

        case "load_progress":
            guard case let .string(engine)? = root["engine"],
                  case let .string(phase)? = root["phase"]
            else { return nil }
            let seconds = numberAsDouble(root["seconds"])
            return .loadProgress(engine: engine, phase: phase, seconds: seconds)

        case "engines_ready":
            return .enginesReady

        case "download_progress":
            guard case let .string(repo)? = root["repo"],
                  case let .bool(done)? = root["done"]
            else { return nil }
            let pct = numberAsDouble(root["pct"])
            return .downloadProgress(repo: repo, pct: pct, done: done)

        case "models":
            guard let installedValue = root["installed"],
                  let installed = decodeValue(installedValue, as: [InstalledModel].self)
            else { return nil }
            return .models(installed)

        case "config_applied":
            guard let config = root["config"],
                  case let .array(reloadedValues)? = root["reloaded"]
            else { return nil }
            var reloaded: [String] = []
            for item in reloadedValues {
                guard case let .string(name) = item else { return nil }
                reloaded.append(name)
            }
            return .configApplied(config: config, reloaded: reloaded)

        case "error":
            guard case let .string(message)? = root["message"] else { return nil }
            return .error(message)

        default:
            return nil
        }
    }

    /// Re-encodes a `JSONValue` and decodes it as `T` — used to reuse
    /// `JSONValue`'s int-before-double number handling for nested
    /// `Codable` payloads (`Latency`, `[InstalledModel]`) instead of
    /// hand-unpacking every field.
    private static func decodeValue<T: Decodable>(_ value: JSONValue, as type: T.Type) -> T? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    private static func decodeSchema(_ value: JSONValue) -> [SchemaField]? {
        decodeValue(value, as: [SchemaField].self)
    }

    /// `JSONValue.int`/`.double` both represent JSON numbers; several
    /// float-typed payload fields (`rms`, `seconds`, `pct`) may arrive as
    /// either depending on whether the engine emitted a whole number, so
    /// this widens both to `Double`. Returns `nil` for `.null`/missing/
    /// non-numeric — callers use that to mean "absent" where the field is
    /// optional.
    private static func numberAsDouble(_ value: JSONValue?) -> Double? {
        switch value {
        case .int(let value): return Double(value)
        case .double(let value): return value
        default: return nil
        }
    }
}
