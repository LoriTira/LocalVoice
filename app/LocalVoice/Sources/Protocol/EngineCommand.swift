import Foundation

/// An app-to-engine protocol command (`docs/gui.md` § Commands).
enum EngineCommand {
    case pttDown
    case pttUp(heldMs: Int)
    case esc
    case setConfig([String: JSONValue])
    case listModels
    case downloadModel(repo: String)
    case previewVoice(voice: String)
    case injectAudio(path: String)
    case shutdown
}

extension EngineCommand {
    /// Encodes this command as a single line of the engine's stdin
    /// protocol: one JSON object, key `"cmd"` plus this command's payload
    /// keys (`held_ms`, `changes`, `repo`, `voice`, `path`) exactly as
    /// named in `docs/gui.md`. The caller is responsible for appending the
    /// newline the transport requires — this returns the JSON text alone.
    func encodedLine() -> String {
        var payload: [String: JSONValue] = [:]

        switch self {
        case .pttDown:
            payload["cmd"] = .string("ptt_down")

        case .pttUp(let heldMs):
            payload["cmd"] = .string("ptt_up")
            payload["held_ms"] = .int(heldMs)

        case .esc:
            payload["cmd"] = .string("esc")

        case .setConfig(let changes):
            payload["cmd"] = .string("set_config")
            payload["changes"] = .object(changes)

        case .listModels:
            payload["cmd"] = .string("list_models")

        case .downloadModel(let repo):
            payload["cmd"] = .string("download_model")
            payload["repo"] = .string(repo)

        case .previewVoice(let voice):
            payload["cmd"] = .string("preview_voice")
            payload["voice"] = .string(voice)

        case .injectAudio(let path):
            payload["cmd"] = .string("inject_audio")
            payload["path"] = .string(path)

        case .shutdown:
            payload["cmd"] = .string("shutdown")
        }

        let data = (try? JSONEncoder().encode(JSONValue.object(payload))) ?? Data()
        return String(data: data, encoding: .utf8) ?? ""
    }
}
