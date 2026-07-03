import XCTest
@testable import LocalVoice

final class ProtocolTests: XCTestCase {
    func testDecodeState() throws {
        let e = EngineEvent.decode(line: #"{"event": "state", "state": "listening"}"#)
        XCTAssertEqual(e, .state("listening"))
    }

    func testDecodeTurnDone() throws {
        let line = #"{"event": "turn_done", "latency": {"stt": 0.34, "ttft": 0.31, "first_clause": 0.07, "tts_first": 0.10, "total": 0.82}}"#
        guard case let .turnDone(l)? = EngineEvent.decode(line: line) else { return XCTFail("wrong case") }
        XCTAssertEqual(l.total, 0.82, accuracy: 1e-9)
        XCTAssertEqual(l.firstClause, 0.07, accuracy: 1e-9)
    }

    func testDecodeReadySchemaAndConfig() throws {
        let line = #"{"event": "ready", "version": 1, "config": {"llm": {"think": false, "max_tokens": 1024}}, "schema": [{"key": "llm.think", "type": "bool", "value": false, "default": false, "section": "Language model", "label": "Thinking mode", "help": "", "widget": "toggle"}]}"#
        guard case let .ready(version, config, schema)? = EngineEvent.decode(line: line) else { return XCTFail("wrong case") }
        XCTAssertEqual(version, 1)
        XCTAssertEqual(schema.count, 1)
        XCTAssertEqual(schema[0].key, "llm.think")
        XCTAssertEqual(schema[0].value, .bool(false))
        guard case let .object(root)? = Optional(config), case let .object(llm)? = root["llm"] else { return XCTFail("config shape") }
        XCTAssertEqual(llm["max_tokens"], .int(1024))
    }

    func testDecodeModelsAndProgressAndError() throws {
        XCTAssertEqual(
            EngineEvent.decode(line: #"{"event": "models", "installed": [{"id": "a/b", "path": "/x", "size_gb": 18.6, "kind": "mlx"}]}"#),
            .models([InstalledModel(id: "a/b", path: "/x", kind: "mlx", sizeGb: 18.6)])
        )
        XCTAssertEqual(
            EngineEvent.decode(line: #"{"event": "download_progress", "repo": "a/b", "pct": null, "done": false}"#),
            .downloadProgress(repo: "a/b", pct: nil, done: false)
        )
        XCTAssertEqual(EngineEvent.decode(line: #"{"event": "error", "message": "boom"}"#), .error("boom"))
        XCTAssertEqual(EngineEvent.decode(line: #"{"event": "engines_ready"}"#), .enginesReady)
        XCTAssertEqual(
            EngineEvent.decode(line: #"{"event": "load_progress", "engine": "llm", "phase": "done", "seconds": 4.6}"#),
            .loadProgress(engine: "llm", phase: "done", seconds: 4.6)
        )
    }

    func testUnknownEventAndGarbageReturnNil() {
        XCTAssertNil(EngineEvent.decode(line: #"{"event": "brand_new_thing", "x": 1}"#))
        XCTAssertNil(EngineEvent.decode(line: "not json"))
        XCTAssertNil(EngineEvent.decode(line: "[1, 2]"))
    }

    func testEncodeCommands() throws {
        func json(_ s: String) throws -> NSDictionary {
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data(s.utf8)) as? NSDictionary)
        }
        try XCTAssertEqual(json(EngineCommand.pttUp(heldMs: 512).encodedLine()), ["cmd": "ptt_up", "held_ms": 512])
        try XCTAssertEqual(json(EngineCommand.pttDown.encodedLine()), ["cmd": "ptt_down"])
        try XCTAssertEqual(
            json(EngineCommand.setConfig(["llm.think": .bool(true)]).encodedLine()),
            ["cmd": "set_config", "changes": ["llm.think": true]]
        )
        try XCTAssertEqual(
            json(EngineCommand.previewVoice(voice: "af_heart").encodedLine()),
            ["cmd": "preview_voice", "voice": "af_heart"]
        )
        XCTAssertFalse(EngineCommand.shutdown.encodedLine().contains("\n"))
    }
}
