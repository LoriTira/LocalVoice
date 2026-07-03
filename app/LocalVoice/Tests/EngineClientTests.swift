import XCTest
@testable import LocalVoice

final class EngineClientTests: XCTestCase {
    /// The bash fixture, staged into a fresh temp directory for each test.
    ///
    /// The fixture source lives at a path relative to this source file
    /// (`#filePath`) so it resolves both under `xcodebuild test` (repo
    /// checkout present) and CI (`actions/checkout` puts the repo at the
    /// same relative layout). It's copied into `NSTemporaryDirectory()`
    /// before each spawn rather than executed in place: on a checkout that
    /// happens to sit under a TCC-protected user folder (e.g. `~/Desktop`),
    /// spawning a child whose executable path resolves inside that folder
    /// silently breaks pipe-readability delivery for the child's stdout on
    /// this OS/Xcode combination (empirically confirmed — `.readabilityHandler`
    /// and `FileHandle.bytes.lines` both hang identically; no TCC prompt or
    /// denial appears in the unified log, so the exact mechanism is opaque,
    /// but the behavior is 100% reproducible by path and disappears entirely
    /// once the executable is staged outside the protected folder). Staging
    /// to `NSTemporaryDirectory()` sidesteps the interaction regardless of
    /// where the repo checkout lives, which also matches how CI's checkout
    /// path (never under a protected folder) behaves.
    private static var fixtureSourceURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/fake_engine.sh")
    }

    private var stagedFixtureURL: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        let stagingDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("EngineClientTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: stagingDir, withIntermediateDirectories: true)
        let staged = stagingDir.appendingPathComponent("fake_engine.sh")
        try FileManager.default.copyItem(at: Self.fixtureSourceURL, to: staged)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: staged.path)
        stagedFixtureURL = staged
    }

    override func tearDownWithError() throws {
        if let stagedFixtureURL {
            try? FileManager.default.removeItem(at: stagedFixtureURL.deletingLastPathComponent())
        }
        stagedFixtureURL = nil
        try super.tearDownWithError()
    }

    private func makeClient(autoRespawn: Bool = true) -> EngineClient {
        EngineClient(mode: .custom(executable: stagedFixtureURL, arguments: []), autoRespawn: autoRespawn)
    }

    /// Collects events from `client.events` until `count` have arrived or
    /// `timeout` elapses, whichever comes first — bounded so a missing event
    /// fails the test instead of hanging the suite.
    private func collectEvents(from client: EngineClient, count: Int, timeout: TimeInterval = 5) async -> [ClientEvent] {
        await withTaskGroup(of: [ClientEvent].self) { group in
            group.addTask {
                var collected: [ClientEvent] = []
                for await event in await client.events {
                    collected.append(event)
                    if collected.count >= count { break }
                }
                return collected
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return []
            }
            let result = await group.next() ?? []
            group.cancelAll()
            return result
        }
    }

    /// Collects events from `client.events` until `stopCount` events matching
    /// `predicate` have arrived (inclusive of that last matching event) or
    /// `timeout` elapses, whichever comes first. Used for scenarios that need
    /// to run past an unbounded number of intervening events (e.g. `.spawned`
    /// / `.exited` pairs) to reach the Nth occurrence of a specific event.
    private func collectEvents(
        from client: EngineClient,
        untilMatching predicate: @escaping @Sendable (ClientEvent) -> Bool,
        stopCount: Int,
        timeout: TimeInterval = 5
    ) async -> [ClientEvent] {
        await withTaskGroup(of: [ClientEvent].self) { group in
            group.addTask {
                var collected: [ClientEvent] = []
                var matches = 0
                for await event in await client.events {
                    collected.append(event)
                    if predicate(event) {
                        matches += 1
                        if matches >= stopCount { break }
                    }
                }
                return collected
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return []
            }
            let result = await group.next() ?? []
            group.cancelAll()
            return result
        }
    }

    func testStartupYieldsSpawnedReadyEnginesReadyInOrder() async throws {
        let client = makeClient()
        addTeardownBlock { await client.stop() }
        await client.start()

        let events = await collectEvents(from: client, count: 3)
        try XCTAssertEqualSequence(events, [
            .spawned,
            .engine(.ready(version: 1, config: .object([:]), schema: [])),
            .engine(.enginesReady),
        ])
    }

    func testSendListModelsYieldsEchoedError() async throws {
        let client = makeClient()
        addTeardownBlock { await client.stop() }
        await client.start()
        _ = await collectEvents(from: client, count: 3) // drain startup

        await client.send(.listModels)
        let events = await collectEvents(from: client, count: 1)
        guard let first = events.first, events.count == 1 else {
            return XCTFail("expected exactly one echoed error event, got \(events)")
        }
        guard case let .engine(.error(message)) = first else {
            return XCTFail("expected .engine(.error), got \(first)")
        }
        XCTAssertTrue(message.contains("list_models"), "message '\(message)' should contain 'list_models'")
    }

    func testStopSendsShutdownAndExitsWithoutRespawn() async throws {
        let client = makeClient()
        await client.start()
        _ = await collectEvents(from: client, count: 3) // drain startup

        await client.stop()

        // stop() drives the shutdown command and waits for the fixture's
        // exit(0). The fixture's final `state: idle` protocol line may or
        // may not be read before pipe teardown (data-write vs. termination-
        // notification race), so in-flight engine events are tolerated; the
        // binding contract is the exit is treated as expected — no .exited,
        // no .respawning, no fresh .spawned — and send() after stop() is a
        // silent no-op (its echo must never appear).
        await client.send(.listModels)
        let events = await collectEvents(from: client, count: 5, timeout: 1)
        for event in events {
            switch event {
            case .respawning:
                XCTFail("stop() must not trigger a respawn, got \(events)")
            case .exited:
                XCTFail("exit after stop() must not be surfaced, got \(events)")
            case .spawned:
                XCTFail("stop() must not spawn a new process, got \(events)")
            case .engine(.error(let message)) where message.contains("list_models"):
                XCTFail("send() after stop() must be a no-op, got \(events)")
            case .engine:
                break // in-flight protocol lines from graceful shutdown are fine
            }
        }
    }

    func testAutoRespawnAfterImmediateExit() async throws {
        let client = EngineClient(
            mode: .custom(executable: URL(fileURLWithPath: "/usr/bin/true"), arguments: []),
            autoRespawn: true
        )
        addTeardownBlock { await client.stop() }
        await client.start()

        // /usr/bin/true still spawns successfully before exiting immediately,
        // so the full sequence is .spawned, then .exited, then .respawning.
        let events = await collectEvents(from: client, count: 3)
        try XCTAssertEqualSequence(events, [
            .spawned,
            .exited(code: 0),
            .respawning(attempt: 1, delaySeconds: 1),
        ])
    }

    /// `/usr/bin/true` exits immediately every time and never emits a
    /// `ready` engine event, so `respawnAttempt` has no reset point — each
    /// respawn must escalate the backoff attempt count with no resets.
    func testRespawnAttemptEscalatesWithoutReadyToResetIt() async throws {
        let client = EngineClient(
            mode: .custom(executable: URL(fileURLWithPath: "/usr/bin/true"), arguments: []),
            autoRespawn: true
        )
        addTeardownBlock { await client.stop() }
        await client.start()

        let events = await collectEvents(
            from: client,
            untilMatching: { if case .respawning = $0 { return true } else { return false } },
            stopCount: 2,
            timeout: 10
        )
        let respawningEvents = events.compactMap { event -> Int? in
            if case let .respawning(attempt, _) = event { return attempt }
            return nil
        }
        XCTAssertEqual(respawningEvents, [1, 2], "expected escalating attempts 1 then 2 with no ready to reset them, got \(events)")
    }

    /// A healthy boot (fixture reaches `ready`) must reset `respawnAttempt`
    /// to 0, so the very next crash-and-respawn reports attempt 1 again —
    /// not a continuation of whatever attempt count preceded the healthy
    /// boot. Triggered by sending a command whose encoded line contains
    /// "die", which the fixture matches before its echo catch-all and exits
    /// 7 without any respawn-suppressing shutdown handshake.
    func testRespawnAttemptResetsAfterReady() async throws {
        let client = makeClient()
        addTeardownBlock { await client.stop() }
        await client.start()

        // Drain startup (.spawned, .engine(.ready), .engine(.enginesReady))
        // and confirm ready was actually observed before proceeding.
        let startup = await collectEvents(from: client, count: 3)
        let sawReady = startup.contains { if case .engine(.ready) = $0 { return true } else { return false } }
        XCTAssertTrue(sawReady, "expected startup to include .engine(.ready), got \(startup)")

        await client.send(.downloadModel(repo: "die"))

        // Bounded at the first .respawning: the respawned fixture emits
        // ready again on its own, so collecting further would race that.
        let events = await collectEvents(
            from: client,
            untilMatching: { if case .respawning = $0 { return true } else { return false } },
            stopCount: 1,
            timeout: 10
        )
        try XCTAssertEqualSequence(events, [
            .exited(code: 7),
            .respawning(attempt: 1, delaySeconds: 1),
        ])
    }

    /// `stop()` must finish the `events` continuation so a `for await`
    /// consumer's loop actually exits instead of hanging forever waiting for
    /// a next element that will never come.
    func testStopFinishesEventStream() async throws {
        let client = makeClient()
        await client.start()
        _ = await collectEvents(from: client, count: 3) // drain startup, confirms ready

        await client.stop()

        let loopExited = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await _ in await client.events {
                    // Drain whatever trails stop() (e.g. a racy `state: idle`
                    // line) — the assertion is that this loop ends on its
                    // own, not what's inside it.
                }
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                return false
            }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }
        XCTAssertTrue(loopExited, "for await over client.events must exit after stop() finishes the continuation")
    }
}

/// Asserts `actual` and `expected` have the same count and equal elements at
/// every index, without ever subscripting past the shorter array's bounds —
/// `XCTAssertEqual(actual[i], expected[i])` after a failed count check would
/// crash the whole test process instead of failing just the one assertion.
private func XCTAssertEqualSequence<T: Equatable>(
    _ actual: [T],
    _ expected: [T],
    file: StaticString = #filePath,
    line: UInt = #line
) throws {
    guard actual.count == expected.count else {
        XCTFail("expected \(expected.count) events \(expected), got \(actual.count): \(actual)", file: file, line: line)
        return
    }
    for (index, (a, e)) in zip(actual, expected).enumerated() {
        XCTAssertEqual(a, e, "mismatch at index \(index)", file: file, line: line)
    }
}
