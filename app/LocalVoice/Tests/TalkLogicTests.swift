import XCTest
@testable import LocalVoice

/// Pins the Stop button's enabled mapping: active exactly while a response
/// is being generated or spoken. Idle has nothing to stop; listening ends
/// via push-to-talk release, not Stop.
final class TalkLogicTests: XCTestCase {
    func testStopEnabledWhileProcessing() {
        XCTAssertTrue(stopButtonEnabled(assistantState: "processing"))
    }

    func testStopEnabledWhileSpeaking() {
        XCTAssertTrue(stopButtonEnabled(assistantState: "speaking"))
    }

    func testStopDisabledWhileIdle() {
        XCTAssertFalse(stopButtonEnabled(assistantState: "idle"))
    }

    func testStopDisabledWhileListening() {
        XCTAssertFalse(stopButtonEnabled(assistantState: "listening"))
    }

    func testStopDisabledForUnknownState() {
        XCTAssertFalse(stopButtonEnabled(assistantState: "rebooting"))
    }
}
