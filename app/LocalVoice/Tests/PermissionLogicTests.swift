import XCTest
@testable import LocalVoice

/// Tests for `permissionSummary(mic:inputMonitoring:)`, the pure helper
/// behind `SetupView`'s two permission cards. `mic` mirrors
/// `AVCaptureDevice.authorizationStatus(for: .audio)` collapsed to the three
/// states the card actually distinguishes: `nil` for `.notDetermined`
/// (no prompt shown yet), `true` for `.authorized`, `false` for anything
/// else (`.denied`/`.restricted` — both mean "the user cannot grant this
/// without System Settings", so the card treats them identically).
/// `inputMonitoring` mirrors `CGPreflightListenEventAccess()` directly — that
/// API is already boolean (macOS has no "not yet asked" state for Input
/// Monitoring the way Photos/Camera do: preflight just reports the current
/// TCC grant).
///
/// The six cases below are the exact matrix: 3 mic states x 2 input-monitoring
/// states. Copy is sentence case, no emoji, no exclamation marks, per the
/// task-7 contract.
final class PermissionLogicTests: XCTestCase {
    func testMicNotDeterminedAndInputMonitoringGranted() {
        XCTAssertEqual(
            permissionSummary(mic: nil, inputMonitoring: true),
            "Microphone access has not been requested yet. Input Monitoring is granted."
        )
    }

    func testMicNotDeterminedAndInputMonitoringDenied() {
        XCTAssertEqual(
            permissionSummary(mic: nil, inputMonitoring: false),
            "Microphone access has not been requested yet. Input Monitoring is not granted."
        )
    }

    func testMicGrantedAndInputMonitoringGranted() {
        XCTAssertEqual(
            permissionSummary(mic: true, inputMonitoring: true),
            "Microphone access is granted. Input Monitoring is granted."
        )
    }

    func testMicGrantedAndInputMonitoringDenied() {
        XCTAssertEqual(
            permissionSummary(mic: true, inputMonitoring: false),
            "Microphone access is granted. Input Monitoring is not granted."
        )
    }

    func testMicDeniedAndInputMonitoringGranted() {
        XCTAssertEqual(
            permissionSummary(mic: false, inputMonitoring: true),
            "Microphone access is denied. Open System Settings to allow it. Input Monitoring is granted."
        )
    }

    func testMicDeniedAndInputMonitoringDenied() {
        XCTAssertEqual(
            permissionSummary(mic: false, inputMonitoring: false),
            "Microphone access is denied. Open System Settings to allow it. Input Monitoring is not granted."
        )
    }

    /// Tests for `screenRecordingStatusText(granted:)`, the pure helper
    /// behind the Screen Recording card added in tools phase T3 Task 5.
    /// `CGPreflightScreenCaptureAccess()` is already a plain grant/no-grant
    /// `Bool` — no `.notDetermined`-style third state the way Microphone's
    /// `AVAuthorizationStatus` has — so this helper is a direct two-case
    /// mapping, not a three-way collapse like `permissionSummary`'s `mic`
    /// parameter above. It is deliberately a separate function rather than
    /// a third parameter on `permissionSummary`: Screen Recording is
    /// excluded from that combined summary sentence (see
    /// `SetupView.screenRecordingCard`'s doc comment for why), so it gets
    /// its own tiny pure helper — and its own tests — instead of growing
    /// the six-case matrix above into a twelve-case one.
    func testScreenRecordingStatusTextGranted() {
        XCTAssertEqual(screenRecordingStatusText(granted: true), "Granted.")
    }

    func testScreenRecordingStatusTextNotGranted() {
        XCTAssertEqual(screenRecordingStatusText(granted: false), "Not granted.")
    }
}
