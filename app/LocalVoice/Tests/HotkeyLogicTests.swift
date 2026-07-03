import XCTest
import CoreGraphics
@testable import LocalVoice

/// Tests for `decide(type:keycode:commandBit:pttCurrentlyDown:)`, the pure
/// helper behind `HotkeyMonitor`'s `CGEventTap` callback — see that type's
/// doc comment for why the tap callback itself stays untested here (it
/// requires a live, granted Input Monitoring tap; this file exercises only
/// the decision logic the callback delegates to, with plain
/// `CGEventType`/`Int64`/`Bool` inputs it can synthesize without any real
/// event or permission).
///
/// Keycodes: 54 is right-command (`kVK_RightCommand`), 53 is Escape
/// (`kVK_Escape`) — the two physical keys `docs/gui.md`'s default
/// `keys.ptt`/`keys.stop` map to, mirrored here as the GUI's hard-coded
/// global-hotkey keys per the task-8 contract (the GUI owns the key tap in
/// Swift; `serve` does not install one — see `docs/gui.md` line 285).
///
/// The complete truth table (task-8 brief, verbatim):
/// - `(flagsChanged, 54, commandBit: true,  pttCurrentlyDown: false)` -> `.pttDown`
/// - `(flagsChanged, 54, commandBit: false, pttCurrentlyDown: true)`  -> `.pttUp`
/// - `(flagsChanged, 54, commandBit: true,  pttCurrentlyDown: true)`  -> `.none`
/// - `(keyDown, 53, _, _)` -> `.esc`
/// - everything else -> `.none`
///
/// Below, each row of that table gets its own test, plus every "else" row
/// the brief calls out by name (wrong keycode, `keyUp` type, Esc while PTT
/// is held) and the one remaining cell of the 2x2 `commandBit` x
/// `pttCurrentlyDown` grid on keycode 54/`flagsChanged` the brief's table
/// doesn't spell out by coordinate (`commandBit: false, pttCurrentlyDown:
/// false` — a stray flags-changed with nothing down and no command bit set,
/// e.g. release of an unrelated modifier) — real input space that must
/// still resolve to `.none`.
final class HotkeyLogicTests: XCTestCase {
    // MARK: - The three affirmative rows

    func testFlagsChangedRightCommandDownWithCommandBitAndNotCurrentlyDownYieldsPttDown() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: true, pttCurrentlyDown: false),
            .pttDown
        )
    }

    func testFlagsChangedRightCommandUpWithoutCommandBitAndCurrentlyDownYieldsPttUp() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: false, pttCurrentlyDown: true),
            .pttUp
        )
    }

    func testKeyDownEscYieldsEsc() {
        XCTAssertEqual(
            decide(type: .keyDown, keycode: 53, commandBit: false, pttCurrentlyDown: false),
            .esc
        )
    }

    /// Esc fires regardless of `commandBit`/`pttCurrentlyDown` (the brief's
    /// `_`/`_` wildcards) — verified with both flags flipped from the
    /// baseline row above to prove the wildcard, not just one fixed value.
    func testKeyDownEscIgnoresCommandBitAndPttCurrentlyDownState() {
        XCTAssertEqual(
            decide(type: .keyDown, keycode: 53, commandBit: true, pttCurrentlyDown: true),
            .esc
        )
    }

    // MARK: - The two named "none" rows on keycode 54 (repeat-suppression)

    /// Command bit still set while PTT is already down: a repeat
    /// `flagsChanged` (or a second modifier changing alongside an already-held
    /// right command) must not fire a second `pttDown`.
    func testFlagsChangedRightCommandWithCommandBitWhileAlreadyDownYieldsNone() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: true, pttCurrentlyDown: true),
            .none
        )
    }

    /// The remaining cell of the 2x2 grid the brief's table doesn't name by
    /// coordinate: command bit clear and PTT not currently down. There is no
    /// held press to release, so this must not fire a spurious `pttUp`.
    func testFlagsChangedRightCommandWithoutCommandBitAndNotCurrentlyDownYieldsNone() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: false, pttCurrentlyDown: false),
            .none
        )
    }

    // MARK: - Explicit "else" rows named in the task-8 contract

    /// Wrong keycode on `flagsChanged` with the command bit set — some other
    /// key entirely (left-command, 55, chosen as a keycode adjacent to but
    /// distinct from 54) must not be mistaken for the right-command PTT key.
    func testFlagsChangedWrongKeycodeWithCommandBitYieldsNone() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 55, commandBit: true, pttCurrentlyDown: false),
            .none
        )
    }

    /// Wrong keycode on `keyDown` (not 53) must not be mistaken for Esc.
    func testKeyDownWrongKeycodeYieldsNone() {
        XCTAssertEqual(
            decide(type: .keyDown, keycode: 51, commandBit: false, pttCurrentlyDown: false),
            .none
        )
    }

    /// `keyUp` is not a type this decision table dispatches on at all (PTT
    /// up/down comes from `flagsChanged` on a modifier key, not `keyDown`/
    /// `keyUp`; the tap doesn't even listen for `keyUp` per the monitor's
    /// tap-creation event mask, but the pure helper must fail closed if it's
    /// ever called with one, e.g. from a future caller change).
    func testKeyUpTypeYieldsNoneRegardlessOfKeycode() {
        XCTAssertEqual(
            decide(type: .keyUp, keycode: 54, commandBit: true, pttCurrentlyDown: false),
            .none
        )
        XCTAssertEqual(
            decide(type: .keyUp, keycode: 53, commandBit: false, pttCurrentlyDown: false),
            .none
        )
    }

    /// Esc while PTT is already held down: still `.esc`, not `.none` and not
    /// `.pttUp` — Esc is an independent cancel signal, not a substitute
    /// release. (Subsumed logically by
    /// `testKeyDownEscIgnoresCommandBitAndPttCurrentlyDownState` above, which
    /// pins `pttCurrentlyDown: true`; kept as its own named test since the
    /// task-8 contract calls this row out explicitly by name as a required
    /// "else" case.)
    func testKeyDownEscWhilePttCurrentlyDownYieldsEscNotNone() {
        XCTAssertEqual(
            decide(type: .keyDown, keycode: 53, commandBit: false, pttCurrentlyDown: true),
            .esc
        )
    }

    // MARK: - Other event types entirely

    /// An event type the tap never even requests (e.g. `leftMouseDown`) must
    /// also fail closed to `.none` rather than trap or default to some other
    /// case.
    func testUnrelatedEventTypeYieldsNone() {
        XCTAssertEqual(
            decide(type: .leftMouseDown, keycode: 54, commandBit: true, pttCurrentlyDown: false),
            .none
        )
    }
}
