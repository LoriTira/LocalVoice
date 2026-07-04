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
/// The complete truth table (task-8 brief, as amended by the B8 review's
/// item 3 — see below):
/// - `(flagsChanged, 54, commandBit: true,  pttCurrentlyDown: false)` -> `.pttDown`
/// - `(flagsChanged, 54, commandBit: false, pttCurrentlyDown: true)`  -> `.pttUp`
/// - `(flagsChanged, 54, commandBit: true,  pttCurrentlyDown: true)`  -> `.pttUp`
/// - `(keyDown, 53, _, _)` -> `.esc`
/// - everything else -> `.none`
///
/// **B8 review, item 3:** the third row above originally read `.none` (the
/// task-8 brief's literal text) on the theory that `commandBit` still being
/// set meant right-command was still held, so nothing should fire. That
/// reasoning doesn't hold: `commandBit` is the *aggregate* command-key flag,
/// true whenever *either* command key is down. Holding left-⌘ and tapping
/// right-⌘ while it's held, then releasing right-⌘ first, produces exactly
/// this row — keycode 54 identifies a right-command transition,
/// `pttCurrentlyDown` is true, but `commandBit` is still true because left-⌘
/// is still held. Under the old `.none` row this swallowed right-⌘'s
/// release outright and left PTT stuck down. `pttCurrentlyDown` now wins
/// unconditionally on keycode 54: once PTT is down, any further transition
/// on that specific key is its release, full stop, regardless of what the
/// aggregate flag happens to read. See `decide(...)`'s doc comment in
/// `HotkeyMonitor.swift` for the complete before/after reasoning, and
/// `testDualCommandKeyReleaseFiresPttUpNotSwallowedByAggregateCommandBit`
/// below for the regression test named for this specific scenario.
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

    // MARK: - Keycode 54 while PTT is already down (B8 review, item 3)

    /// Command bit still set while PTT is already down now yields `.pttUp`,
    /// not `.none` — this is the row the B8 review's item 3 corrected (see
    /// the file-level doc comment above and `decide(...)`'s doc comment in
    /// `HotkeyMonitor.swift` for the full reasoning). `pttCurrentlyDown`
    /// wins unconditionally on keycode 54: once PTT is down, *any* further
    /// transition on that key — command bit set or clear — must be its
    /// release, because `commandBit` only ever reports the aggregate
    /// command-key state, never which command key actually moved.
    func testFlagsChangedRightCommandWithCommandBitWhileAlreadyDownYieldsPttUp() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: true, pttCurrentlyDown: true),
            .pttUp
        )
    }

    /// Regression test for the B8 review's item 3 finding, named for the
    /// concrete real-world scenario rather than the abstract truth-table
    /// cell (the test directly above covers the same cell mechanically —
    /// this one exists so the bug is discoverable by name/grep on its own).
    /// Holding left-⌘ and right-⌘ together, then releasing right-⌘ first,
    /// leaves the aggregate `.maskCommand` bit set (left-⌘ is still down)
    /// even though the right-⌘ transition this event reports *is* a
    /// release. Before this fix, the aggregate bit being true made this
    /// event fall through to `.none`, swallowing the release and leaving
    /// PTT stuck down until something unrelated (left-⌘'s own release)
    /// happened to clear the aggregate flag. It must resolve to `.pttUp`.
    func testDualCommandKeyReleaseFiresPttUpNotSwallowedByAggregateCommandBit() {
        XCTAssertEqual(
            decide(type: .flagsChanged, keycode: 54, commandBit: true, pttCurrentlyDown: true),
            .pttUp
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
