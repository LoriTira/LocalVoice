import CoreGraphics
import Foundation
import os

/// What a single tapped keyboard event should cause `HotkeyMonitor` to do —
/// the four outcomes its three callbacks (`onPttDown`/`onPttUp`/`onEsc`) can
/// produce, plus `none` for every event that isn't one of the two hotkeys in
/// a state where it means something.
enum KeyDecision: Equatable {
    case pttDown
    case pttUp
    case esc
    case none
}

/// The pure decision table behind `HotkeyMonitor`'s tap callback — see
/// `HotkeyLogicTests` for the complete truth table this implements
/// (task-8 contract, verbatim) plus every "else" row spelled out as its own
/// test. Kept as a free function (not a method) so it can be driven with
/// plain `CGEventType`/`Int64`/`Bool` values and needs no live tap, no
/// permission, and no `HotkeyMonitor` instance to test.
///
/// Right-command (keycode 54) reports its press/release state through the
/// `.maskCommand` bit of a `flagsChanged` event rather than through
/// `keyDown`/`keyUp` the way ordinary keys do — modifier keys never generate
/// `keyDown`/`keyUp` at all, only `flagsChanged` with the *aggregate* flags
/// mask after the change, which is why `commandBit` (not the event's own
/// type) is what a naive reading would use to distinguish a right-command
/// press from a release. `pttCurrentlyDown` — the caller's own last-known
/// PTT state, not anything read from the event — is threaded through so a
/// repeat `flagsChanged` with the command bit still set (e.g. another
/// modifier changing while right-command stays held) can't fire a second
/// `pttDown`.
///
/// **B8 review, item 3 — why `pttCurrentlyDown` wins over `commandBit` on
/// release, not just on the repeat-suppression case above:** `commandBit` is
/// `flags.contains(.maskCommand)` on the *aggregate* event flags, which is
/// true if *either* command key is down, not specifically right-command.
/// Holding left-⌘ and right-⌘ together and releasing only right-⌘ produces
/// `(flagsChanged, 54, commandBit: true, pttCurrentlyDown: true)` — keycode
/// 54 identifies this as a right-command transition, `pttCurrentlyDown` is
/// true so PTT is live, but `commandBit` is *still* true because left-⌘ is
/// still down. The original `!commandBit && pttCurrentlyDown` release check
/// missed this: it required the aggregate bit to be fully clear, so this
/// event fell through to the `commandBit && !pttCurrentlyDown` guard
/// (false, `pttCurrentlyDown` is true), then the second guard (false,
/// `commandBit` is true), landing on `.none` — right-⌘'s release was
/// silently swallowed and PTT stuck down until something else (left-⌘'s own
/// release, unrelated to the key the user thinks they let go of) happened to
/// clear the aggregate bit. Keycode 54 on `flagsChanged` is *always* a
/// right-command transition regardless of what `commandBit` reads (that bit
/// only ever tells you the aggregate command state, never which command key
/// moved) — so once `pttCurrentlyDown` is true, any right-command transition
/// at all must be its release; `commandBit`'s value at that point is
/// irrelevant. Checking `pttCurrentlyDown` first, unconditionally, is what
/// makes that hold: press can only be reached once release has already been
/// ruled out by `pttCurrentlyDown` being false.
///
/// Esc (keycode 53) is an ordinary key, so it reports through `keyDown`
/// directly and its decision ignores both `commandBit` and
/// `pttCurrentlyDown` (the brief's `_`/`_` wildcards) — it's an independent
/// cancel signal, not a PTT release substitute, so it fires the same way
/// whether or not PTT happens to be held.
func decide(type: CGEventType, keycode: Int64, commandBit: Bool, pttCurrentlyDown: Bool) -> KeyDecision {
    switch (type, keycode) {
    case (.flagsChanged, HotkeyMonitor.rightCommandKeycode):
        if pttCurrentlyDown {
            return .pttUp
        }
        if commandBit {
            return .pttDown
        }
        return .none

    case (.keyDown, HotkeyMonitor.escKeycode):
        return .esc

    default:
        return .none
    }
}

/// Global push-to-talk: a session-scoped, listen-only `CGEventTap` on
/// `flagsChanged` + `keyDown` that fires `onPttDown`/`onPttUp(heldMs:)` for
/// right-command and `onEsc` for Escape — regardless of which app currently
/// has keyboard focus, per `docs/gui.md`'s "the GUI owns the actual key tap
/// in Swift" (line 285) and the v1 terminal design
/// (`src/localvoice/hotkey.py`'s `pynput`-backed `HotkeyListener`, whose
/// down/up-with-held-ms shape this mirrors one-for-one on the GUI side).
///
/// **Why `.listenOnly`, not `.defaultTap`:** the tap only *observes* these
/// two keys; it never intercepts or swallows them, so right-command and Esc
/// keep working normally in whatever app is actually focused while
/// LocalVoice is running. `.listenOnly` is also the mode that keeps this
/// entirely a "does the user have Input Monitoring granted?" question with
/// no separate Accessibility requirement (`.defaultTap`, which can eat
/// events, requires more).
///
/// **Why the pure `decide(...)` helper above lives outside this class:** the
/// tap callback itself (`tapCallback`, a free `@convention(c)` function —
/// `CGEventTapCallBack`'s C function-pointer signature cannot capture Swift
/// closure context, hence the `Unmanaged<HotkeyMonitor>` refcon
/// round-trip below) requires a live, *granted* tap to exercise at all, and
/// this Mac's Input Monitoring is denied for `LocalVoice.app` (see the
/// task-8 report's denied-path acceptance evidence) — untestable in CI or
/// on this dev machine either way. Extracting the actual press/release/esc
/// *decision* into a plain value-in-value-out function moves all of the
/// logic worth unit-testing out of that untestable shell.
///
/// **Permission handling:** `CGEvent.tapCreate` returns `nil` (no tap, no
/// dialog) when Input Monitoring is denied — verified empirically against
/// this machine's actual denied state (see the task-8 report's "CGEventTap
/// creation failure" section for the live evidence). `available` reflects
/// that outcome so `SetupView`'s Input Monitoring card — which already
/// re-checks `CGPreflightListenEventAccess()` on app refocus — can also
/// retry tap creation via `refresh()` on that same signal, without the app
/// crashing or re-prompting on every launch.
@MainActor
final class HotkeyMonitor {
    /// `kVK_RightCommand` — not defined by any framework this project
    /// imports as a named constant (Carbon's `HIToolbox/Events.h` has it,
    /// but pulling in Carbon for one integer literal isn't worth the
    /// dependency), so it's a `static let` here instead, shared between
    /// `decide(...)` above and the tap callback below rather than the two
    /// duplicating the literal `54`.
    static let rightCommandKeycode: Int64 = 54

    /// `kVK_Escape`, same rationale as `rightCommandKeycode` above.
    static let escKeycode: Int64 = 53

    var onPttDown: (() -> Void)?
    var onPttUp: ((_ heldMs: Int) -> Void)?
    var onEsc: (() -> Void)?

    /// `false` when the tap could not be created — no Input Monitoring
    /// grant, most commonly. `SetupView`'s Input Monitoring card reflects
    /// this (wired in the same `didBecomeActiveNotification` handler that
    /// already re-checks `CGPreflightListenEventAccess()`) so the user sees
    /// one consistent status rather than a permission checkbox that looks
    /// granted while the actual global hotkey silently does nothing.
    private(set) var available: Bool = false

    /// Set on a `pttDown` decision, cleared on `pttUp` — both the
    /// `pttCurrentlyDown` state `decide(...)` needs and, via
    /// `ContinuousClock.now.duration(to:)` at release, the source of
    /// `onPttUp`'s `heldMs`. `nil` doubles as "not currently down," so no
    /// separate `Bool` is needed alongside it.
    private var pressStart: ContinuousClock.Instant?

    /// `nonisolated(unsafe)`: `deinit` runs `nonisolated` (Swift 6 does not
    /// allow actor-isolated `deinit` for a plain class), so it cannot touch
    /// `@MainActor`-isolated storage directly even though this whole class
    /// otherwise is — and `CFMachPort`/`CFRunLoopSource` are non-`Sendable`
    /// (unaudited legacy Core Foundation types), which is what makes that a
    /// compile error rather than a warning. The "unsafe" is escaping the
    /// compiler's *proof*, not introducing a real data race: every other
    /// access to these two properties (`refresh()`) is already
    /// actor-serialized by `@MainActor`, and `deinit` itself only runs once
    /// there are zero other references to `self` left to race with.
    private nonisolated(unsafe) var tap: CFMachPort?
    private nonisolated(unsafe) var runLoopSource: CFRunLoopSource?

    private static let logger = Logger(subsystem: "dev.localvoice", category: "hotkey")

    init() {
        refresh()
    }

    deinit {
        if let tap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        }
    }

    /// (Re)creates the tap if it isn't already live — a no-op if one is
    /// already running. Called once from `init()` and again by
    /// `LocalVoiceApp`'s `didBecomeActiveNotification` handler (the same
    /// refocus signal `SetupView` already re-checks permissions on), so
    /// granting Input Monitoring in System Settings and switching back to
    /// LocalVoice picks up the hotkey without a relaunch.
    func refresh() {
        guard tap == nil else { return }

        let eventMask: CGEventMask =
            (1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.keyDown.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: eventMask,
            callback: HotkeyMonitor.tapCallback,
            userInfo: refcon
        ) else {
            // No dialog on this path — `CGEvent.tapCreate` denied returns
            // `nil` silently (verified against this machine's actual
            // denied-Input-Monitoring state; see the task-8 report). Logged
            // once per `refresh()` call, not looped or retried
            // automatically, so a repeatedly-refocused-but-still-denied app
            // doesn't spam the system log either.
            Self.logger.notice("hotkey tap unavailable (Input Monitoring not granted)")
            available = false
            return
        }

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        self.tap = tap
        self.runLoopSource = source
        available = true
        Self.logger.notice("hotkey tap active")
    }

    /// The `CGEventTapCallBack` C function pointer registered with the tap.
    /// `refcon` round-trips the `HotkeyMonitor` instance across the C
    /// boundary (`@convention(c)` closures cannot capture Swift context
    /// directly) — `passUnretained`/`takeUnretainedValue` rather than
    /// `passRetained`/`release`, since `self` already owns the tap for at
    /// least as long as the tap can be calling back into it (the tap is torn
    /// down in `deinit` before `self` goes away, and this callback cannot
    /// fire after `CGEvent.tapEnable(tap:enable: false)`).
    ///
    /// Always returns the event unmodified (`Unmanaged.passUnretained(event)`)
    /// — required for `.listenOnly` taps, which must not swallow events;
    /// returning `nil` here would drop the keystroke from every other app,
    /// which is exactly the `.defaultTap` behavior this design deliberately
    /// avoids (see the class doc comment).
    ///
    /// `keycode`/`commandBit` are read from `event` right here, before
    /// crossing into `MainActor.assumeIsolated` below — `CGEvent` itself is
    /// not `Sendable` (another unaudited legacy Core Foundation type), so
    /// capturing it into the `@MainActor`-isolated closure that follows
    /// would hit the same "sending risks data races" diagnostic
    /// `tap`/`runLoopSource` did above. Extracting the two `Sendable`
    /// primitives this decision actually needs first, and passing only
    /// those across the boundary, sidesteps it — and happens to be exactly
    /// the split `decide(...)`'s signature already wants (plain
    /// `CGEventType`/`Int64`/`Bool`, no `CGEvent`).
    ///
    /// **B8 review, item 1 (critical):** macOS disables an event tap —
    /// without tearing it down or calling back into `refresh()` — after a
    /// timeout if the callback doesn't return promptly enough
    /// (`kCGEventTapDisabledByTimeout`), or after the user disables it from
    /// Accessibility/Input Monitoring settings
    /// (`kCGEventTapDisabledByUserInput`). Left unhandled, either one is a
    /// silent death: `available` still reads `true`, `tap` is still
    /// non-`nil`, and the global hotkey simply stops firing with no signal
    /// anywhere. A single stall on the main thread (a long synchronous call,
    /// a modal, a slow first-launch model load) is enough to trip the
    /// timeout case, so this is not a rare corner. The fix re-enables the
    /// tap right here, inline, rather than routing through `refresh()`
    /// (which no-ops when `tap != nil` and exists to *create* a tap, not
    /// revive a disabled one) or `handle`/`decide` (neither of which is
    /// event-type-shaped for this; these two types carry no keycode).
    private static let tapCallback: CGEventTapCallBack = { _, type, event, refcon in
        guard let refcon else { return Unmanaged.passUnretained(event) }
        let monitor = Unmanaged<HotkeyMonitor>.fromOpaque(refcon).takeUnretainedValue()

        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            MainActor.assumeIsolated {
                if let tap = monitor.tap {
                    CGEvent.tapEnable(tap: tap, enable: true)
                    HotkeyMonitor.logger.notice("event tap re-enabled after timeout")
                }
            }
            return Unmanaged.passUnretained(event)
        }

        let keycode = event.getIntegerValueField(.keyboardEventKeycode)
        let commandBit = event.flags.contains(.maskCommand)
        MainActor.assumeIsolated {
            monitor.handle(type: type, keycode: keycode, commandBit: commandBit)
        }
        return Unmanaged.passUnretained(event)
    }

    /// Defers to `decide(...)` for the actual decision and fires the
    /// matching callback — the only state this needs beyond `decide(...)`'s
    /// own parameters is `pressStart`, which supplies `pttCurrentlyDown` and
    /// (on release) the hold-duration measurement.
    private func handle(type: CGEventType, keycode: Int64, commandBit: Bool) {
        switch decide(type: type, keycode: keycode, commandBit: commandBit, pttCurrentlyDown: pressStart != nil) {
        case .pttDown:
            pressStart = .now
            onPttDown?()

        case .pttUp:
            let heldMs = pressStart.map(Self.heldMilliseconds(since:)) ?? 0
            pressStart = nil
            onPttUp?(heldMs)

        case .esc:
            onEsc?()

        case .none:
            break
        }
    }

    /// Milliseconds elapsed from `start` to a single, freshly-captured
    /// "now" — same shape as `TalkView.heldMilliseconds(since:)` (the
    /// on-screen button's equivalent measurement), duplicated here rather
    /// than shared since the two live in different layers (view vs. engine)
    /// and neither currently has a natural shared home for a two-line pure
    /// function.
    private static func heldMilliseconds(since start: ContinuousClock.Instant) -> Int {
        let components = start.duration(to: .now).components
        return Int(components.seconds * 1000 + components.attoseconds / 1_000_000_000_000_000)
    }
}
