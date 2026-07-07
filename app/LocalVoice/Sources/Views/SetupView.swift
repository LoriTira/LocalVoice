import SwiftUI
import AVFoundation
import CoreGraphics
import AppKit

/// Summarizes the current Microphone + Input Monitoring permission state
/// into one sentence for `SetupView`'s combined-status line — the pure,
/// directly-testable half of the two permission cards (see
/// `PermissionLogicTests`).
///
/// - `mic`: collapses `AVCaptureDevice.authorizationStatus(for: .audio)` to
///   the three states the card distinguishes — `nil` for `.notDetermined`
///   (no prompt shown yet, so there's nothing to "open System Settings" for:
///   the request button itself is the next step), `true` for `.authorized`,
///   `false` for anything else (`.denied`/`.restricted` collapse together —
///   both mean System Settings is now the only path forward).
/// - `inputMonitoring`: as of the B8 review's item 2, this is
///   `hotkeyMonitor.available` — the *operational* truth of whether the
///   `CGEventTap` is actually live — not `CGPreflightListenEventAccess()`.
///   The two disagreed on at least one real machine (preflight reporting
///   not-granted while the tap was live and the hotkey worked), which made
///   this summary line contradict the Input Monitoring card's own caption
///   underneath it. `available` is what determines whether the global
///   hotkey actually fires, which is the thing this sentence is trying to
///   tell the user about, so it wins. `SetupView`'s Input Monitoring card
///   shows the preflight/available disagreement itself, separately, only
///   when the two differ — see `inputMonitoringCard` below. Still a plain
///   `Bool`, not an optional: `available` has no "not yet asked" state
///   either (it's `false` until a tap exists, `true` once one does).
///
/// Sentence case, no emoji, no exclamation marks, per the task-7 contract —
/// the six exact strings are pinned in `PermissionLogicTests` first (those
/// strings are unchanged by the B8 review; only the caller's *source* for
/// the `inputMonitoring` boolean moved, not the pure function's behavior).
func permissionSummary(mic: Bool?, inputMonitoring: Bool) -> String {
    let micSentence: String
    switch mic {
    case .none:
        micSentence = "Microphone access has not been requested yet."
    case .some(true):
        micSentence = "Microphone access is granted."
    case .some(false):
        micSentence = "Microphone access is denied. Open System Settings to allow it."
    }
    let inputMonitoringSentence = inputMonitoring
        ? "Input Monitoring is granted."
        : "Input Monitoring is not granted."
    return "\(micSentence) \(inputMonitoringSentence)"
}

/// Maps `CGPreflightScreenCaptureAccess()`'s boolean straight to the Screen
/// Recording card's status line — the pure, testable half of that card
/// (see `PermissionLogicTests`), added in tools phase T3 Task 5. Unlike
/// Microphone's `AVAuthorizationStatus` there is no "not yet requested"
/// state to collapse here: the preflight call is already a plain
/// grant/no-grant boolean (same shape as Input Monitoring's
/// `CGPreflightListenEventAccess()`), so this is a direct two-case naming,
/// not a three-way collapse like `permissionSummary`'s `mic` parameter.
///
/// Deliberately a separate function from `permissionSummary` above, not an
/// extra parameter on it — Screen Recording gates only the optional
/// `look_at_screen` tool and has no bearing on the core push-to-talk loop,
/// so it does not belong in the mic + Input Monitoring summary sentence.
/// See `SetupView.screenRecordingCard`'s doc comment for the full
/// rationale.
///
/// Sentence case, no emoji, no exclamation marks, matching every other
/// permission string in this file.
func screenRecordingStatusText(granted: Bool) -> String {
    granted ? "Granted." : "Not granted."
}

/// Permission onboarding + engine diagnostics — the fourth sidebar
/// destination (`docs/superpowers/specs/2026-07-03-localvoice-gui-design.md`
/// § Views: "Setup — permission checks with deep links … live re-check on
/// focus, engine status and versions"). Four cards:
///
/// 1. **Microphone** — `AVCaptureDevice.authorizationStatus(for: .audio)`,
///    a request button while `.notDetermined`, and a deep link to the
///    Microphone privacy pane once a decision exists (request buttons don't
///    fire twice — macOS only shows the system prompt on the *first*
///    request per app; a second call while `.denied` is silently a no-op).
/// 2. **Input Monitoring** — headline status is `hotkeyMonitor.available`
///    (the tap's own operational truth, per the B8 review's item 2; see
///    `inputMonitoringCard`'s doc comment), with `CGPreflightListenEventAccess()`
///    demoted to a caption shown only when it disagrees. `CGRequestListenEventAccess()`
///    for the prompt (this API isn't gated on "not yet asked" the way
///    `AVCaptureDevice` is, so its request button is always available),
///    plus a deep link to the Input Monitoring pane.
/// 3. **Screen Recording** (tools phase T3 Task 5) — `CGPreflightScreenCaptureAccess()`
///    for status, `CGRequestScreenCaptureAccess()` for the prompt, and a
///    deep link to the Screen Recording pane. Gates only the optional
///    `look_at_screen` tool, not the core push-to-talk loop — so, unlike
///    the first two cards, its grant state is deliberately left out of
///    `summaryLine` / `permissionSummary`; see `screenRecordingCard`'s doc
///    comment for why.
/// 4. **Engine** — live connection state, the protocol version this build
///    speaks, an editable dev-checkout path (`UserDefaults`-backed, same
///    key `LocalVoiceApp` reads at launch), and a restart button that
///    stops then restarts the engine subprocess so a path edit takes
///    effect without quitting the app.
///
/// All three permission statuses re-check on `NSApplication.didBecomeActiveNotification`
/// — the only reliable signal that the user might have just come back from
/// System Settings (macOS gives no direct "permission changed" callback).
struct SetupView: View {
    @Bindable var appState: AppState
    let client: EngineClient
    let hotkeyMonitor: HotkeyMonitor

    /// The protocol version this build's `EngineEvent`/`EngineCommand`
    /// types speak — `docs/gui.md`'s `ready.version` is checked against
    /// this same constant (see `AppState.protocolMismatch`), so it's
    /// displayed here rather than re-derived some other way.
    private static let supportedProtocolVersion = 1

    @State private var micStatus: AVAuthorizationStatus = .notDetermined
    @State private var inputMonitoringGranted: Bool = false
    @State private var screenRecordingGranted: Bool = false
    @State private var devCheckoutPath: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                summaryLine
                microphoneCard
                inputMonitoringCard
                screenRecordingCard
                engineCard
            }
            .padding()
        }
        .onAppear {
            refreshPermissions()
            devCheckoutPath = UserDefaults.standard.string(forKey: LocalVoiceApp.devCheckoutPathDefaultsKey) ?? ""
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            refreshPermissions()
            // Task 8: the same refocus signal that re-checks
            // `CGPreflightListenEventAccess()` above also retries tap
            // creation — a `HotkeyMonitor.refresh()` no-ops if a tap is
            // already live, so this is safe to call on every refocus, not
            // just the first one after a permission grant.
            hotkeyMonitor.refresh()
        }
    }

    // MARK: - Summary

    private var summaryLine: some View {
        // B8 review, item 2: `hotkeyMonitor.available` — the tap's own
        // operational truth — not `inputMonitoringGranted` (the preflight
        // check), is the authoritative source for the summary's Input
        // Monitoring half. See `permissionSummary`'s doc comment above.
        Text(permissionSummary(mic: micAsBool, inputMonitoring: hotkeyMonitor.available))
            .font(.callout)
            .foregroundStyle(.secondary)
    }

    /// `permissionSummary`'s `Bool?` mic parameter, derived from the live
    /// `AVAuthorizationStatus` — see that function's doc comment for the
    /// three-state collapse this mirrors.
    private var micAsBool: Bool? {
        switch micStatus {
        case .notDetermined: return nil
        case .authorized: return true
        default: return false
        }
    }

    // MARK: - Microphone card

    private var microphoneCard: some View {
        card(title: "Microphone", symbolName: "mic") {
            Text(microphoneStatusText)
                .font(.body)
            HStack {
                if micStatus == .notDetermined {
                    Button("Request Access") {
                        Task {
                            _ = await AVCaptureDevice.requestAccess(for: .audio)
                            await MainActor.run { refreshPermissions() }
                        }
                    }
                }
                Button("Open System Settings") {
                    openSystemSettings(pane: "Privacy_Microphone")
                }
            }
        }
    }

    private var microphoneStatusText: String {
        switch micStatus {
        case .notDetermined: return "Not requested yet."
        case .authorized: return "Granted."
        case .denied: return "Denied."
        case .restricted: return "Restricted by system policy."
        @unknown default: return "Unknown status."
        }
    }

    // MARK: - Input Monitoring card

    /// **B8 review, item 2:** this card used to headline
    /// `CGPreflightListenEventAccess()` (`inputMonitoringGranted`) with the
    /// tap's own `hotkeyMonitor.available` relegated to a caption
    /// underneath. On at least one real machine those two disagreed —
    /// preflight reported not-granted while the tap was in fact live and
    /// the hotkey worked — which produced a headline reading "Not granted."
    /// directly above a caption reading "Global hotkey is active.": a
    /// live, visible contradiction on the one screen whose entire job is to
    /// tell the user whether the hotkey works.
    ///
    /// Decided fix: `available` is the *operational* truth (it's
    /// `CGEvent.tapCreate` actually having succeeded, not a TCC database
    /// read), and the operational truth is what determines whether the
    /// global hotkey fires — so it becomes the headline. The old preflight
    /// read demotes to a caption shown only when it *disagrees* with
    /// `available`, since agreement is the expected case and not worth a
    /// permanent line of UI; disagreement is the "known macOS quirk" worth
    /// calling out so the user doesn't chase a permission that, per the
    /// tap's own evidence, isn't actually the problem.
    private var inputMonitoringCard: some View {
        card(title: "Input Monitoring", symbolName: "keyboard") {
            Text(hotkeyMonitor.available ? "Global hotkey is active." : "Not granted.")
                .font(.body)
            if inputMonitoringGranted != hotkeyMonitor.available {
                Text("System preflight disagrees (known macOS quirk); the tap is what matters.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                if !inputMonitoringGranted {
                    Button("Request Access") {
                        _ = CGRequestListenEventAccess()
                        refreshPermissions()
                    }
                }
                Button("Open System Settings") {
                    openSystemSettings(pane: "Privacy_ListenEvent")
                }
            }
        }
    }

    // MARK: - Screen Recording card

    /// Screen Recording gates only the optional `look_at_screen` tool
    /// (tools phase T3): when this permission is missing, `screencapture`
    /// silently writes a near-empty file instead of erroring, and the
    /// Python tool detects that and returns a graceful, spoken-friendly
    /// failure (`ToolResult(ok: false, ...)`; see
    /// `src/localvoice/tools/screenshot.py`'s `_capture_failed()`) — it
    /// never blocks or degrades the microphone/Input Monitoring
    /// push-to-talk loop that `permissionSummary` describes. That is why,
    /// unlike the mic and Input Monitoring cards above, this card's grant
    /// state is NOT folded into `summaryLine` / `permissionSummary`: that
    /// sentence is reserved for the two permissions the core interaction
    /// loop cannot work without, and adding a third, feature-scoped,
    /// often-never-needed permission to it would dilute the signal for the
    /// two that actually matter every time this pane is opened.
    ///
    /// Structurally this mirrors `inputMonitoringCard`: a status line from
    /// the preflight check, a request button shown only while not granted
    /// (the preflight API has no "not yet asked" state to gate on, so —
    /// same as Input Monitoring — the button is simply always offered
    /// until granted), and a deep link to the pane. There is no
    /// `available`-vs-preflight split like Input Monitoring's, though:
    /// there is no live resource here comparable to the `CGEventTap` whose
    /// own success could disagree with the preflight read, so the
    /// preflight boolean is the only signal there is.
    private var screenRecordingCard: some View {
        card(title: "Screen Recording", symbolName: "display") {
            Text(screenRecordingStatusText(granted: screenRecordingGranted))
                .font(.body)
            Text("Needed only for the look-at-my-screen tool. The screen is captured only when you ask.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                if !screenRecordingGranted {
                    Button("Request Access") {
                        _ = CGRequestScreenCaptureAccess()
                        refreshPermissions()
                    }
                }
                Button("Open System Settings") {
                    openSystemSettings(pane: "Privacy_ScreenCapture")
                }
            }
        }
    }

    // MARK: - Engine card

    private var engineCard: some View {
        card(title: "Engine", symbolName: "cpu") {
            LabeledContent("Connection", value: connectionLabel)
            LabeledContent("Protocol version", value: String(Self.supportedProtocolVersion))
            LabeledContent("Dev checkout path") {
                TextField("Repo root", text: $devCheckoutPath)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { saveDevCheckoutPath() }
            }
            HStack {
                Button("Restart Engine") {
                    saveDevCheckoutPath()
                    // `restart()`, not `stop()` + `start()`: the latter
                    // finishes the shared `events` stream on the way down,
                    // permanently severing the app's one consumer loop so the
                    // transcript/orb/footer would never update again after the
                    // restart. `restart()` cycles the child without finishing
                    // the stream.
                    Task { await client.restart() }
                }
            }
        }
    }

    private var connectionLabel: String {
        switch appState.connection {
        case "connecting": return "Connecting…"
        case "ready": return "Ready"
        case "engines_ready": return "Engines ready"
        case "dead": return "Disconnected"
        default: return appState.connection
        }
    }

    private func saveDevCheckoutPath() {
        UserDefaults.standard.set(devCheckoutPath, forKey: LocalVoiceApp.devCheckoutPathDefaultsKey)
    }

    // MARK: - Shared card chrome

    @ViewBuilder
    private func card(title: String, symbolName: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbolName)
                .font(.headline)
            content()
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Refresh + deep links

    private func refreshPermissions() {
        micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        inputMonitoringGranted = CGPreflightListenEventAccess()
        screenRecordingGranted = CGPreflightScreenCaptureAccess()
    }

    /// Opens the given Security & Privacy sub-pane via the
    /// `x-apple.systempreferences:` deep-link scheme — `pane` is one of
    /// `"Privacy_Microphone"` / `"Privacy_ListenEvent"` / `"Privacy_ScreenCapture"`
    /// (the last added in tools phase T3 Task 5) per the task-7 contract.
    /// Does not require Input Monitoring, Microphone, or Screen Recording
    /// access itself; this is a plain URL open.
    private func openSystemSettings(pane: String) {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)") else { return }
        NSWorkspace.shared.open(url)
    }
}
