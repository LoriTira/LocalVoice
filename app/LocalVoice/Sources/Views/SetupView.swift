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
/// - `inputMonitoring`: `CGPreflightListenEventAccess()` verbatim. Unlike
///   camera/mic, macOS has no "not yet asked" TCC state for Input
///   Monitoring — preflight just reports the current grant — so this is a
///   plain `Bool`, not an optional.
///
/// Sentence case, no emoji, no exclamation marks, per the task-7 contract —
/// the six exact strings are pinned in `PermissionLogicTests` first.
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

/// Permission onboarding + engine diagnostics — the fourth sidebar
/// destination (`docs/superpowers/specs/2026-07-03-localvoice-gui-design.md`
/// § Views: "Setup — permission checks with deep links … live re-check on
/// focus, engine status and versions"). Three cards:
///
/// 1. **Microphone** — `AVCaptureDevice.authorizationStatus(for: .audio)`,
///    a request button while `.notDetermined`, and a deep link to the
///    Microphone privacy pane once a decision exists (request buttons don't
///    fire twice — macOS only shows the system prompt on the *first*
///    request per app; a second call while `.denied` is silently a no-op).
/// 2. **Input Monitoring** — `CGPreflightListenEventAccess()` for the
///    status, `CGRequestListenEventAccess()` for the prompt (this API isn't
///    gated on "not yet asked" the way `AVCaptureDevice` is, so its request
///    button is always available), plus a deep link to the Input Monitoring
///    pane.
/// 3. **Engine** — live connection state, the protocol version this build
///    speaks, an editable dev-checkout path (`UserDefaults`-backed, same
///    key `LocalVoiceApp` reads at launch), and a restart button that
///    stops then restarts the engine subprocess so a path edit takes
///    effect without quitting the app.
///
/// Both permission statuses re-check on `NSApplication.didBecomeActiveNotification`
/// — the only reliable signal that the user might have just come back from
/// System Settings (macOS gives no direct "permission changed" callback).
struct SetupView: View {
    @Bindable var appState: AppState
    let client: EngineClient

    /// The protocol version this build's `EngineEvent`/`EngineCommand`
    /// types speak — `docs/gui.md`'s `ready.version` is checked against
    /// this same constant (see `AppState.protocolMismatch`), so it's
    /// displayed here rather than re-derived some other way.
    private static let supportedProtocolVersion = 1

    @State private var micStatus: AVAuthorizationStatus = .notDetermined
    @State private var inputMonitoringGranted: Bool = false
    @State private var devCheckoutPath: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                summaryLine
                microphoneCard
                inputMonitoringCard
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
        }
    }

    // MARK: - Summary

    private var summaryLine: some View {
        Text(permissionSummary(mic: micAsBool, inputMonitoring: inputMonitoringGranted))
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

    private var inputMonitoringCard: some View {
        card(title: "Input Monitoring", symbolName: "keyboard") {
            Text(inputMonitoringGranted ? "Granted." : "Not granted.")
                .font(.body)
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
                    Task {
                        await client.stop()
                        await client.start()
                    }
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
    }

    /// Opens the given Security & Privacy sub-pane via the
    /// `x-apple.systempreferences:` deep-link scheme — `pane` is one of
    /// `"Privacy_Microphone"` / `"Privacy_ListenEvent"` per the task-7
    /// contract. Does not require Input Monitoring or Microphone access
    /// itself; this is a plain URL open.
    private func openSystemSettings(pane: String) {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)") else { return }
        NSWorkspace.shared.open(url)
    }
}
