import SwiftUI

/// Two independent banner rows, stacked when both are active:
///
/// 1. A **persistent** red row when the connection itself is unusable —
///    `protocolMismatch` (the engine speaks a protocol version this build
///    doesn't understand) or `connection == "dead"` (no subprocess running
///    and no respawn will happen, e.g. mid-shutdown). This row has no
///    dismiss control: it reflects present tense of the connection, not a
///    one-off event, so dismissing it would just leave the user staring at
///    a broken app with no explanation.
/// 2. A **dismissable** row for `appState.banner` — the latest `error`
///    event's message, or a rejected `set_config`, etc. (`AppState.reduce`
///    sets this to `nil` on the next successful `config_applied`, but the
///    user can also clear it immediately via the dismiss button).
struct StatusBanner: View {
    @Bindable var appState: AppState

    private var isConnectionBroken: Bool {
        appState.protocolMismatch || appState.connection == "dead"
    }

    var body: some View {
        VStack(spacing: 0) {
            if isConnectionBroken {
                row(text: connectionMessage, dismissible: false, action: nil)
            }
            if let banner = appState.banner {
                row(text: banner, dismissible: true) {
                    appState.banner = nil
                }
            }
        }
    }

    private var connectionMessage: String {
        if appState.protocolMismatch {
            return "This app doesn't understand the engine's protocol version — update LocalVoice."
        }
        return "Engine connection is down."
    }

    @ViewBuilder
    private func row(text: String, dismissible: Bool, action: (() -> Void)?) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(text)
                .font(.callout)
            Spacer()
            if dismissible, let action {
                Button(action: action) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity)
        .background(Color.red.opacity(0.85))
        .foregroundStyle(.white)
    }
}
