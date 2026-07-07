import SwiftUI

/// Whether the on-screen Stop button can do anything: a response is being
/// generated ("processing") or spoken ("speaking"). While idle there is
/// nothing to stop, and while listening the recording is ended by releasing
/// push-to-talk, not by Stop. Free function so the mapping is testable
/// without constructing the view (same pattern as `decide(...)` for the
/// hotkey).
func stopButtonEnabled(assistantState: String) -> Bool {
    assistantState == "processing" || assistantState == "speaking"
}

/// The main conversation screen: state orb, transcript, mic level while
/// listening, last-turn latency chips, the on-screen hold-to-talk button,
/// and a Stop button that interrupts the current response. Every outgoing
/// command (`pttDown`/`pttUp`/`esc`) goes through the same `EngineClient`
/// the app shell started — this view holds no process or protocol state of
/// its own beyond what `AppState` already tracks.
///
/// The Stop button exists because the global Esc path depends on `keyDown`
/// CGEventTap delivery, which on at least one real machine is not delivered
/// to this app's tap even though `flagsChanged` (push-to-talk) is — an
/// on-screen button goes straight to the `esc` protocol command and cannot
/// be affected by input-tap delivery at all.
///
/// Each command send below is a `Task { await client.send(...) }`
/// constructed inline at its gesture/key callback, with the `EngineCommand`
/// literal written directly inside the closure rather than built earlier
/// and passed in through a parameter — routing it through an intermediate
/// synchronous function parameter first was observed to trip the Swift 6
/// compiler's "sending value risks causing data races" diagnostic under
/// `xcodebuild` (but not in isolated `swiftc -typecheck` repros of the same
/// shape, so this reads as a real toolchain false-positive rather than an
/// actual soundness issue — `EngineCommand` is implicitly `Sendable`, its
/// payload is exclusively `String`/`Int`/`JSONValue`); constructing the
/// value at its point of use sidesteps it.
struct TalkView: View {
    @Bindable var appState: AppState
    let client: EngineClient

    /// Set on the `DragGesture`'s first change (press) and cleared on end
    /// (release) — guards `pttDown` from firing again on every subsequent
    /// drag-change callback while the button is held, and gives the
    /// release handler a start time to measure the hold duration from.
    @State private var pressStart: ContinuousClock.Instant?

    var body: some View {
        VStack(spacing: 16) {
            orb

            if let activity = appState.toolActivity {
                Text(activity)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(Color.gray.opacity(0.15))
                    .clipShape(Capsule())
                    .accessibilityLabel("Tool activity: \(activity)")
            }

            transcript

            if appState.assistantState == "listening" {
                ProgressView(value: appState.micLevel, total: 1)
                    .frame(maxWidth: 240)
            }

            if let latency = appState.lastLatency {
                latencyChips(latency)
            }

            HStack(spacing: 12) {
                holdToTalkButton
                stopButton
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onKeyPress(.escape) {
            Task { await client.send(.esc) }
            return .handled
        }
    }

    // MARK: - State orb

    private var orbColor: Color {
        switch appState.assistantState {
        case "listening": return .red
        case "processing": return .orange
        case "speaking": return .green
        default: return .gray // "idle" and any unrecognized value
        }
    }

    private var orb: some View {
        Circle()
            .fill(orbColor)
            .frame(width: 44, height: 44)
            .accessibilityLabel("Assistant state: \(appState.assistantState)")
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(appState.turns) { turn in
                    turnRow(turn)
                }
            }
            .padding(.horizontal, 4)
        }
    }

    @ViewBuilder
    private func turnRow(_ turn: Turn) -> some View {
        let isUser = turn.role == "user"
        HStack {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                if let reasoning = turn.reasoning, !reasoning.isEmpty {
                    DisclosureGroup("Reasoning") {
                        Text(reasoning)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.caption)
                }
                Text(turn.text)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isUser ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            if !isUser { Spacer(minLength: 40) }
        }
    }

    // MARK: - Latency chips

    @ViewBuilder
    private func latencyChips(_ latency: Latency) -> some View {
        HStack(spacing: 8) {
            latencyChip(label: "STT", seconds: latency.stt)
            latencyChip(label: "First word", seconds: latency.ttft)
            latencyChip(label: "Total", seconds: latency.total)
        }
    }

    private func latencyChip(label: String, seconds: Double) -> some View {
        Text("\(label) \(String(format: "%.2fs", seconds))")
            .font(.caption)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Color.gray.opacity(0.15))
            .clipShape(Capsule())
    }

    // MARK: - Hold-to-talk

    /// `DragGesture(minimumDistance: 0)` fires on the very first touch/click
    /// with no minimum travel required, which is what makes it usable as a
    /// press/release detector for a stationary on-screen button — a
    /// `TapGesture`/`Button` only reports the completed tap, with no signal
    /// at press-down and no hold-duration measurement.
    private var holdToTalkButton: some View {
        Text("Hold to talk")
            .font(.headline)
            .frame(width: 160, height: 44)
            .background(pressStart == nil ? Color.accentColor : Color.accentColor.opacity(0.6))
            .foregroundStyle(.white)
            .clipShape(Capsule())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard pressStart == nil else { return }
                        pressStart = ContinuousClock.now
                        Task { await client.send(.pttDown) }
                    }
                    .onEnded { _ in
                        let heldMs = pressStart.map(Self.heldMilliseconds(since:)) ?? 0
                        pressStart = nil
                        Task { await client.send(.pttUp(heldMs: heldMs)) }
                    }
            )
    }

    // MARK: - Stop

    private var stopButton: some View {
        Button {
            Task { await client.send(.esc) }
        } label: {
            Label("Stop", systemImage: "stop.fill")
                .font(.headline)
                .frame(height: 44)
                .padding(.horizontal, 16)
        }
        .buttonStyle(.bordered)
        .tint(.red)
        .disabled(!stopButtonEnabled(assistantState: appState.assistantState))
        .help("Stop the current response (Esc)")
    }

    /// Milliseconds elapsed from `start` to a single, freshly-captured
    /// "now" — captured once here (not by calling `.now` separately at each
    /// call site) so the duration reflects one consistent instant rather
    /// than two racing clock reads.
    private static func heldMilliseconds(since start: ContinuousClock.Instant) -> Int {
        let components = start.duration(to: .now).components
        return Int(components.seconds * 1000 + components.attoseconds / 1_000_000_000_000_000)
    }
}
