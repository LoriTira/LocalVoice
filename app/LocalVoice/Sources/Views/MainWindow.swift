import SwiftUI

/// The four sidebar destinations, in sidebar order. `rawValue` doubles as
/// both the sidebar row label and the SF Symbol name lookup key below —
/// kept as a `String` (not the symbol name itself) since the label is
/// user-facing text and the symbol is a separate presentation detail.
enum SidebarItem: String, CaseIterable, Identifiable {
    case talk = "Talk"
    case models = "Models"
    case settings = "Settings"
    case setup = "Setup"

    var id: String { rawValue }

    var symbolName: String {
        switch self {
        case .talk: return "mic"
        case .models: return "square.stack.3d.up"
        case .settings: return "gearshape"
        case .setup: return "lock.shield"
        }
    }
}

/// The app's root layout: a `NavigationSplitView` sidebar switching between
/// the four destinations, a `StatusBanner` above the content, and a footer
/// reporting connection state + the active language model. All four
/// destinations are fully wired: Talk (Task 5), Settings (Task 6), Models
/// and Setup (Task 7).
struct MainWindow: View {
    @Bindable var appState: AppState
    let client: EngineClient
    let hotkeyMonitor: HotkeyMonitor

    @State private var selection: SidebarItem? = .talk

    var body: some View {
        NavigationSplitView {
            List(SidebarItem.allCases, selection: $selection) { item in
                Label(item.rawValue, systemImage: item.symbolName)
                    .tag(item)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
        } detail: {
            VStack(spacing: 0) {
                StatusBanner(appState: appState)
                content
                Divider()
                footer
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch selection ?? .talk {
        case .talk:
            TalkView(appState: appState, client: client)
        case .models:
            ModelsView(appState: appState, client: client)
        case .settings:
            SettingsView(appState: appState, client: client)
        case .setup:
            SetupView(appState: appState, client: client, hotkeyMonitor: hotkeyMonitor)
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            Text(connectionLabel)
            if let modelId = currentModelId {
                Text("·")
                    .foregroundStyle(.secondary)
                Text(modelId)
            }
            Spacer()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
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

    /// Digs `llm.model` out of the raw `config` tree (`docs/gui.md`'s
    /// `ready`/`config_applied` payload shape: a nested object keyed by
    /// section, e.g. `config.llm.model`). Returns `nil` before the first
    /// `ready` arrives (`config` starts `.null`) or if the shape is ever
    /// unexpected — the footer just omits the model chip rather than
    /// showing a placeholder for missing data.
    private var currentModelId: String? {
        guard case let .object(root) = appState.config,
              case let .object(llmSection)? = root["llm"],
              case let .string(model)? = llmSection["model"]
        else { return nil }
        return model
    }
}
