import SwiftUI

/// The three config keys a model on disk can be assigned to via the
/// per-row "Use for…" menu, paired with the menu's display label. Order
/// here is the menu's display order.
private let assignmentTargets: [(label: String, key: String)] = [
    ("Language model", "llm.model"),
    ("Speech recognition", "stt.model"),
    ("Speech synthesis", "tts.model"),
]

/// Installed-model browser + download manager — the second sidebar
/// destination. Fetches `appState.models` once per appearance, renders it
/// as a table (id, kind badge, size in GB), and lets each row be assigned
/// to any of the three model-backed config keys via `setConfig`. Below the
/// table, a repo-id field + Download button kicks off `downloadModel`;
/// `appState.downloads` renders live progress for whatever's in flight, and
/// a completion (the reducer removes the repo's key once `done` — see
/// `AppState.reduce`) triggers a fresh `listModels` so the newly-landed
/// model shows up in the table without a manual refresh.
struct ModelsView: View {
    @Bindable var appState: AppState
    let client: EngineClient

    @State private var repoIdDraft: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            table
            Divider()
            downloadSection
        }
        .onAppear {
            Task { await client.send(.listModels) }
        }
        .onChange(of: Array(appState.downloads.keys).sorted()) {
            // A download's key is removed from `appState.downloads` the
            // moment its `download_progress` arrives with `done: true`
            // (see `AppState.reduce`) — by the time this fires, there's no
            // separate "it just finished" signal left to key off other than
            // the key-set itself changing. Any change here (an addition
            // from starting a new download, or a removal from one
            // finishing) triggers a refresh; `list_models` is a cheap
            // re-scan, so refreshing on a new-download start too is
            // harmless rather than worth special-casing away.
            Task { await client.send(.listModels) }
        }
    }

    // MARK: - Table

    private var table: some View {
        Table(appState.models) {
            TableColumn("ID", value: \.id)
            TableColumn("Kind") { model in
                kindBadge(model.kind)
            }
            TableColumn("Size (GB)") { model in
                Text(String(format: "%.1f", model.sizeGb))
                    .monospacedDigit()
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            TableColumn("") { model in
                useForMenu(model)
            }
        }
    }

    private func kindBadge(_ kind: String) -> some View {
        Text(kind.uppercased())
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.accentColor.opacity(0.15))
            .clipShape(Capsule())
    }

    /// The per-row assignment menu: picking a target sends a one-key
    /// `setConfig` with this row's `path` (not `id`) — `path` is the actual
    /// filesystem location the engine loads from, `id` is just the
    /// `publisher/name` display label (see `InstalledModel`).
    private func useForMenu(_ model: InstalledModel) -> some View {
        Menu("Use for…") {
            ForEach(assignmentTargets, id: \.key) { target in
                Button(target.label) {
                    Task { await client.send(.setConfig([target.key: .string(model.path)])) }
                }
            }
        }
    }

    // MARK: - Download section

    private var downloadSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Download a model")
                .font(.headline)
            HStack {
                TextField("Hugging Face repo id", text: $repoIdDraft)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(startDownload)
                Button("Download", action: startDownload)
                    .disabled(repoIdDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            ForEach(Array(appState.downloads.keys).sorted(), id: \.self) { repo in
                downloadRow(repo: repo, pct: appState.downloads[repo] ?? nil)
            }
        }
        .padding()
    }

    private func downloadRow(repo: String, pct: Double?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(repo)
                .font(.caption)
            if let pct {
                ProgressView(value: pct, total: 100)
            } else {
                ProgressView()
            }
        }
    }

    private func startDownload() {
        let repo = repoIdDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repo.isEmpty else { return }
        Task { await client.send(.downloadModel(repo: repo)) }
        repoIdDraft = ""
    }
}
