import SwiftUI

/// The derived settings form: every field `appState.schema` describes,
/// rendered generically and grouped by `section` — see `docs/gui.md` §
/// "Derived settings schema". Nothing here is hand-written per config key;
/// a new field added to the Python `Config` dataclasses shows up here with
/// no Swift change, which is the entire point of shipping a schema over
/// the wire instead of a fixed protocol.
struct SettingsView: View {
    @Bindable var appState: AppState
    let client: EngineClient

    /// Drives the "Restore default settings?" confirmation alert.
    @State private var showRestoreConfirm = false

    /// The four config keys `reset_config` is told to KEEP when restoring
    /// defaults. WHY keep model assignments across a reset: the shipped
    /// defaults (`localvoice.toml`) name a Hugging Face repo id
    /// (`mlx-community/Qwen3.6-35B-A3B-4bit`, `prince-canuma/Kokoro-82M`, …)
    /// that is typically NOT already on disk. Reverting a model key to that
    /// default would make the very next engine reload silently start a
    /// ~19 GB download — a hostile surprise for someone who just wanted to
    /// undo a speed/prompt tweak. Keeping the user's actual model paths makes
    /// "Restore defaults" a safe, instant operation.
    private static let keptModelKeys = ["llm.model", "llm.deep_model", "stt.model", "tts.model"]

    var body: some View {
        Form {
            ForEach(sections, id: \.self) { section in
                Section(section) {
                    // `SchemaField` isn't `Identifiable` (it's a pure
                    // protocol-contract type from Task 2/4, out of scope to
                    // extend here) — `key` is already its natural unique
                    // id, so `ForEach(_:id:)` uses that directly instead.
                    ForEach(fields(in: section), id: \.key) { field in
                        SettingRow(field: field, appState: appState, client: client)
                    }
                }
            }
            // Spec §6: the derived form's footer names the overlay file where
            // edits land. `set_config` writes changes to `localvoice.local.toml`
            // (the local overlay), leaving the committed defaults untouched.
            Section {
                Button("Restore defaults", role: .destructive) {
                    showRestoreConfirm = true
                }
                Text("Changes are saved to localvoice.local.toml, your local settings overlay.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .alert("Restore default settings?", isPresented: $showRestoreConfirm) {
            Button("Cancel", role: .cancel) {}
            Button("Restore", role: .destructive) {
                Task { await client.send(.resetConfig(keep: Self.keptModelKeys)) }
            }
        } message: {
            Text("All settings return to their defaults. Model assignments are kept — change models in the Models tab.")
        }
        .onAppear {
            // B6 review item 3: fetch the model list once per Settings
            // appearance here, not per `model_picker` row — this view has
            // three such rows (`stt.model`, `llm.model`, `llm.deep_model`),
            // and a per-row `.onAppear` trigger sends `listModels` three
            // times on first render plus again on every re-appear a scroll
            // can cause. `SettingRow` no longer sends it at all.
            Task { await client.send(.listModels) }
        }
    }

    /// Section names in first-seen order from `appState.schema` — the
    /// engine already emits fields grouped by section in a fixed order
    /// (`src/localvoice/schema.py::_SECTIONS`: Speech recognition,
    /// Language model, Speech, Keys, Audio), so preserving that order here
    /// rather than alphabetizing keeps the form's section order matching
    /// what a reader of that file would expect, and stays stable across
    /// re-renders without needing a separately-maintained ordering list on
    /// the Swift side.
    private var sections: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for field in appState.schema where !seen.contains(field.section) {
            seen.insert(field.section)
            ordered.append(field.section)
        }
        return ordered
    }

    private func fields(in section: String) -> [SchemaField] {
        appState.schema.filter { $0.section == section }
    }
}
