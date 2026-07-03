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
        }
        .formStyle(.grouped)
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
