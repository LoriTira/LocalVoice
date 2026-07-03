import SwiftUI

/// Turns a row's local text draft into the typed `JSONValue` `setConfig`
/// sends over the wire, per `docs/gui.md` § "Derived settings schema"
/// (`type` is always exactly `"bool"`, `"int"`, `"float"`, or `"str"` —
/// straight from the Python `Config` dataclass annotations, see
/// `src/localvoice/schema.py::_TYPE_NAMES`). Pure and side-effect free so
/// it's directly unit-testable without any SwiftUI/view machinery.
///
/// This is a client-side pre-check, not a re-implementation of the
/// engine's coercion: `src/localvoice/schema.py::coerce` is the actual
/// authority and still runs engine-side on whatever this sends. Bool
/// parsing matches `coerce`'s case-sensitive-off `"true"/"false"` exactly;
/// int/float reject anything `Int`/`Double`'s own `init?(String)` would
/// reject (e.g. `"3.5"` for int, matching Python `int("3.5")` raising
/// rather than truncating); `str` never fails, including on the empty
/// string, since several `str` fields (`audio.input_device`,
/// `llm.deep_model`) use `""` as a meaningful "unset" value.
///
/// Returns `nil` on parse failure so the caller can show an inline error
/// and send nothing — a bad value never reaches `setConfig`.
func jsonValue(fromDraft draft: String, type: String) -> JSONValue? {
    switch type {
    case "bool":
        switch draft {
        case "true": return .bool(true)
        case "false": return .bool(false)
        default: return nil
        }
    case "int":
        guard let value = Int(draft) else { return nil }
        return .int(value)
    case "float":
        guard let value = Double(draft) else { return nil }
        return .double(value)
    case "str":
        return .string(draft)
    default:
        // Defensive: docs/gui.md only ever emits the four types above.
        return nil
    }
}

/// Renders `draft`'s current string content back out for a fresh
/// `JSONValue` — the inverse companion to `jsonValue(fromDraft:type:)`,
/// used to seed a row's editable draft from the live config value (and to
/// re-seed it after an external `configApplied` supersedes an in-flight
/// edit). `.string` passes through verbatim (no added quoting); numbers use
/// Swift's default `String(describing:)`-equivalent formatting, which round
/// trips through `jsonValue` for every value the engine can actually send
/// (schema fields are plain scalars, never NaN/infinity).
private func draftString(from value: JSONValue) -> String {
    switch value {
    case .string(let s): return s
    case .int(let i): return String(i)
    case .double(let d): return String(d)
    case .bool(let b): return b ? "true" : "false"
    case .object, .array, .null: return ""
    }
}

/// One editable row for a single `SchemaField`, dispatched to the widget
/// per `docs/gui.md`'s widget-to-control mapping. Every widget owns a local
/// `draft` string seeded from the field's current value and commits back
/// through `setConfig` — see the `commit()` doc comment for exactly when
/// that happens per widget kind.
///
/// `field` only supplies the row's *identity* and static metadata (key,
/// type, label, help, min/max/step) — the live value used to seed/reseed
/// `draft` and to detect "has my pending send landed yet" comes from
/// `appState.config` instead, dug out via `field.key`'s dotted path. This
/// matters because `schema` (and therefore `field.value`) is only ever
/// refreshed by a fresh `ready`, never by `config_applied` (see
/// `AppState.reduce`) — using `field.value` as the pending-comparison
/// baseline would make the indicator stick forever after the very first
/// edit.
struct SettingRow: View {
    let field: SchemaField
    @Bindable var appState: AppState
    let client: EngineClient

    @State private var draft: String = ""
    @FocusState private var isFocused: Bool

    /// The exact value most recently sent to `setConfig`, kept until
    /// `appState.config` reflects it (a fresh `configApplied`) — the
    /// pending indicator is `pendingValue != nil`. Cleared either by that
    /// convergence (`.onChange(of: appState.config)` below) or by a later
    /// edit superseding it (a fresh commit just overwrites it with the new
    /// pending value).
    @State private var pendingValue: JSONValue?

    /// True when the most recent commit attempt failed `jsonValue` parsing
    /// — shown as inline text instead of sending anything. Cleared on the
    /// next keystroke/commit attempt.
    @State private var parseFailed = false

    private static let modelKeys: Set<String> = ["llm.deep_model"]

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                widget

                if parseFailed {
                    Text("Couldn't parse \"\(draft)\" as \(field.type) — nothing was sent.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                if field.widget == "key_capture" || field.widget == "device_picker" {
                    Text(field.help)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Captured live in a later phase.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if !field.help.isEmpty {
                    Text(field.help)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if pendingValue != nil {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel("\(field.label) change pending")
            }
        }
        .onAppear {
            draft = draftString(from: liveValue)
            if field.widget == "model_picker" {
                Task { await client.send(.listModels) }
            }
        }
        .onChange(of: appState.config) {
            // A fresh configApplied landed. If it carries the value we're
            // waiting on, the pending edit is done — clear the indicator
            // and let the (now-authoritative) live value own the draft
            // again. If some other actor changed this same key to a
            // different value in the meantime, that external value wins
            // over our stale local draft too.
            guard let pending = pendingValue else { return }
            let current = liveValue
            if current == pending {
                pendingValue = nil
            } else if current != jsonValue(fromDraft: draft, type: field.type) {
                pendingValue = nil
                draft = draftString(from: current)
            }
        }
    }

    // MARK: - Widget dispatch

    @ViewBuilder
    private var widget: some View {
        switch field.widget {
        case "toggle":
            toggleWidget
        case "number":
            numberWidget
        case "slider":
            sliderWidget
        case "model_picker":
            modelPickerWidget
        case "voice_picker":
            voicePickerWidget
        case "text":
            textWidget
        case "key_capture", "device_picker":
            plainTextWidget
        default:
            // Defensive fallback for a widget name this build doesn't know
            // about yet — still editable rather than silently dropped.
            plainTextWidget
        }
    }

    private var toggleWidget: some View {
        Toggle(field.label, isOn: Binding(
            get: { draft == "true" },
            set: { newValue in
                draft = newValue ? "true" : "false"
                commit()
            }
        ))
    }

    private var numberWidget: some View {
        LabeledContent(field.label) {
            TextField(field.label, text: $draft)
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 160)
                .focused($isFocused)
                .onSubmit { commit() }
                .onChange(of: isFocused) { wasFocused, nowFocused in
                    if wasFocused, !nowFocused { commit() }
                }
        }
    }

    private var sliderWidget: some View {
        let minimum = field.minimum ?? 0
        let maximum = field.maximum ?? max(minimum + 1, 1)
        let step = field.step ?? 1
        return VStack(alignment: .leading, spacing: 2) {
            Text(field.label)
            HStack {
                Slider(
                    value: Binding(
                        get: { Double(draft) ?? minimum },
                        set: { newValue in
                            draft = Self.formatSliderValue(newValue)
                            commit()
                        }
                    ),
                    in: minimum...maximum,
                    step: step
                )
                Text(draft)
                    .font(.caption)
                    .monospacedDigit()
                    .frame(minWidth: 40, alignment: .trailing)
            }
        }
    }

    /// Whole-number slider values (e.g. `audio.rebuffer_ms`, `type ==
    /// "int"`) format without a trailing `.0` so the committed draft still
    /// parses via `jsonValue`'s `Int(draft)` path; fractional values
    /// (`tts.speed`, `type == "float"`) keep one decimal place, matching
    /// the `step: 0.1` granularity `docs/gui.md` documents for it.
    private static func formatSliderValue(_ value: Double) -> String {
        if value.rounded() == value {
            return String(Int(value))
        }
        return String(format: "%.1f", value)
    }

    private var modelPickerWidget: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(field.label)
            HStack {
                Menu(draft.isEmpty ? "Choose a model" : draft) {
                    ForEach(appState.models) { model in
                        Button(model.id) {
                            draft = model.id
                            commit()
                        }
                    }
                    if Self.modelKeys.contains(field.key) {
                        Button("None") {
                            draft = ""
                            commit()
                        }
                    }
                }
                TextField("HF repo id or local path", text: $draft)
                    .textFieldStyle(.roundedBorder)
                    .focused($isFocused)
                    .onSubmit { commit() }
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commit() }
                    }
            }
        }
    }

    private var voicePickerWidget: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(field.label)
            HStack {
                TextField(field.label, text: $draft)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .focused($isFocused)
                    .onSubmit { commit() }
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commit() }
                    }
                Button("Preview") {
                    Task { await client.send(.previewVoice(voice: draft)) }
                }
            }
        }
    }

    /// `text`: single-line `TextField` by default, upgraded to a
    /// multi-line `TextEditor` specifically for `llm.system_prompt` once
    /// its current draft is longer than 60 characters — short custom
    /// prompts stay a compact single line, but the field never forces a
    /// long prompt through a horizontally-scrolling one-liner.
    @ViewBuilder
    private var textWidget: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(field.label)
            if field.key == "llm.system_prompt", draft.count > 60 {
                TextEditor(text: $draft)
                    .frame(minHeight: 80, maxHeight: 160)
                    .font(.body)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.gray.opacity(0.3)))
                    .focused($isFocused)
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commit() }
                    }
            } else {
                TextField(field.label, text: $draft)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .focused($isFocused)
                    .onSubmit { commit() }
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commit() }
                    }
            }
        }
    }

    /// `key_capture`/`device_picker` and the unrecognized-widget fallback:
    /// a plain `TextField`, no special commit behavior beyond the shared
    /// submit/focus-loss handling every text control above already uses.
    private var plainTextWidget: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(field.label)
            TextField(field.label, text: $draft)
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .focused($isFocused)
                .onSubmit { commit() }
                .onChange(of: isFocused) { wasFocused, nowFocused in
                    if wasFocused, !nowFocused { commit() }
                }
        }
    }

    // MARK: - Commit

    /// Parses `draft` via `jsonValue(fromDraft:type:)` and, on success,
    /// sends it as a one-key `setConfig` and marks the row pending. On
    /// failure, sets `parseFailed` for the inline error and sends nothing
    /// — per contract, a bad value never reaches the wire.
    ///
    /// A no-op re-commit of the value the row is already waiting on (e.g.
    /// a stray focus-loss firing right after a slider drag already
    /// committed the same number) is allowed to resend — `setConfig` is
    /// idempotent engine-side (it just rewrites the same value and applies
    /// it again), so this stays simple rather than tracking "did the draft
    /// actually change since the last commit" separately.
    private func commit() {
        guard let value = jsonValue(fromDraft: draft, type: field.type) else {
            parseFailed = true
            pendingValue = nil
            return
        }
        parseFailed = false
        pendingValue = value
        Task { await client.send(.setConfig([field.key: value])) }
    }

    // MARK: - Live value lookup

    /// Digs `field.key`'s current value out of `appState.config`'s nested
    /// `section -> field` object tree (`docs/gui.md`'s `ready`/
    /// `config_applied` payload shape, e.g. `config.llm.think`) — the same
    /// two-level lookup `MainWindow.currentModelId` already does for
    /// `llm.model`, generalized to any dotted key. Falls back to
    /// `field.default` if `config` doesn't have the shape yet (before the
    /// first `ready`) or the key is unexpectedly absent, so a row always
    /// has *something* sane to seed its draft from.
    private var liveValue: JSONValue {
        let parts = field.key.split(separator: ".", maxSplits: 1).map(String.init)
        guard parts.count == 2,
              case let .object(root) = appState.config,
              case let .object(section)? = root[parts[0]],
              let value = section[parts[1]]
        else { return field.default }
        return value
    }
}
