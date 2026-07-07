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
///
/// The draft is trimmed of leading/trailing whitespace (including
/// newlines, relevant for the `llm.system_prompt` `TextEditor`) before any
/// per-type parsing — stray whitespace from a paste or a multi-line editor
/// selection shouldn't turn an otherwise-valid `"  5  "` into a parse
/// failure. The one case this changes on purpose: a draft that's *entirely*
/// whitespace trims to `""`, which keeps each type's existing empty-string
/// semantics exactly — still a parse failure for bool/int/float, still a
/// valid `.string("")` for str (see the `str` case below and its doc
/// comment above for why empty is meaningful there).
func jsonValue(fromDraft draft: String, type: String) -> JSONValue? {
    let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    switch type {
    case "bool":
        switch trimmed {
        case "true": return .bool(true)
        case "false": return .bool(false)
        default: return nil
        }
    case "int":
        guard let value = Int(trimmed) else { return nil }
        return .int(value)
    case "float":
        // `Double.init?(String)` parses "inf"/"infinity"/"nan" (case
        // insensitively) into non-finite values — none of which are valid
        // TOML/JSON config scalars the engine's `coerce` would accept, so
        // reject them here rather than sending a value that can't survive
        // the wire round-trip.
        guard let value = Double(trimmed), value.isFinite else { return nil }
        return .double(value)
    case "str":
        return .string(trimmed)
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

/// What a row should do with its draft when a fresh `config_applied`
/// changes the live config underneath it — the pure decision behind
/// `SettingRow`'s `.onChange(of: appState.config)`, kept as a free function
/// (like `jsonValue(fromDraft:type:)` and `HotkeyMonitor.decide`) so the
/// restore-defaults reconciliation is unit-testable without a live SwiftUI
/// view host.
enum DraftReconcile: Equatable {
    case keep                    // leave the draft (and pending) exactly as-is
    case clearPending            // the value we were waiting on landed; drop the indicator
    case reseed(JSONValue)       // adopt this value as the new draft (clears any pending)
}

/// Decides how a row reconciles against a new live `config` value.
///
/// WHY re-seed idle rows (the restore-defaults pitfall fix): the old
/// `.onChange` bailed at `guard let pending` for any row without a pending
/// change, so a row the user never touched this session ignored every
/// `config_applied` and kept showing its stale draft. After `reset_config`
/// clears a key, that key's row is exactly such an untouched row — it MUST
/// pick up the new (default) value. So an idle, non-editing row now re-seeds
/// whenever the live value diverges from what it's displaying. The one thing
/// still never stomped is a row the user is *actively editing* (`isEditing`):
/// an unrelated `config_applied` landing mid-keystroke must not yank the
/// half-typed draft out from under them.
///
/// - `newValue`: the key's value in the just-applied config.
/// - `currentDraftValue`: what the row's current draft would itself commit as.
/// - `pending`: the value the row is waiting to see land, or `nil` if idle.
/// - `isEditing`: whether this row's control currently has focus.
func reconcileDraft(
    newValue: JSONValue,
    currentDraftValue: JSONValue,
    pending: JSONValue?,
    isEditing: Bool
) -> DraftReconcile {
    // Never disturb a draft the user is typing into right now.
    if isEditing { return .keep }
    if let pending {
        // Pending case (pre-existing behavior, unchanged): the awaited value
        // landed → just clear the indicator; a different (external) value
        // landed → that external value wins over our stale draft.
        if newValue == pending { return .clearPending }
        if newValue != currentDraftValue { return .reseed(newValue) }
        return .clearPending
    }
    // Idle case (the fix): adopt the new value unless the row already shows it.
    return newValue == currentDraftValue ? .keep : .reseed(newValue)
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

    /// Typed drafts for the `number` widget's `int`/`float` fields (B6
    /// review item 1) — kept alongside the generic string `draft` rather
    /// than replacing it, since `draft` is still what seeds/clears
    /// `parseFailed`'s message and every other widget kind still owns it.
    /// Only one of `intDraft`/`doubleDraft` is ever live for a given row,
    /// picked by `field.type` in `numberWidget`; the other just sits at its
    /// zero value, unused. Seeded in `.onAppear` and reconciled in
    /// `.onChange(of: appState.config)` exactly like `draft` is, just
    /// through `format: .number`'s own `Int`/`Double` binding instead of a
    /// string round-trip.
    @State private var intDraft: Int = 0
    @State private var doubleDraft: Double = 0

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
            seedTypedDrafts(from: liveValue)
        }
        .onChange(of: appState.config) {
            // A fresh configApplied landed. `reconcileDraft` decides what
            // happens (see its doc comment for the full rationale) — crucially
            // it now re-seeds an *idle* row too, so a key cleared by
            // reset_config underneath an untouched row updates on screen
            // instead of showing a stale value, while never stomping a draft
            // the user is actively editing (`isFocused`).
            switch reconcileDraft(
                newValue: liveValue,
                currentDraftValue: typedDraftValue,
                pending: pendingValue,
                isEditing: isFocused
            ) {
            case .keep:
                break
            case .clearPending:
                pendingValue = nil
            case .reseed(let value):
                pendingValue = nil
                draft = draftString(from: value)
                seedTypedDrafts(from: value)
            }
        }
    }

    /// What the row's current draft would currently commit as a
    /// `JSONValue`, for the `.onChange(of: appState.config)` reconciliation
    /// above — `numberWidget` rows (the only ones with live
    /// `intDraft`/`doubleDraft` state, per the doc comment on those
    /// properties) read the typed drafts directly; every other widget kind
    /// falls back to the existing `jsonValue(fromDraft: draft, type:
    /// field.type)` string-parse path unchanged.
    private var typedDraftValue: JSONValue {
        guard field.widget == "number" else {
            return jsonValue(fromDraft: draft, type: field.type) ?? .null
        }
        switch field.type {
        case "int": return .int(intDraft)
        case "float": return .double(doubleDraft)
        default: return jsonValue(fromDraft: draft, type: field.type) ?? .null
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

    /// B6 review item 1: `type`-constrained numeric entry rather than a
    /// free-text field. `schema.py::_DEFAULT_WIDGETS` only ever pairs the
    /// `number` widget with `int`/`float` (see the doc comment above
    /// `intDraft`), so this switch is exhaustive for every field this build
    /// actually renders; the `str`/`bool` arms exist purely as a defensive
    /// fallback to the old free-text behavior rather than dropping the row,
    /// matching how the outer `widget` dispatch treats an unrecognized
    /// `field.widget`.
    ///
    /// Each typed `TextField` uses SwiftUI's `format:`-backed `value:`
    /// binding, which itself rejects/reverts non-numeric keystrokes at
    /// entry time — there is no way to type a draft this can fail to
    /// parse, so unlike every other widget's `commit()` call this goes
    /// straight to `commitTyped` with an already-valid `JSONValue`, no
    /// `jsonValue(fromDraft:type:)` round-trip needed.
    @ViewBuilder
    private var numberWidget: some View {
        LabeledContent(field.label) {
            switch field.type {
            case "int":
                TextField(field.label, value: $intDraft, format: .number.grouping(.never))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 160)
                    .focused($isFocused)
                    .onSubmit { commitTyped(.int(intDraft)) }
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commitTyped(.int(intDraft)) }
                    }
            case "float":
                TextField(
                    field.label, value: $doubleDraft,
                    format: .number.grouping(.never).precision(.fractionLength(0...3))
                )
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 160)
                    .focused($isFocused)
                    .onSubmit { commitTyped(.double(doubleDraft)) }
                    .onChange(of: isFocused) { wasFocused, nowFocused in
                        if wasFocused, !nowFocused { commitTyped(.double(doubleDraft)) }
                    }
            default:
                // Defensive: no schema field pairs `widget: "number"` with
                // a type other than int/float, but fail open to the old
                // free-text behavior rather than rendering nothing.
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

    /// `numberWidget`'s int/float commit path (B6 review item 1): `value`
    /// comes straight from a `format:`-backed `TextField` binding, which
    /// cannot hold anything `jsonValue(fromDraft:type:)` would reject — so
    /// unlike `commit()` there's no parse step and no `parseFailed` branch
    /// to reach. Otherwise identical bookkeeping to `commit()`: mark
    /// pending, send the one-key `setConfig`.
    private func commitTyped(_ value: JSONValue) {
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

    /// Seeds `intDraft`/`doubleDraft` from `value` for `numberWidget` (B6
    /// review item 1) — the typed-field counterpart to `draftString(from:)`
    /// above. A no-op for every widget other than `number` since those
    /// rows never read the typed drafts at all. `value` is whatever the
    /// caller already resolved as "current" (`liveValue` on first appear,
    /// or a fresh `configApplied`'s value on reconciliation) — this only
    /// handles pulling a numeric payload back out of it.
    ///
    /// `Int`/`Double` extraction tolerates the value arriving in the
    /// "other" numeric `JSONValue` case (e.g. `.double` for an `int`-typed
    /// field) defensively, matching `sliderWidget`'s existing
    /// `Double(draft) ?? minimum` fallback style — `JSONValue`'s own
    /// decoder tries `Int` before `Double`, so this shouldn't happen for
    /// values that actually came off the wire, but a field's `.default`
    /// (Swift-literal-constructed, not decoded) has no such guarantee.
    private func seedTypedDrafts(from value: JSONValue) {
        guard field.widget == "number" else { return }
        switch field.type {
        case "int":
            switch value {
            case .int(let i): intDraft = i
            case .double(let d): intDraft = Int(d)
            default: break
            }
        case "float":
            switch value {
            case .double(let d): doubleDraft = d
            case .int(let i): doubleDraft = Double(i)
            default: break
            }
        default:
            break
        }
    }
}
