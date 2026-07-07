import XCTest
@testable import LocalVoice

/// Tests for `jsonValue(fromDraft:type:)`, the pure helper that turns a
/// row's local text draft into the typed `JSONValue` `setConfig` sends —
/// see `docs/gui.md` § "Derived settings schema" for the `type` strings
/// (`bool`/`int`/`float`/`str`) this must agree with, and
/// `src/localvoice/schema.py::coerce` for the engine-side counterpart this
/// stays a client-side pre-check for (a value this helper accepts must
/// still coerce engine-side; a value it rejects is never sent at all).
final class SettingsLogicTests: XCTestCase {
    // MARK: - bool

    func testBoolTrueDraftYieldsBoolTrue() {
        XCTAssertEqual(jsonValue(fromDraft: "true", type: "bool"), .bool(true))
    }

    func testBoolFalseDraftYieldsBoolFalse() {
        XCTAssertEqual(jsonValue(fromDraft: "false", type: "bool"), .bool(false))
    }

    func testBoolDraftWithOtherTextFailsToParse() {
        XCTAssertNil(jsonValue(fromDraft: "yes", type: "bool"))
        XCTAssertNil(jsonValue(fromDraft: "1", type: "bool"))
        XCTAssertNil(jsonValue(fromDraft: "", type: "bool"))
    }

    // MARK: - int

    func testIntDraftYieldsInt() {
        XCTAssertEqual(jsonValue(fromDraft: "2048", type: "int"), .int(2048))
    }

    func testIntDraftAcceptsNegativeValues() {
        XCTAssertEqual(jsonValue(fromDraft: "-12", type: "int"), .int(-12))
    }

    func testIntDraftWithNonIntegerTextFailsToParse() {
        XCTAssertNil(jsonValue(fromDraft: "abc", type: "int"))
        XCTAssertNil(jsonValue(fromDraft: "3.5", type: "int"), "a float-looking draft must not silently truncate to an int")
        XCTAssertNil(jsonValue(fromDraft: "", type: "int"))
    }

    // MARK: - float

    func testFloatDraftYieldsDouble() {
        XCTAssertEqual(jsonValue(fromDraft: "1.2", type: "float"), .double(1.2))
    }

    func testFloatDraftAcceptsWholeNumberText() {
        XCTAssertEqual(jsonValue(fromDraft: "2", type: "float"), .double(2.0))
    }

    func testFloatDraftWithNonNumericTextFailsToParse() {
        XCTAssertNil(jsonValue(fromDraft: "abc", type: "float"))
        XCTAssertNil(jsonValue(fromDraft: "", type: "float"))
    }

    // MARK: - str

    func testStrDraftPassesThroughVerbatim() {
        XCTAssertEqual(jsonValue(fromDraft: "af_heart", type: "str"), .string("af_heart"))
    }

    func testStrDraftAllowsEmptyString() {
        // Several str fields (e.g. audio.input_device, llm.deep_model) use ""
        // as a meaningful "unset" value per docs/gui.md, so empty text is not
        // a parse failure the way it is for bool/int/float.
        XCTAssertEqual(jsonValue(fromDraft: "", type: "str"), .string(""))
    }

    // MARK: - unknown type

    /// A `type` string outside the four the schema ever emits (defensive —
    /// `docs/gui.md` only documents bool/int/float/str) must fail closed,
    /// not silently pass the draft through as a string.
    func testUnknownTypeFailsToParse() {
        XCTAssertNil(jsonValue(fromDraft: "anything", type: "enum"))
    }

    // MARK: - whitespace trimming (B6 review)

    /// Leading/trailing whitespace around an otherwise-valid int draft
    /// (e.g. from a paste) must not turn it into a parse failure.
    func testIntDraftWithSurroundingWhitespaceTrimsAndParses() {
        XCTAssertEqual(jsonValue(fromDraft: "  5  ", type: "int"), .int(5))
    }

    /// Same trimming behavior for bool, including the case-sensitive-off
    /// wire values (`coerce`'s `"true"/"false"` match).
    func testBoolDraftWithSurroundingWhitespaceTrimsAndParses() {
        XCTAssertEqual(jsonValue(fromDraft: " true ", type: "bool"), .bool(true))
    }

    /// `Double.init?(String)` happily parses "inf"/"nan" into non-finite
    /// values — neither is a value the engine's TOML/JSON-backed config can
    /// actually hold, so both must be rejected rather than sent.
    func testFloatDraftOfInfOrNanFailsToParse() {
        XCTAssertNil(jsonValue(fromDraft: "inf", type: "float"))
        XCTAssertNil(jsonValue(fromDraft: "nan", type: "float"))
    }

    /// A draft that's entirely whitespace trims to "" — documenting the
    /// exact current intent: this is *still* a meaningful, valid "unset"
    /// value for str (matching `testStrDraftAllowsEmptyString`'s untrimmed
    /// empty-string case), not a new parse failure introduced by trimming.
    func testStrDraftOfOnlyWhitespaceTrimsToEmptyStringPassthrough() {
        XCTAssertEqual(jsonValue(fromDraft: "  ", type: "str"), .string(""))
    }

    // MARK: - Row reconciliation on config_applied (Restore-defaults pitfall)

    /// THE pitfall the restore-defaults feature must fix: a row the user
    /// never touched this session (no pending change) previously ignored
    /// every `config_applied` — its `onChange(of: config)` bailed at
    /// `guard let pending`. So after a reset cleared that key underneath it,
    /// the row kept showing its STALE pre-reset value. An idle row must
    /// re-seed from the new live value.
    func testIdleRowReseedsFromNewConfigValue() {
        let outcome = reconcileDraft(
            newValue: .double(1.0),      // reset reverted tts.speed to its default
            currentDraftValue: .double(1.4),  // what the idle row is still showing
            pending: nil,                // user never touched this row this session
            isEditing: false
        )
        XCTAssertEqual(outcome, .reseed(.double(1.0)),
                       "an idle row must adopt the new config value, not keep its stale draft")
    }

    /// A row the user is actively editing right now (focused) must NOT have
    /// its in-progress draft stomped by a config_applied that arrived for
    /// some other reason — the spec's "without stomping a draft the user is
    /// actively editing in that moment" clause.
    func testActivelyEditedRowIsNotStompedByConfigChange() {
        let outcome = reconcileDraft(
            newValue: .double(1.0),
            currentDraftValue: .double(1.7),  // half-typed value the user is entering
            pending: nil,
            isEditing: true
        )
        XCTAssertEqual(outcome, .keep,
                       "a focused row keeps the user's in-flight draft")
    }

    /// The normal pending-converged case still works: the row was waiting on
    /// a value it sent, and the config_applied carries exactly that value —
    /// clear the pending indicator, the live value now owns the draft.
    func testPendingConvergedClearsPendingIndicator() {
        let outcome = reconcileDraft(
            newValue: .bool(true),
            currentDraftValue: .bool(true),  // draft already matches what landed
            pending: .bool(true),
            isEditing: false
        )
        XCTAssertEqual(outcome, .clearPending,
                       "when the landed value matches both draft and pending, just drop the indicator")
    }

    /// A pending row where some *other* actor changed the same key to a
    /// different value: the external value wins over the stale local draft,
    /// and the pending indicator clears.
    func testPendingButExternalValueWinsAndReseeds() {
        let outcome = reconcileDraft(
            newValue: .int(2048),      // someone else set max_tokens to 2048
            currentDraftValue: .int(9000),  // our stale pending draft
            pending: .int(4096),       // what we were actually waiting on
            isEditing: false
        )
        XCTAssertEqual(outcome, .reseed(.int(2048)),
                       "an external change to a pending key reseeds from the external value")
    }

    /// Idempotent no-op: an idle row whose draft already equals the new live
    /// value needs no change at all (avoids a needless redraw / cursor jump).
    func testIdleRowAlreadyMatchingNeedsNoChange() {
        let outcome = reconcileDraft(
            newValue: .string("af_heart"),
            currentDraftValue: .string("af_heart"),
            pending: nil,
            isEditing: false
        )
        XCTAssertEqual(outcome, .keep,
                       "no reseed when the idle row already shows the live value")
    }
}
