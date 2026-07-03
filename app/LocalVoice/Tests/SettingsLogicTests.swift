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
}
