import SwiftUI
import os

@main
struct LocalVoiceApp: App {
    /// `UserDefaults` key backing the dev-checkout repo root, editable later
    /// from the Setup pane (Task 7) — a plain `String` constant here so both
    /// this file and that future view read/write the exact same key name.
    static let devCheckoutPathDefaultsKey = "devCheckoutPath"

    private let client: EngineClient
    @State private var appState = AppState()

    private static let logger = Logger(subsystem: "dev.localvoice", category: "app")

    init() {
        let mode = LaunchMode.devCheckout(Self.resolveDevCheckoutRoot())
        let client = EngineClient(mode: mode)
        self.client = client

        // `EngineClient.events` finishes only after `stop()` or a final
        // (no-respawn) death — see Task 3/4's notes — so this loop is
        // expected to run for the lifetime of the process, not a view. It's
        // an unstructured `Task` (not a view `.task`) precisely because
        // closing the last window on macOS does not tear down `App` state
        // or terminate the process (per the design doc: "window closing
        // hides to background; assistant keeps working") — a view-attached
        // `.task` would be cancelled the moment its hosting view left the
        // hierarchy, which a window close can trigger even though the app
        // itself stays alive. Tying the loop to `App.init()` instead keeps
        // it running across any number of window close/reopen cycles.
        Task { @MainActor [appState] in
            await client.start()
            for await event in client.events {
                appState.reduce(event)
            }
        }
    }

    var body: some Scene {
        WindowGroup("LocalVoice") {
            MainWindow(appState: appState, client: client)
                .frame(minWidth: 720, minHeight: 480)
        }
    }

    /// Resolves the dev-checkout repo root `EngineClient` spawns
    /// `uv run localvoice serve` from: a `UserDefaults` override if one has
    /// been set (via the Setup pane, later), else a source-tree-relative
    /// default computed from this file's own `#filePath`.
    ///
    /// The default deliberately does NOT use `Bundle.main.bundleURL` —
    /// under `xcodebuild`/Xcode, the built `.app` lands in DerivedData
    /// (`~/Library/Developer/Xcode/DerivedData/.../Build/Products/Debug/
    /// LocalVoice.app`), a path with no fixed relationship to the repo
    /// checkout, so no fixed number of `deletingLastPathComponent()` calls
    /// from *that* URL could ever reach it. `#filePath` instead captures
    /// this source file's location at compile time
    /// (`<repo>/app/LocalVoice/Sources/LocalVoiceApp.swift`), which is
    /// stable relative to the checkout on both a dev machine and CI
    /// (`actions/checkout` preserves the same relative layout) — the same
    /// reasoning `EngineClientTests` already applied to its fixture path
    /// (see that file's doc comment). Four `deletingLastPathComponent()`
    /// calls walk the file's own directory (`Sources/`) up to the repo
    /// root: `Sources/LocalVoiceApp.swift` → `Sources/` → `LocalVoice/` →
    /// `app/` → `<repo root>` (verified empirically against a live run —
    /// an earlier 3-call draft under-walked by one level and landed on
    /// `app/`, silently falling through to the `#if DEBUG` literal
    /// fallback below on every single launch; see the task report).
    private static func resolveDevCheckoutRoot() -> URL {
        if let override = UserDefaults.standard.string(forKey: devCheckoutPathDefaultsKey),
           !override.isEmpty {
            return URL(fileURLWithPath: override)
        }

        let sourceRelativeRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Sources/LocalVoiceApp.swift -> Sources/
            .deletingLastPathComponent() // Sources/ -> LocalVoice/
            .deletingLastPathComponent() // LocalVoice/ -> app/
            .deletingLastPathComponent() // app/ -> <repo root>
        if FileManager.default.fileExists(atPath: sourceRelativeRoot.appendingPathComponent("pyproject.toml").path) {
            return sourceRelativeRoot
        }

        // Belt-and-suspenders dev fallback: `#filePath`'s resolution is
        // reliable in every configuration this project actually ships in
        // (any `xcodebuild`/Xcode build compiles from the real checkout, so
        // the compile-time path above is always correct there) — this
        // branch should be unreachable outside an unusual toolchain that
        // rewrites `#filePath`. Kept as a guarded last resort rather than a
        // silent `fatalError`/wrong-path spawn, and explicitly `#if DEBUG`
        // so it can never surface in a phase-C release build once the
        // bundled-engine `LaunchMode` supersedes `devCheckout` there.
        #if DEBUG
        logger.error("dev-checkout root not found via #filePath; falling back to the known checkout literal")
        return URL(fileURLWithPath: "/Users/lorenzo/Desktop/MIT/code/LocalVoice")
        #else
        return sourceRelativeRoot
        #endif
    }
}
