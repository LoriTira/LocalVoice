import Foundation

/// How `EngineClient` should spawn the engine subprocess.
enum LaunchMode: Equatable {
    /// Runs `uv run localvoice serve` with the given repo root as the
    /// working directory — the phase-A dev-checkout entry point.
    case devCheckout(URL)

    /// Runs an arbitrary executable with arguments — used by tests (the
    /// bash fixture) and, later, phase C's bundled engine binary.
    case custom(executable: URL, arguments: [String])
}

extension LaunchMode {
    /// The executable and arguments this mode maps to, plus the working
    /// directory `Process` should run it from.
    var launchSpec: (executable: URL, arguments: [String], workingDirectory: URL) {
        switch self {
        case .devCheckout(let repoRoot):
            return (
                executable: URL(fileURLWithPath: "/usr/bin/env"),
                arguments: ["uv", "run", "localvoice", "serve"],
                workingDirectory: repoRoot
            )
        case .custom(let executable, let arguments):
            return (
                executable: executable,
                arguments: arguments,
                workingDirectory: executable.deletingLastPathComponent()
            )
        }
    }
}
