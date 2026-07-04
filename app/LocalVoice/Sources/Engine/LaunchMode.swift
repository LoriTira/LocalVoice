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
            let (executable, uvArguments) = Self.resolveUv()
            return (
                executable: executable,
                arguments: uvArguments + ["run", "localvoice", "serve"],
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

    /// Locates the `uv` binary without depending on `PATH` carrying it.
    ///
    /// A GUI app launched by Finder/`open` (as opposed to a shell) inherits
    /// `launchd`'s minimal `PATH` (roughly `/usr/bin:/bin:/usr/sbin:/sbin`),
    /// which excludes `~/.local/bin` — the default `uv` install location —
    /// and typically excludes Homebrew's `/opt/homebrew/bin` too. Routing
    /// through `/usr/bin/env uv` alone (as `EngineClient`'s Task 3
    /// implementation did) silently fails in that launch context even
    /// though it works fine from a terminal, since a terminal shell's
    /// login/interactive PATH is richer. Checking known absolute install
    /// locations first sidesteps the whole PATH question.
    ///
    /// Returns the executable to run plus any arguments that must precede
    /// `["run", "localvoice", "serve"]` — empty when a concrete `uv` path
    /// was found (the executable *is* `uv`), or `["uv"]` for the
    /// `/usr/bin/env` fallback (the executable is `env`, which needs the
    /// target name as its first argument).
    private static func resolveUv() -> (executable: URL, leadingArguments: [String]) {
        let candidates = [
            "\(NSHomeDirectory())/.local/bin/uv", // uv's own installer default
            "/opt/homebrew/bin/uv", // Homebrew, Apple Silicon
            "/usr/local/bin/uv", // Homebrew, Intel; other manual installs
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return (URL(fileURLWithPath: path), [])
        }
        // Fall back to a PATH search via /usr/bin/env — covers an install
        // location this list didn't anticipate, for a process that does
        // happen to inherit a PATH containing it (e.g. launched from a
        // terminal, or a machine with a differently-configured install).
        return (URL(fileURLWithPath: "/usr/bin/env"), ["uv"])
    }
}
