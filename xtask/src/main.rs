//! Repository commands. Run them as `cargo xtask <command>`.
//!
//! Each command runs Cargo from the toolchain that `rust-toolchain.toml` pins,
//! in the workspace root, without ambient compiler overrides. Local work and CI
//! use the same commands. xtask needs only `std`.

#![expect(
    clippy::print_stderr,
    reason = "xtask is a command-line tool, and standard error is its report channel"
)]

use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Instant,
};

const USAGE: &str = "\
Usage: cargo xtask <command> [arguments]

Commands
  fix [Cargo args]     Apply Clippy fixes, format, then fail on the lints that remain (local only)
  check [Cargo args]   Check formatting, lints, and rustdoc, and change no file (the CI gate)
  test [Nextest args]  Run the tests; with no arguments, also run the tests of
                       rscodec without `std`, then the doctests

fix, check, and test select the whole workspace, unless their arguments select
packages with -p or --package. With the whole workspace, fix and check also
lint rscodec with no default feature: its library with each codec on
wasm32v1-none, a target with no `std`, without and with simd128, and all of its
targets, the tests too, on the host. They also lint its library with each codec
alone, on wasm32v1-none without and with simd128, and with `std` on the host.
rustdoc documents each codec.
";

const NO_STD_TARGET: &str = "wasm32v1-none";

const SIMD128: &str = r#"target.wasm32v1-none.rustflags=["-C", "target-feature=+simd128"]"#;

const NO_STD_PACKAGE: &str = "rscodec";

const LEAVES: &str = "base64,hex,json";

const WORKSPACE: &[&str] = &["--workspace"];

const SCRUBBED: &[&str] = &[
    "CARGO",
    "CARGO_ENCODED_RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_INCREMENTAL",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTDOC",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
];

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        eprint!("{USAGE}");
        return ExitCode::SUCCESS;
    };
    let result = match command.as_str() {
        "-h" | "--help" => {
            eprint!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        "fix" => Repo::open().and_then(|repo| repo.fix(rest)),
        "check" => Repo::open().and_then(|repo| repo.check(rest)),
        "test" => Repo::open().and_then(|repo| repo.test(rest)),
        other => Err(Failure::new(format!("unknown command `{other}`\n\n{USAGE}"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("error: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}

struct Repo {
    root: PathBuf,
    channel: String,
}

impl Repo {
    fn open() -> Result<Self, Failure> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.parent().unwrap_or(manifest).to_path_buf();
        let path = root.join("rust-toolchain.toml");
        let text = fs::read_to_string(&path).map_err(|error| Failure::new(format!("{}: {error}", path.display())))?;
        let channel = text
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once('=')?;
                (key.trim() == "channel").then(|| value.trim().trim_matches('"').to_owned())
            })
            .filter(|channel| !channel.is_empty())
            .ok_or_else(|| Failure::new(format!("{} pins no channel", path.display())))?;
        Ok(Self { root, channel })
    }

    fn cargo(&self, args: &[&str]) -> Command {
        let mut command = Command::new("rustup");
        command
            .args(["run", &self.channel, "cargo"])
            .args(args)
            .current_dir(&self.root);
        for (name, _) in env::vars_os() {
            if scrubbed(&name) {
                command.env_remove(name);
            }
        }
        command
            .env("RUSTUP_AUTO_INSTALL", "0")
            .env("RUSTC_WRAPPER", "")
            .env("RUSTC_WORKSPACE_WRAPPER", "");
        command
    }

    // Clippy reports unfixed lints as warnings during --fix. Run lint again to enforce the workspace levels.
    fn fix(&self, args: &[String]) -> Result<(), Failure> {
        let scope = scope(args, WORKSPACE);
        step(
            "clippy --fix",
            self.cargo(&[
                "clippy",
                "--fix",
                "--locked",
                "--allow-dirty",
                "--all-targets",
                "--quiet",
            ])
            .args(&scope),
        )?;
        step("fmt", &mut self.cargo(&["fmt", "--all"]))?;
        step(
            "clippy",
            self.cargo(&["clippy", "--locked", "--all-targets", "--quiet"])
                .args(&scope),
        )?;
        self.clippy_no_std(&scope)
    }

    fn clippy_no_std(&self, scope: &[&str]) -> Result<(), Failure> {
        if !scope.contains(&"--workspace") {
            eprintln!("── clippy without std\n   not run: the arguments select packages; run with no -p or --package");
            return Ok(());
        }
        step(
            "clippy without std",
            &mut self.cargo(&[
                "clippy",
                "--locked",
                "--quiet",
                "--lib",
                "--no-default-features",
                "--features",
                LEAVES,
                "--target",
                NO_STD_TARGET,
                "--package",
                NO_STD_PACKAGE,
            ]),
        )?;
        step(
            "clippy without std, simd128",
            &mut self.cargo(&[
                "clippy",
                "--locked",
                "--quiet",
                "--lib",
                "--no-default-features",
                "--features",
                LEAVES,
                "--target",
                NO_STD_TARGET,
                "--package",
                NO_STD_PACKAGE,
                "--config",
                SIMD128,
            ]),
        )?;
        step(
            "clippy tests without std",
            &mut self.cargo(&[
                "clippy",
                "--locked",
                "--quiet",
                "--all-targets",
                "--no-default-features",
                "--package",
                NO_STD_PACKAGE,
            ]),
        )?;
        for leaf in LEAVES.split(',') {
            let with_std = format!("std,{leaf}");
            let wasm = ["--target", NO_STD_TARGET];
            let simd128 = ["--target", NO_STD_TARGET, "--config", SIMD128];
            for (features, label, target) in [
                (leaf, "", &wasm[..]),
                (leaf, ", simd128", &simd128[..]),
                (&with_std, "", &[][..]),
            ] {
                step(
                    &format!("clippy {features} alone{label}"),
                    self.cargo(&[
                        "clippy",
                        "--locked",
                        "--quiet",
                        "--lib",
                        "--no-default-features",
                        "--features",
                        features,
                        "--package",
                        NO_STD_PACKAGE,
                    ])
                    .args(target),
                )?;
            }
        }
        Ok(())
    }

    fn check(&self, args: &[String]) -> Result<(), Failure> {
        let scope = scope(args, WORKSPACE);
        step("fmt --check", &mut self.cargo(&["fmt", "--all", "--check"]))?;
        step(
            "clippy",
            self.cargo(&["clippy", "--locked", "--all-targets", "--quiet"])
                .args(&scope),
        )?;
        self.clippy_no_std(&scope)?;
        step(
            "rustdoc",
            self.cargo(&["doc", "--locked", "--no-deps", "--all-features", "--quiet"])
                .args(&scope),
        )
    }

    fn test(&self, args: &[String]) -> Result<(), Failure> {
        step(
            "nextest",
            self.cargo(&["nextest", "run", "--locked"]).args(scope(args, WORKSPACE)),
        )?;
        if !args.is_empty() {
            return Ok(());
        }
        step(
            "nextest without std",
            &mut self.cargo(&[
                "nextest",
                "run",
                "--locked",
                "--no-default-features",
                "--package",
                NO_STD_PACKAGE,
            ]),
        )?;
        step(
            "doctest",
            &mut self.cargo(&["test", "--locked", "--doc", "--workspace", "--quiet"]),
        )
    }
}

fn scrubbed(name: &OsStr) -> bool {
    let name = name.to_string_lossy().to_ascii_uppercase();
    SCRUBBED.contains(&name.as_str())
        || ["CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

fn scope<'a>(args: &'a [String], default: &[&'a str]) -> Vec<&'a str> {
    let selects = args
        .iter()
        .any(|arg| arg == "--workspace" || arg.starts_with("--package") || arg.starts_with("-p"));
    let default = if selects { &[][..] } else { default };
    default.iter().copied().chain(args.iter().map(String::as_str)).collect()
}

fn step(name: &str, command: &mut Command) -> Result<(), Failure> {
    eprintln!("── {name}");
    let started = Instant::now();
    let status = command.status().map_err(|error| {
        Failure::new(format!(
            "{name}: cannot start {}: {error}",
            command.get_program().to_string_lossy()
        ))
    })?;
    if status.success() {
        eprintln!("   ok in {:.2}s", started.elapsed().as_secs_f64());
        return Ok(());
    }
    let code = status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .filter(|code| *code != 0)
        .unwrap_or(1);
    Err(Failure {
        message: format!("{name} failed ({status})"),
        code,
    })
}

struct Failure {
    message: String,
    code: u8,
}

impl Failure {
    fn new(message: String) -> Self {
        Self { message, code: 1 }
    }
}

#[cfg(test)]
mod tests {
    //! The package selection of the Cargo commands.

    use super::{WORKSPACE, scope};

    #[test]
    fn only_package_flags_narrow_the_scope() {
        let joined =
            |args: &[&str]| scope(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(), WORKSPACE).join(" ");
        for args in [
            &[][..],
            &["--all-features"],
            &["--target", "wasm32-wasip1"],
            &["-E", "test(only_package_flags_narrow_the_scope)"],
            &["--exclude", "xtask"],
        ] {
            assert_eq!(joined(args), [&["--workspace"][..], args].concat().join(" "));
        }
        for args in [
            &["-p", "rscodec"][..],
            &["-prscodec"],
            &["--package", "rscodec"],
            &["--package=rscodec"],
            &["--workspace"],
        ] {
            assert_eq!(joined(args), args.join(" "));
        }
    }
}
