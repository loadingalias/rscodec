# Contributing

## Setup

1. Install [rustup](https://rustup.rs).
2. In the repository, run `rustup toolchain install`. `rust-toolchain.toml` pins the nightly
   toolchain, its components, and its targets.
3. Install cargo-nextest at the version that `.config/nextest.toml` pins:
   `cargo install --locked cargo-nextest@0.9.150`.
4. Fetch the test data, which the repository does not hold. `benches/data/README.md` gives the
   commands for three JSON corpora, and `tests/data/README.md` gives them for the number data (275 MB).
5. For the WebAssembly tests, install [Wasmtime](https://wasmtime.dev). CI uses the version that
   `.github/workflows/ci.yml` names.

## Commands

| Command | Use |
| --- | --- |
| `cargo xtask fix` | Applies the Clippy fixes and formats the code. Then it fails on each lint that remains |
| `cargo xtask check` | Checks the format, the lints, and rustdoc, and changes no file. This is the CI gate |
| `cargo xtask test` | Runs the tests, then the doctests |
| `cargo xtask test -E 'test(<name>)'` | Runs the tests that the Nextest filter selects |
| `cargo xtask test --target wasm32-wasip1` | Runs the tests in Wasmtime |

Each command uses `--locked`, so no command changes `Cargo.lock`.

## Rules of a change

- Each SIMD kernel gives the same output, error, and error offset as the portable kernel, for each
  input. A test compares them.
- A codec allocates no memory, and the library has no dependency.
- A codec that can carry a secret has no branch on the data and no load at an index from the data.
  After a change of a base64 or hex kernel, run `cargo run --release -p rscodec-ct -- scan`.
- `Cargo.toml` and `clippy.toml` hold the lint policy. Each `unsafe` item has an
  `#[expect(unsafe_code, reason = "...")]` that states why it is sound.

## Sign-off and license

Sign off each commit with `git commit -s`. The `Signed-off-by` line states that you agree with the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). Each contribution has the
license of the repository, [Apache-2.0](LICENSE).
