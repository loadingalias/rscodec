# Contributing

Run `rustup toolchain install` in the checkout. Run `cargo install --locked cargo-nextest@0.9.150`.
Fetch the [number data](tests/data/README.md) and [JSON corpora](benches/data/README.md). WebAssembly tests also need Wasmtime.

Run `cargo xtask fix`, `cargo xtask check`, and `cargo xtask test`.
Sign off each commit with `git commit -s`. [Developer Certificate of Origin](https://developercertificate.org/)

## Security

Report vulnerabilities privately. Include the version, features, CPU, target, and a synthetic input.
Security fixes go into the latest release. [Report a vulnerability](https://github.com/loadingalias/rscodec/security/advisories/new)
