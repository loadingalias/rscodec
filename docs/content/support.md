# Platform and Feature Support

rscodec 0.1.0 requires Rust 1.101. The tested toolchain is `nightly-2026-11-02`.
The lib uses no unstable Rust features. Before v1.0, minor releases can change (likely, will change) the API.

`base64`, `hex`, and `json` are independent. The default enables only `std`.
With default features disabled, the library uses only `core`. All codecs use caller-owned storage.

| Target        | Run evidence                                                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux x86_64  | Xeon 6975P-C ran the portable and supported x86 kernels at `e03f296`, including AVX2 and AVX-512. CI tested the review fixes at `51b7280`.            |
| Linux AArch64 | Graviton4 ran base64, hex, and JSON with and without `std` at `e03f296`. Corrected secret paths also ran in a Linux ARM64 VM.                         |
| macOS AArch64 | M1 Pro ran the default and core-only test suites. The JSON reader also had differential fuzz runs on the Neon kernel.                                 |
| Windows x64   | Native tests at `7074899` passed with and without `std`, including the JSON number-run test and doctests.                                             |
| WebAssembly   | Wasmtime ran the WASI tests with SIMD128 enabled and disabled in [CI at `51b7280`](https://github.com/loadingalias/rscodec/actions/runs/46100650431). |

Windows ARM64 last ran at `3ca205d`. Coverage of the current reader on that target is experimental.
The table lists the targets with run evidence.

The JSON reader selects SWAR (word-at-a-time scalar code), AVX2, AVX-512, Neon, Neon w/ polynomial multiply, or SIMD128.
The base64 and hex guide lists the kernels and hardware limits for secret data. [Base64 and hex](base64-and-hex.md#timing-evidence)
