# Run the Benchmarks

Run these commands from the repository root. The last argument selects the operation, library, and input.

```sh
cargo bench --locked --bench base_n -- 'base64-decode/.*/url-32$'
cargo bench --locked --bench json -- 'json-typed/.*/a2-jose-header$'
```

Base64 and hex benchmarks require a host that can run `with_dit`. Fetch the three large corpora before running the JSON benchmarks. [Benchmark inputs and checksums](data/README.md)

| Group | One timed operation |
| --- | --- |
| `base64-encode`, `base64-decode`, `hex-encode`, `hex-decode` | Encode or decode one input. Sizes count decoded bytes. Base64 uses unpadded URL text through 4,096 bytes and padded standard text at 65,536 bytes. |
| `json-read` | Read all documents in one input class. rscodec visits tokens, decodes strings, and converts numbers. Tree-based readers construct their trees. |
| `json-typed` | Decode the fields of each document into the same application type. |
| `json-write` | Write each prepared document as compact JSON. |
| `f64-print`, `f64-read` | Print or parse one set of float values. |

Base64 and hex rows share 64-byte-aligned buffers. Base64 and hex output must match the reference bytes.
`rscodec-secret` measures calls within one DIT scope. `rscodec-secret-cold` opens a scope per call.
JSON rows check values against serde_json. The harness records allocations and peak heap growth outside the timed loop.

The Linux comparison pinned one process to a core and varied the stack address eight times per build.
It used 0.1 seconds of warm-up and 0.3 seconds of measurement per row. Published results include every loss.
The CSV gives each library's mean, minimum, and maximum across the eight medians, in nanoseconds per operation.
The rival had the lowest mean among eligible libraries. [All 308 comparisons](results/e03f296.csv) · [Results and hosts](../README.md#benchmarks)

## Measured Losses

| Owner | What the measurements show |
| --- | --- |
| JSON reader | On Canada, a faster digit scan cut 8.7 cycles per float in isolation and about 3.5 in the full read on M1 Pro. No measured cause accounts for the entire gap to sonic-rs. On Xeon, emitting each scalar directly reduced the sparse-index gap. |
| Hex kernels | On Xeon, the 32-byte decode kernel beat faster-hex in an earlier run. The public call lost. AVX-512 warm-up and power state remain unconfirmed explanations. On Graviton4, 256-byte encode used the same loop instructions in both builds. The cycle difference remains unexplained. |
| Base64 kernels | The Xeon bulk encoder uses full vector loads until the tail. The Arm 32-byte URL decode lost across builds. The 4 KiB URL decode changed rank between earlier runs. These observations do not establish the remaining causes. |
| Float writer | Removing the second rounding path reduced random float print time in all four builds. On native Xeon, printing random values took 1.054 times as long as zmij. The remaining cause is unknown. |
