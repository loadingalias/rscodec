# Fuzz targets

This workspace holds the fuzz targets of rscodec. libFuzzer runs them with AddressSanitizer. The
workspace is separate from the workspace of the crate, because libFuzzer needs flags that the crate
does not take. It uses the toolchain of the repository and `cargo-fuzz`.

## `read`: the JSON reader

For each input, in `Rfc8259` and in `IJson`, the target checks these contracts:

1. Each read path of rscodec gives the same result, error, and offset: the tokens of `visit`, the
   value walk, a walk that skips each odd item, `Value::skip`, and `read` with a closure that reads
   nothing. The tokens and the value walk give the same tree.
2. `IJson` refuses each input that `Rfc8259` refuses. It also refuses each noncharacter and each
   number that binary64 does not hold, at its offset.
3. The accessors of each number give the values of the parsers of `core`.
4. serde_json and simdjson give the tree of `Rfc8259`. They differ only by the rules below.

The fuzz runs the kernel that the CPU selects. On an M1 Mac, an `aarch64-apple-darwin` build runs
the Neon kernel. An `x86_64-apple-darwin` build runs under Rosetta, which has no AVX2, so it runs
the SWAR reference. The minimized corpus is in `tests/data/fuzz/`. The kernel tests of the crate
compare each kernel with the reference on each of its inputs.

### The rules where the trusted readers differ

| Input | rscodec | serde_json | simdjson |
| --- | --- | --- | --- |
| A duplicate key | refuses, in each profile | keeps the last value | keeps each member |
| A number whose magnitude rounds to infinity | `Rfc8259` accepts it, because RFC 8259 section 6 sets no range. `IJson` refuses it | refuses | refuses |
| The 128th container (`depth_max` 127) | refuses | refuses | can accept it, because it does not count an empty container |
| A byte order mark | refuses | refuses | skips it. The target compares simdjson with rscodec on the input after the mark |

serde_json keeps each object in a `BTreeMap`, so the target compares its objects without the order
of the members. simdjson keeps the order. Each trusted reader selects its own type for a number. The
target compares the number in that type with the parser of `core` on the text that rscodec gives.

## Fetch simdjson

`build.rs` builds simdjson with its shim `oracle.cpp`. Fetch the amalgamation of simdjson 5.2.0
(2026-11-15, Apache-2.0 or MIT) into `simdjson/`, which Git ignores:

```sh
cd fuzz
mkdir -p simdjson
for name in simdjson.h simdjson.cpp; do
  curl -fsSL -o "simdjson/$name" "https://github.com/simdjson/simdjson/releases/download/v5.2.0/$name"
done
sha256sum -c simdjson.sha256    # On macOS: shasum -a 256 -c simdjson.sha256
```

## Run a campaign

Run each command in `fuzz/`.

1. Give the build target. `cargo-fuzz` takes its own build target as the default, and that can be
   another architecture:

   ```sh
   cargo fuzz build read --target aarch64-apple-darwin --debug-assertions
   ```

2. Put the seeds into `corpus/read/`, which Git ignores:

   ```sh
   mkdir -p corpus/read
   cp ../tests/data/fuzz/* ../tests/data/JSONTestSuite/test_parsing/*.json corpus/read/
   for file in ../benches/data/json/*/*.jsonl; do split -l 1 "$file" "corpus/read/$(basename "$file")-"; done
   cp ../benches/data/json/a*/*.json corpus/read/
   ```

3. Run the target with a time limit. Each crash stops the run and writes its input to `artifacts/`.
   On a workstation, run for at most 10 minutes. Run a longer campaign on an ephemeral host:

   ```sh
   ./target/aarch64-apple-darwin/release/read -fork=4 -max_total_time=600 -max_len=4096 \
     -timeout=10 -dict=json.dict -artifact_prefix=artifacts/ corpus/read
   ```

   To stop a run early, send SIGTERM to the parent process. In fork mode, SIGINT does not stop it.

4. After the campaign, minimize the corpus into `tests/data/fuzz/`:

   ```sh
   mkdir -p minimized
   ./target/aarch64-apple-darwin/release/read -merge=1 -use_counters=0 -max_len=4096 minimized corpus/read
   ./target/x86_64-apple-darwin/release/read -merge=1 -use_counters=0 -max_len=4096 minimized corpus/read
   rm ../tests/data/fuzz/* && mv minimized/* ../tests/data/fuzz/ && rmdir minimized
   ```
