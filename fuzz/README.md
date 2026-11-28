# Fuzz the JSON reader

The `read` target runs both JSON profiles. It compares token visits, typed reads, alternating skips,
whole-value skips, and reads that leave validation to the reader.
It checks retained keys and values in order, number conversions, and the first error kind and byte offset.

Two independent parsers check the RFC tree. serde_json sorts keys and drops earlier duplicates.
The simdjson shim preserves every member in order. The harness checks each exception against the input.

| Difference | Required evidence |
| --- | --- |
| Duplicate key | The error points to a repeated decoded key in the same object. |
| Depth | Only simdjson can accept the empty 128th container when the harness limit is 127. |
| Numeric overflow | Replace each verified overflow with zero, pad with spaces to preserve offsets, and compare the whole document again. |
| UTF-8 byte order mark | Compare simdjson with the rscodec result after removing the mark. |

## Build and replay

Fetch simdjson 5.2.0 and verify both files. Only the fuzz workspace builds its C++ shim.

```sh
cd fuzz
mkdir -p simdjson
for name in simdjson.h simdjson.cpp; do
  curl -fsSL -o "simdjson/$name" "https://github.com/simdjson/simdjson/releases/download/v5.2.0/$name"
done
sha256sum -c simdjson.sha256    # On macOS: shasum -a 256 -c simdjson.sha256
```

From the repository root, run the comparison tests and replay each permanent seed once.

```sh
cargo test --locked --manifest-path fuzz/Cargo.toml --bin read
cargo build --locked --release --manifest-path fuzz/Cargo.toml --bin read
./fuzz/target/release/read tests/data/fuzz-seeds/*.json
```

Both seeds are 4,096 bytes and enter the dense number loop. The valid seed emits 988 numbers.
The malformed seed emits 987 numbers and returns `Syntax` at byte 4,093.

## Run a campaign

From `fuzz/`, build for the host you intend to test. This example selects Apple Silicon.
The target uses the CPU's selected reader. The kernel differential tests also replay the saved corpus.

```sh
target=aarch64-apple-darwin
cargo fuzz build read --target "$target" --debug-assertions
mkdir -p artifacts corpus/read
cp ../tests/data/fuzz-seeds/*.json ../tests/data/fuzz/* ../tests/data/JSONTestSuite/test_parsing/*.json corpus/read/
for file in ../benches/data/json/*/*.jsonl; do split -l 1 "$file" "corpus/read/$(basename "$file")-"; done
cp ../benches/data/json/a*/*.json corpus/read/
./target/"$target"/release/read -fork=4 -max_total_time=600 -max_len=4096 \
  -timeout=10 -dict=json.dict -artifact_prefix=artifacts/ corpus/read
```

Use at most ten minutes on a workstation. A failure saves its input under `artifacts/`.
In fork mode, send SIGTERM to the parent to stop early.

To minimize a corpus, build the target on each architecture you tested. Run each binary once against the same initially empty output directory.

```sh
mkdir minimized
./target/"$target"/release/read -merge=1 -use_counters=0 -max_len=4096 minimized corpus/read
```

Review the result before replacing `tests/data/fuzz/`. Keep both permanent inputs in `tests/data/fuzz-seeds/`.
