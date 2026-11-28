# Test data

The conformance tests of the JSON reader read these files, so they need no network. Each file in
`JSONTestSuite/` and `kuhn/` is an unchanged copy of its upstream file, under the license of its
upstream.

`fuzz/` holds the minimized corpus of the JSON fuzz target (`fuzz/README.md`). The kernel tests read
each of its inputs. libFuzzer made the inputs from seeds of this repository and of JSONTestSuite, so
the MIT license of `JSONTestSuite/LICENSE` applies to the inputs that come from JSONTestSuite.

`fuzz-seeds/` holds the permanent valid and malformed arrays for the reader's number run.
Each is exactly 4,096 bytes. The campaign recipe copies both files into its working corpus.

The number tests also read Nigel Tao's data, which `fetched/` holds and Git ignores. It is too large
for the repository: 14 files, 275 MB, 5,299,993 lines. Fetch it before the tests:

```sh
cd tests/data
mkdir -p fetched/numbers
for name in exhaustive-float16 freetype-2-7 google-double-conversion google-wuffs ibm-fpgen \
  lemire-fast-double-parser lemire-fast-float more-test-cases remyoudompheng-fptest-0 \
  remyoudompheng-fptest-1 remyoudompheng-fptest-2 remyoudompheng-fptest-3 tencent-rapidjson \
  ulfjack-ryu; do
  curl -fsSL --compressed -o "fetched/numbers/$name.txt" \
    "https://raw.githubusercontent.com/nigeltao/parse-number-fxx-test-data/55d79b184b7d8fac2e143e89dc19b766ec4e54b8/data/$name.txt"
done
sha256sum -c numbers.sha256    # On macOS: shasum -a 256 -c numbers.sha256
```

Each line holds the bits of the f16, f32, and f64 values of a string, and then the string. The
upstream is <https://github.com/nigeltao/parse-number-fxx-test-data> at commit
`55d79b184b7d8fac2e143e89dc19b766ec4e54b8` (2022-04-13), under Apache-2.0.

| Directory | Upstream | Version | License |
| --- | --- | --- | --- |
| `JSONTestSuite/` | `test_parsing/` and `LICENSE` of <https://github.com/nst/JSONTestSuite> | commit `1ef36fa01286573e846ac449e8683f8833c5b26a` (2024-11-22) | MIT, in `JSONTestSuite/LICENSE` |
| `kuhn/` | `UTF-8-test.txt` of Markus Kuhn, <https://www.cl.cam.ac.uk/~mgk25/ucs/examples/UTF-8-test.txt> | the file of 2015-08-28, retrieved on 2026-11-19 | CC BY 4.0, in `kuhn/LICENSE` (<https://creativecommons.org/licenses/by/4.0/legalcode.txt>) |

The SHA-256 of each download, on 2026-11-19:

| Download | SHA-256 |
| --- | --- |
| `https://codeload.github.com/nst/JSONTestSuite/tar.gz/1ef36fa01286573e846ac449e8683f8833c5b26a` | `32ab2fb37cc26679dbd9164dbb7053bcef10134643d53cb0cef45de7680dcdcc` |
| `UTF-8-test.txt` | `b51cfe9a8d2689c90b10a13a3624092d546e0837c6ff835b6e5d713c5749c8c6` |
| `legalcode.txt` (`kuhn/LICENSE`) | `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411` |

Kuhn's file has no version control, so its date and its SHA-256 identify it.
