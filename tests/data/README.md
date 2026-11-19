# Test data

The conformance tests of the JSON reader read these files, so they need no network. Each file is
an unchanged copy of its upstream file, under the license of its upstream.

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
