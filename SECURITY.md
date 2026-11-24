# Security policy

Report a vulnerability in private through GitHub private vulnerability reporting. Select "Report a
vulnerability" on the [Security tab](https://github.com/loadingalias/rscodec/security) of the
repository. Do not open a public issue.

Include these items:

- The release or the commit, the features, the target, and the CPU.
- An input that shows the defect, and the result that you expect.

Send no real key, token, or personal data. Use made-up values.

## Supported releases

Security fixes go into the latest release.

## Scope

Report each of these defects in a codec of the support scope, which the
[README](README.md#support) gives:

- An input that causes a panic, undefined behavior, memory use without a limit, or a wrong result.
- A SIMD kernel that gives a result or an error that the portable kernel does not give.
- A secret function whose time or memory access depends on its secret data, in a build that the
  [timing evidence](docs/content/base64-and-hex.md#timing-evidence) covers.
