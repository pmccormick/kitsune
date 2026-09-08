# Caravel

Caravel is the LANL kitsune downstream branch of LLVM. It tracks an upstream
LLVM release branch and carries a small stack of local patches rebased directly
on top of it. Changes here have not yet been opened as an upstream RFC or
submitted to `llvm/llvm-project` for review.

The branch is deliberately thin. If the local delta starts growing into
something that is hard to rebase, that is a signal to reconsider the patch, not
to switch to merges.

- **Current base:** `llvmorg-23.1.1-1-ge280c5056ed1` (upstream `release/23.x`)
- **Version:** `23.1.2-caravel`

Facts in this file that name a specific release are marked with a date; the
commands are the durable part.

## Identifying a Caravel build

The version suffix is set in `cmake/Modules/LLVMVersion.cmake`, so every tool
reports it:

```console
$ clang --version | head -1
clang version 23.1.2-caravel (...)

$ llvm-config --version
23.1.2-caravel
```

`llvm-config` also carries a dedicated predicate for build scripts that need to
branch on this being a Caravel tree. It prints a bare `YES` and nothing else:

```console
$ llvm-config --caravel
YES
```

## Remotes

| Remote | Points at |
| --- | --- |
| `origin` | `git@lanl.github.com:pmccormick/kitsune.git` |
| `upstream` | `https://github.com/llvm/llvm-project.git` |

## The local patch stack

The complete local delta is always this, and nothing else:

```console
git log --oneline upstream/release/<N>.x..caravel
```

As of 2026-09-08 that is seven commits: Flang prescanner comment retention plus
its doc-comment binding fix and tests, the `-caravel` version suffix and
`llvm-config --caravel` option, and CUDA 13.3 recognition mapping to PTX ISA 9.3.

Because the history is linear and rebased rather than merged, that one command
is a complete and trustworthy answer to "what have we changed?". Keep it that
way.

## Syncing with upstream

### When to move to a new major release

Wait for the `.1.0` tag to ship **without** an `-rc` suffix. Rebasing onto a
branch still taking release-candidate churn means re-resolving the same
conflicts on every rc drop for no benefit. Point releases within a major version
are low-risk by comparison.

```console
git ls-remote --tags upstream 'llvmorg-24*' | grep -v '\^{}'
```

### Always rebase with `--onto`

This is the single most important thing in this file. A plain
`git rebase upstream/release/23.x` is **wrong** for a major-version move.
Because the old and new release branches diverged from `main` long ago, the
merge base is that old branch point, so git tries to replay every
release-branch-only commit — version bumps, backports, the lot — roughly 94 of
them for 22.x to 23.x, blowing up immediately on `libcxx/include/__config`.

Replay only the local stack:

```console
git fetch upstream release/<NEW>.x --tags
git branch caravel-<OLD>.x caravel                      # backup ref, cheap insurance
git rebase --onto upstream/release/<NEW>.x upstream/release/<OLD>.x caravel
```

### Resolving conflicts

Check what upstream actually did before resolving; do not reflexively keep the
local side. Local patches exist because upstream had not solved a problem yet,
and when upstream does solve it, its version is often better.

The CUDA patch is the worked example. Upstream landed its own CUDA 13.x support
with a *per-version* PTX mapping (13.0 to ptx90, 13.1 to ptx91, 13.2 to ptx92),
while the local patch had mapped all of 13.x to a flat ptx90 — correct when
written, because ptx90 was then the NVPTX backend ceiling. Keeping the local
side would have silently downgraded 13.1 and 13.2. The patch was reduced to what
was genuinely still missing: CUDA 13.3 mapping to ptx93.

Rewrite any commit message that no longer describes what its patch does.

### Before pushing

Build, run the tests below, then push with `--force-with-lease`. Never a plain
`--force`: the lease is what catches someone else having moved the branch.

## Building

This configuration is known good on the current base:

```console
cmake -G Ninja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;flang;mlir;lld;lldb;bolt" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp;flang-rt" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DFLANG_RUNTIME_F128_MATH_LIB=libquadmath
```

### A GCC upgrade will break the flang-rt build

Symptom, from the `runtimes` sub-build:

```
fatal error: '/usr/lib64/gcc/x86_64-pc-linux-gnu/<old-version>/include/quadmath.h' file not found
```

`flang/cmake/modules/FlangCommon.cmake` parses the GCC installation out of
`clang -v -###` and stores it as `CACHE INTERNAL ... FORCE`, guarded by
`if (NOT DEFINED FLANG_GCC_RESOURCE_DIR)`. The absolute path is then baked into
the generated `flang-rt/quadmath_wrapper.h`. When the system GCC version
directory changes, the cached path dangles — and the `NOT DEFINED` guard means
no ordinary reconfigure will ever re-detect it.

Clear the two entries and let detection re-run:

```console
cmake -U FLANG_GCC_RESOURCE_DIR -U FOUND_GCC_QUADMATH_H <build>/runtimes/runtimes-bins
```

Prefer this over pinning the path with `-D`, which re-arms the same trap on the
next upgrade. Only the runtimes sub-build is affected: it uses the freshly built
clang, whereas the top-level build uses the system compiler, which already has
the GCC include directory on its default search path.

## Testing the local patches

Every local patch has coverage. Run these before pushing a rebase:

```console
ninja check-flang                                    # includes both unit test targets below
ninja FlangParserTests FlangDocCommentTests
./bin/llvm-lit ../clang/test/Driver/cuda-ptx-version.cu
./bin/llvm-lit ../llvm/test/tools/llvm-config/booleans.test
```

| Patch | Covered by |
| --- | --- |
| Flang comment retention | `flang/unittests/Parser/CommentRetentionTest.cpp` |
| Doc-comment to symbol binding | `flang/unittests/Semantics/DocCommentTest.cpp` |
| CUDA version to PTX ISA mapping | `clang/test/Driver/cuda-ptx-version.cu` |
| `llvm-config --caravel` | `llvm/test/tools/llvm-config/booleans.test` |

The CUDA test is worth understanding before changing it. A missing rung in the
version ladder does not fail loudly: an unrecognized toolkit falls through to
`CudaVersion::NEW`, which `CudaInstallationDetector::version()` clamps to
`PARTIALLY_SUPPORTED`, so the compile quietly proceeds against an older ISA than
the toolkit provides. The test therefore also asserts that a genuinely unknown
future version *does* still warn, so the first assertion cannot be satisfied by
widening the ladder without bound.

A quick end-to-end check that quadmath and CUDA both work in an installed build:

```console
$ cat > q.f90 <<'EOF'
program q
  real(kind=16) :: x
  x = 2.0_16
  print *, sqrt(x)
end program q
EOF
$ flang -o q q.f90 && ./q
 1.414213562373095048801688724209698

$ clang -x cuda --cuda-path=/opt/cuda --offload-arch=sm_120 -### -c empty.cu 2>&1 | grep -o '+ptx[0-9]*'
+ptx93
```

Thirty-three significant digits confirms real binary128 rather than an 80-bit
`long double` fallback.
