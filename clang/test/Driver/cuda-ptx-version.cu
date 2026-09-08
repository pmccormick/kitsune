// Check that a detected CUDA toolkit version selects the matching PTX ISA.
//
// The driver maps each supported CUDA version to a +ptxNN target feature. When
// a toolkit is newer than every entry in that table it falls through to
// CudaVersion::NEW, which CudaInstallationDetector::version() clamps down to
// PARTIALLY_SUPPORTED -- so a missing table entry does not fail loudly, it
// silently compiles against an older ISA than the toolkit provides. These tests
// pin the mapping so that regression stays visible.

// CUDA 13.3 ships PTX ISA 9.3, which is also the highest version the NVPTX
// backend can emit (see the PTX version list in NVPTX.td).
// RUN: %clang -### --target=x86_64-linux -x cuda --offload-arch=sm_90 -c \
// RUN:   --cuda-path=%S/Inputs/CUDA_133/usr/local/cuda %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CUDA133

// CUDA133: "-target-feature" "+ptx93"

// A toolkit that is actually recognized must not warn about being too new.
// RUN: %clang -### --target=x86_64-linux -x cuda --offload-arch=sm_90 -c \
// RUN:   --cuda-path=%S/Inputs/CUDA_133/usr/local/cuda %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NO_VERSION_WARNING

// NO_VERSION_WARNING-NOT: CUDA version is newer than the latest
// NO_VERSION_WARNING-NOT: unknown CUDA version

// For contrast, an unrecognized future version does warn. This guards against
// "fixing" the case above by widening the version ladder without bound.
// RUN: %clang -### --target=x86_64-linux -x cuda --offload-arch=sm_90 -c \
// RUN:   --cuda-path=%S/Inputs/CUDA-new/usr/local/cuda %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=TOO_NEW

// TOO_NEW: CUDA version is newer than the latest{{.*}} supported version
