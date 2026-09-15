# Mirage CUTLASS Compatibility Layer

This directory provides a minimal CUTLASS API compatibility layer for building
Mirage's main library (`mirage_runtime`) on non-NVIDIA toolchains such as the
Hygon DTK / HCU (AMD CDNA) compiler.

The real CUTLASS submodule (`deps/cutlass`) contains NVIDIA PTX and
SM-specific code that cannot compile on AMD hardware. However, the parts of
CUTLASS that Mirage's main library actually uses are tiny: a few macros
(`CUTLASS_DEVICE`, `CUTLASS_HOST_DEVICE`), `cutlass::half_t`,
`cutlass::Array<T, N>`, `cutlass::MatrixCoord`, and `cutlass::fast_exp_op`.

By placing this directory ahead of `deps/cutlass/include` in the include
search path, the compiler resolves the small set of CUTLASS headers that
Mirage includes to these minimal implementations instead of the full CUTLASS
tree. This keeps Mirage source code close to upstream and avoids scattering
`#ifdef MIRAGE_USE_CUTLASS` guards throughout the codebase.

## Scope

Only the symbols required by the following files are implemented:

- `include/mirage/utils/cuda_helper.h`
- `include/mirage/threadblock/cuda/*.h`
- `include/mirage/threadblock/smem_tensor.h`
- `include/mirage/cpu/cmem_tensor.h`
- `src/kernel/cuda/*.cu`
- `src/threadblock/cuda/*.cu`

JIT-only headers under `include/mirage/transpiler/runtime/` and Blackwell task
headers under `include/mirage/persistent_kernel/tasks/` are **not** covered by
this layer; they are compiled at runtime by the target CUDA toolchain and are
not part of `mirage_runtime`.
