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

## CUDA compat layer (`cuda/`)

In addition to the CUTLASS layer above, this directory also provides a CUDA
compatibility layer for builds that target the HIP toolchain with `hipcc`
(`dcc -x hip`). Mirage sources keep including the CUDA headers they were
written against (`<cuda_runtime.h>`, `<cuda_bf16.h>`, `<cooperative_groups.h>`,
...). The `cuda/` directory shadows those headers and maps the CUDA API surface
that Mirage actually uses onto the **real HIP headers**
(`hip/hip_runtime_api.h`, `hip/hip_bf16.h`, `hip/hip_fp16.h`,
`hip/hip_cooperative_groups.h`, `<hipblas/hipblas.h>`):

- `cuda_runtime.h` / `cuda_runtime_api.h`: CUDA Runtime API mapping (types,
  enum constants, and the ~30 runtime calls used by the main library).
- `cuda_bf16.h`: `__nv_bfloat16(2)` -> `__hip_bfloat16(2)`; all bf16
  conversion / arithmetic intrinsics already share CUDA naming in HIP.
- `cuda_fp16.h`: forward to `hip/hip_fp16.h` (`__half`, `__half2`, `half`
  and the intrinsics use identical names).
- `vector_types.h`: `dim3`, `int2`, `float4`, ... from
  `hip/hip_vector_types.h`.
- `cooperative_groups.h`: forward to `hip/hip_cooperative_groups.h`
  (thread_block, this_thread_block, tiled_partition, thread_block_tile).
- `cuda.h`: driver-API shim (forward to the HIP runtime API).
- `cublas_v2.h`: cuBLAS v2 symbols -> hipBLAS.
- `cudaTypedefs.h`: no-op shim (only Blackwell task headers need it).
- `cuda/std/limits`: forwards to `<limits>`.
- `cuda/pipeline`, `cuda_pipeline.h`, `cuda_pipeline_primitives.h`:
  synchronous fallbacks for the CUDA cp.async pipeline primitives.

Unlike the DTK cudamocker headers, this layer pulls in genuine HIP headers,
so other real HIP headers such as `ck_tile/core.hpp` (Composable Kernel) can
be included in the same translation unit without any type/API conflicts.

This directory must only be added to the include search path for HIP builds:
`-DUSE_HIP=ON` in CMake (main library), and the `hipcc` path in
`get_compile_command()` (MPK JIT). On NVIDIA builds the real CUDA toolkit
headers must win.
