/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Minimal CUTLASS compatibility header for non-NVIDIA toolchains (e.g. Hygon
// DTK / HCU). This provides the macros and small types that Mirage's main
// library actually uses, without pulling in the full CUTLASS codebase which
// contains NVIDIA PTX that cannot compile on AMD CDNA hardware.

#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE __device__
#endif

#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE __host__ __device__
#endif

#ifndef CUTLASS_GEMM_LOOP
#define CUTLASS_GEMM_LOOP
#endif

// CUTLASS uses a custom namespace for cmath helpers. On non-NVIDIA
// toolchains just fall back to the global namespace.
#ifndef CUTLASS_CMATH_NAMESPACE
#define CUTLASS_CMATH_NAMESPACE
#endif

// CUTLASS uses CUTE_GCC_UNREACHABLE in a few switch defaults. Provide a
// portable no-op definition when compiling without the full CUTLASS tree.
#ifndef CUTE_GCC_UNREACHABLE
#define CUTE_GCC_UNREACHABLE
#endif

// Declare the cutlass namespace so that upstream files that write
// `using namespace cutlass;` remain valid even when the full CUTLASS
// library is not present.
namespace cutlass {
} // namespace cutlass
