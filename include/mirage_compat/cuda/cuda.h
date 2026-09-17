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

// CUDA driver API compatibility: Mirage includes <cuda.h> only for the
// device-side CUmemoryPool-like types and CUstream / CUcontext. The HIP
// runtime header provides compatible device-side builtins for the runtime
// API surface, which covers everything Mirage's `#include <cuda.h>` path
// uses from host code. Device-side PTX intrinsics in tma*.cuh etc. are
// already guarded by `__HIP_PLATFORM_AMD__` / `__AMDGCN__` upstream, so they
// never reach the driver API on the HIP path.

#include <hip/hip_runtime_api.h>

using CUcontext = hipCtx_t;
using CUstream = hipStream_t;
using CUresult = hipError_t;

#define CUDA_SUCCESS hipSuccess
