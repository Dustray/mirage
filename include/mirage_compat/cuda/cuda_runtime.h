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

// Mirage CUDA Runtime compatibility layer for HIP toolchains (Hygon DTK /
// ROCm). When mirage_runtime (or the MPK JIT-compiled kernel) is built with
// hipcc (= `dcc -x hip`), Mirage sources keep including the CUDA headers they
// were written against; this header shadows <cuda_runtime.h> and maps the
// small CUDA Runtime API surface that Mirage actually uses onto the real HIP
// runtime headers.
//
// Unlike the DTK cudamocker headers, this layer pulls in the genuine HIP
// headers (hip/hip_runtime_api.h, hip/hip_bf16.h, ...), so other real HIP
// headers such as ck_tile/core.hpp can be included in the same translation
// unit without any type/API conflicts.
//
// NOTE: this directory must only appear on the include search path for HIP
// builds (see USE_HIP in CMakeLists.txt / get_compile_command() in
// python/mirage/mpk/persistent_kernel.py). On NVIDIA builds the real CUDA
// toolkit headers must win.

// ==== MIRAGE HIP COMPAT: include the full HIP runtime header (API + device
// builtins such as threadIdx/__shfl_*_sync that CUDA's cuda_runtime.h
// transitively provides) rather than only the API-only header, and the
// library_types header that defines hipDataType (cudaDataType_t). ====
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaDataType_t = hipDataType;

// cuBLAS/cuDNN library data type constants map 1:1 onto the hipDataType enum
// values in hip/library_types.h (same numeric encoding as CUDA's
// library_types.h).
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_8I HIP_R_8I
#define CUDA_C_32F HIP_C_32F
#define CUDA_C_64F HIP_C_64F
#define CUDA_C_16F HIP_C_16F
#define CUDA_C_8I HIP_C_8I
#define CUDA_R_8U HIP_R_8U
#define CUDA_C_8U HIP_C_8U
#define CUDA_R_32I HIP_R_32I
#define CUDA_C_32I HIP_C_32I
#define CUDA_R_32U HIP_R_32U
#define CUDA_C_32U HIP_C_32U
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_C_16BF HIP_C_16BF
#define CUDA_R_4I HIP_R_4I
#define CUDA_C_4I HIP_C_4I
#define CUDA_R_4U HIP_R_4U
#define CUDA_C_4U HIP_C_4U
#define CUDA_R_16I HIP_R_16I
#define CUDA_C_16I HIP_C_16I
#define CUDA_R_16U HIP_R_16U
#define CUDA_C_16U HIP_C_16U
#define CUDA_R_64I HIP_R_64I
#define CUDA_C_64I HIP_C_64I
#define CUDA_R_64U HIP_R_64U
#define CUDA_C_64U HIP_C_64U
#define CUDA_R_8F_E4M3 HIP_R_8F_E4M3
#define CUDA_R_8F_E5M2 HIP_R_8F_E5M2

// ---------------------------------------------------------------------------
// Enum constants (macros: plain value substitution, identical semantics)
// ---------------------------------------------------------------------------
#define cudaSuccess hipSuccess

// cudaMemcpyKind
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

// Stream / event creation flags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming

// Function attributes
#define cudaFuncAttributeMaxDynamicSharedMemorySize                               \
  hipFuncAttributeMaxDynamicSharedMemorySize

// Device limits
#define cudaLimitPrintfFifoSize hipLimitPrintfFifoSize

// Device attributes
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

// ---------------------------------------------------------------------------
// Runtime API functions
// ---------------------------------------------------------------------------
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemset hipMemset

#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent

#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy

#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor                             \
  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaPointerGetAttributes hipPointerGetAttributes

#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

// ---------------------------------------------------------------------------
// Device runtime
//
// Kernel launches (`<<<...>>>`) and the device builtins (blockIdx, threadIdx,
// __syncthreads, atomicAdd, __shfl_*, ...) are compiler builtins in HIP
// language mode (-x hip) and need no header support. Device-only runtime
// entry points used by Mirage device code are provided by
// hip/hip_runtime_api.h above.
// ---------------------------------------------------------------------------
