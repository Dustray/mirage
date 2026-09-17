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

// CUDA cublas_v2.h compatibility: map the cuBLAS symbols used by Mirage
// (cublasHandle_t in the main library, cublasGemmStridedBatchedEx in the
// JIT-compiled matmul kernel) onto hipBLAS. The hipBLAS API mirrors the
// cuBLAS v2 signatures 1:1; only the enum type/constant names differ.

// Upstream cublas_v2.h transitively pulls in the driver/runtime types
// (cudaStream_t, ...). Include our runtime shim first so headers using
// cublasHandle_t together with cudaStream_t keep compiling.
#include <cuda_runtime.h>
#include <hipblas/hipblas.h>

using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;
using cublasOperation_t = hipblasOperation_t;
using cublasComputeType_t = hipblasComputeType_t;

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T

#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_16F_PEDANTIC HIPBLAS_COMPUTE_16F_PEDANTIC
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_PEDANTIC HIPBLAS_COMPUTE_32F_PEDANTIC

#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetStream hipblasGetStream
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx

// CUDA's cublasGetStatusName/StatusString have no hipBLAS equivalents; the
// closest API is hipblasStatusToString, which covers the same use case
// (human-readable status in error messages).
#define cublasGetStatusName(status) hipblasStatusToString(status)
#define cublasGetStatusString(status) hipblasStatusToString(status)
