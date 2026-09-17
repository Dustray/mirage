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

// CUDA pipeline primitives compatibility for HIP. AMD GPUs on the DTK path
// have no cp.async-equivalent hardware pipeline, so the primitives degrade to
// synchronous copies / no-op fences, which is semantically safe (just not
// asynchronous).

#include <cstddef>
#include <cstring>

// CUDA packs size (upper bits) and log2(alignment) (lowest 4 bits) into one
// operand: size = size_and_align >> 4, align = 1 << (size_and_align & 0xF).
__device__ __forceinline__ void __pipeline_memcpy_async(void *__restrict__ dst_shared,
                                                        const void *__restrict__ src_global,
                                                        size_t size_and_align,
                                                        size_t zfill = 0) {
  size_t const size = size_and_align >> 4;
  __builtin_memcpy(dst_shared, src_global, size - zfill);
  if (zfill != 0) {
    __builtin_memset(reinterpret_cast<char *>(dst_shared) + (size - zfill), 0, zfill);
  }
}

__device__ __forceinline__ void __pipeline_commit() {
  // No hardware pipeline to commit on the HIP path.
}

__device__ __forceinline__ void __pipeline_wait_prior(size_t /*n*/) {
  // Copies above are synchronous; make the data visible to the block.
  __threadfence_block();
}
