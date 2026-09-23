/* Copyright 2025 CMU
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

// NUM_THREADS: The number of threads used for loop strides in kernels
// Must match the actual thread count used when launching kernels
// MIRAGE HIP COMPAT: 与 megakernel 对齐——AMD 用 256 线程 + native 64-lane
// wave；128 线程与 CK GEMM 策略静态 4-warp 假设不匹配（输出全零）
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
constexpr int NUM_THREADS = 256;
constexpr int NUM_THREADS_PER_WARP = 64; // Native AMD wavefront size
constexpr int NUM_WARPS = 4;             // 256 / 64 = 4 wavefronts
#else
constexpr int NUM_THREADS = 128;
constexpr int NUM_THREADS_PER_WARP = 32;
constexpr int NUM_WARPS = 4;
#endif
constexpr int WARPGROUP_WARPS = 4;

constexpr float inf = 5e4;
// TODO: only setting this for Hopper can have compilation issues on blackwell
// and presumably ampere
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL) || defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
constexpr int WORKER_NUM_THREADS = 256;   // Grace Hopper/AMD MI300 setting
constexpr int CONSUMER_NUM_THREADS = 128; // Grace Hopper setting
#endif
