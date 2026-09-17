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

// CUDA fp16 compatibility: forward to the real HIP half implementation.
// hip/hip_fp16.h (amd_detail/amd_hip_fp16.h) defines __half, __half2, `half`,
// and all of the conversion / arithmetic intrinsics that Mirage uses
// (__float2half, __half2float, __float2half2_rn, __half22float2, __hadd,
// __hmul, __hsub, __hdiv, __hfma, __hmax/__hmin, ...) under their CUDA names,
// so a plain include is sufficient.

#include <hip/hip_fp16.h>
