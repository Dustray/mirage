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

// CUDA bf16 compatibility: map CUDA's __nv_bfloat16 types onto the real HIP
// bfloat16 implementation (hip/hip_bf16.h). All conversion and arithmetic
// intrinsics used by Mirage (__float2bfloat16, __bfloat162float, __bfloat1622float2,
// __hadd/__hsub/__hmul/__hdiv, __heq/__hne/__hlt/__hle/__hgt/__hge/__hneg,
// __bfloat16_as_ushort, ...) share their CUDA naming in HIP, so only the type
// names differ.
//
// This pulls in the genuine HIP header, so it can coexist with ck_tile and
// other real HIP headers in the same translation unit.

#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
using __nv_bfloat16_raw = __hip_bfloat16_raw;
using __nv_bfloat162_raw = __hip_bfloat162_raw;
