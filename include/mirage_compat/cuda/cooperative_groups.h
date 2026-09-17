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

// CUDA cooperative_groups compatibility: forward to HIP's cooperative groups
// implementation. hip/hip_cooperative_groups.h provides thread_block,
// this_thread_block(), tiled_partition<Size>() and thread_block_tile with the
// CUDA API shape (the implementation lives in
// namespace cooperative_groups { namespace v<N> with a using-directive), so
// code like `cooperative_groups::tiled_partition<K>(
// cooperative_groups::this_thread_block()).sync()` works unmodified.

#include <hip/hip_cooperative_groups.h>
