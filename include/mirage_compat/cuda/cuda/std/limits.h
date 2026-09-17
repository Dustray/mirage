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

// CUDA <cuda/std/*> compatibility: the common sampling task and the Blackwell
// task headers include <cuda/std/limits> and <cuda/std/cstdint>. On the HIP
// build path only the common sampling task is reachable; std::numeric_limits
// and <cstdint> provide the same API, so we just forward to the standard
// library.

#include <limits>
#include <cstdint>

namespace cuda {
namespace std {
using ::std::numeric_limits;
} // namespace std
} // namespace cuda
