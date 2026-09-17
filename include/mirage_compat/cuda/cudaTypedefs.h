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

// CUDA Typedefs.h compatibility: the Blackwell/Hopper persistent-kernel task
// headers include <cudaTypedefs.h> for a small set of legacy CUDA typedefs.
// On the HIP build path only the generic (non-Blackwell) task directory is
// compiled, so this header is a no-op shim. If any symbols are needed
// later, they can be defined here as aliases of the corresponding HIP types.

#include <hip/hip_runtime_api.h>
