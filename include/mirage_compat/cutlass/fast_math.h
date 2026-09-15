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

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <cmath>

namespace cutlass {

// fast_exp_op is used by include/mirage/utils/cuda_helper.h for the
// element-wise activation functions.
template <typename T>
struct fast_exp_op;

template <int N>
struct fast_exp_op<Array<half_t, N>> {
  CUTLASS_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &rhs) const {
    Array<half_t, N> result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = half_t(expf(__half2float(rhs[i])));
    }
    return result;
  }
};

} // namespace cutlass
