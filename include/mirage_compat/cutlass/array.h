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

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

// Minimal Array<T, N> used by Mirage activation functions in
// include/mirage/utils/cuda_helper.h. This is intentionally not a full
// CUTLASS Array implementation; it only exposes the operators the main library
// needs.
template <typename T, int N>
struct Array {
  T data[N];

  CUTLASS_HOST_DEVICE
  T const &operator[](int i) const {
    return data[i];
  }

  CUTLASS_HOST_DEVICE
  T &operator[](int i) {
    return data[i];
  }

  CUTLASS_HOST_DEVICE
  Array() = default;

  CUTLASS_HOST_DEVICE
  Array(T value) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      data[i] = value;
    }
  }

  CUTLASS_HOST_DEVICE
  Array operator+(Array const &rhs) const {
    Array result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = data[i] + rhs[i];
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array operator-(Array const &rhs) const {
    Array result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = data[i] - rhs[i];
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array operator*(Array const &rhs) const {
    Array result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = data[i] * rhs[i];
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array operator/(Array const &rhs) const {
    Array result;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = data[i] / rhs[i];
    }
    return result;
  }
};

} // namespace cutlass
