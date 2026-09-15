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

namespace cutlass {

// Minimal MatrixCoord used by src/kernel/cuda/customized_kernel.cu.
struct MatrixCoord {
  int row;
  int column;

  CUTLASS_HOST_DEVICE
  MatrixCoord() : row(0), column(0) {}

  CUTLASS_HOST_DEVICE
  MatrixCoord(int r, int c) : row(r), column(c) {}

  CUTLASS_HOST_DEVICE
  int row_value() const {
    return row;
  }

  CUTLASS_HOST_DEVICE
  int column_value() const {
    return column;
  }
};

} // namespace cutlass
