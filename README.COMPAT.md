# Mirage Hygon HCU 兼容层说明

本文档记录将 Mirage 移植到海光 HCU（AMD CDNA 架构，通过 DTK 兼容层）的修改思路、编译方法和使用方式。

## 1. 背景与目标

- **硬件/工具链**：海光 HCU + DTK 26.04 兼容层，`dcc` 26.02.0-0（clang 17.0.0）模拟 nvcc CUDA 12.6。
- **核心问题**：Mirage 主库依赖 NVIDIA CUTLASS，其中包含大量 NVIDIA PTX/ASM，无法直接在 AMD CDNA 上编译。
- **目标**：让主库 `mirage_runtime` 在不修改大量源码的前提下编译通过，同时保留后续同步官方代码的能力。

## 2. 修改思路

### 2.1 最小化 CUTLASS 兼容层

不直接修改 Mirage 源码中的 CUTLASS 调用，而是在 `include/mirage_compat/cutlass/` 下提供一组 stub 头文件，覆盖主库实际使用的少量 CUTLASS 符号：

- `cutlass.h`：定义 `CUTLASS_DEVICE`、`CUTLASS_HOST_DEVICE`、`CUTLASS_GEMM_LOOP`，以及空的 `namespace cutlass {}` 和 fallback 宏（`CUTLASS_CMATH_NAMESPACE`、`CUTE_GCC_UNREACHABLE`、`CUTLASS_PRAGMA_UNROLL`）。
- `numeric_types.h`：提供 `cutlass::half_t`（映射到 `half`）。
- `array.h`：最小化 `cutlass::Array<T, N>` 实现。
- `fast_math.h`：提供 `cutlass::fast_exp_op<T>`。
- `matrix_coord.h`：提供 `cutlass::MatrixCoord`，接口与上游 CUTLASS 保持一致（`row()` / `column()`）。

### 2.2 包含路径优先级

在 `CMakeLists.txt` 和 `setup.py` 中，将 `include/mirage_compat` 放在 `deps/cutlass/include` **之前**。这样编译器会优先使用 stub 头文件，不会进入真正的 CUTLASS 头文件，从而避开 PTX/ASM 错误。

### 2.3 移除未使用的 warp GEMM 头文件

`src/kernel/cuda/customized_kernel.cu` 原本包含 `mirage/warp/cuda/matmul.h`，该头文件内部大量引用 CUTLASS warp GEMM 头文件，但 `GemmExecutor` 类在整个项目中从未被实例化。因此将其注释掉，避免把大量用不到的 CUTLASS 代码拉入主库编译。

### 2.4 为什么用兼容层而不是直接改源码

- **减少 Mirage 源码改动**：只有 1 处包含被注释掉，其余改动都在 `include/mirage_compat/` 中。
- **方便同步官方代码**：官方 CUTLASS 头文件路径被 stub 替换，Mirage 源码保持原样，后续 rebase/merge 冲突更少。
- **可维护**：兼容层只包含主库实际用到的符号，范围明确。

## 3. 文件变更清单

### 新增文件

- `include/mirage_compat/README.md`
- `include/mirage_compat/cutlass/cutlass.h`
- `include/mirage_compat/cutlass/numeric_types.h`
- `include/mirage_compat/cutlass/array.h`
- `include/mirage_compat/cutlass/fast_math.h`
- `include/mirage_compat/cutlass/matrix_coord.h`

### 修改文件

- `CMakeLists.txt`：添加 `include_directories(include/mirage_compat)`，并确保在 `deps/cutlass/include` 之前。
- `setup.py`：将 `include/mirage_compat` 加入 cython `include_dirs` 和 `src_dirs`。
- `src/kernel/cuda/customized_kernel.cu`：注释掉 `mirage/warp/cuda/matmul.h` 的包含。

## 4. 编译方法

### 4.1 环境准备

确保已安装：

- CMake >= 3.24
- DTK / ROCm 工具链（`dcc` 在 PATH 中）
- Python 3.10 及虚拟环境
- Z3、nlohmann/json、Rust 等依赖已就绪

### 4.2 CMake 配置

```bash

cd /public/home/panyq/yiny/projects/mirage
export MIRAGE_HOME=$(pwd)
rm -rf build && mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DZ3_CXX_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/z3/include \
  -DZ3_LIBRARIES=/usr/local/lib/python3.10/dist-packages/z3/lib/libz3.so \
  -DABSTRACT_SUBEXPR_LIB=$(MIRAGE_HOME)/build/abstract_subexpr/release \
  -DABSTRACT_SUBEXPR_LIBRARIES=$(MIRAGE_HOME)/build/abstract_subexpr/release/libabstract_subexpr.so \
  -DFORMAL_VERIFIER_LIB=$(MIRAGE_HOME)/build/formal_verifier/release \
  -DFORMAL_VERIFIER_LIBRARIES=$(MIRAGE_HOME)/build/formal_verifier/release/libformal_verifier.so
```

> 注意：Z3 路径请根据实际环境调整。如果 `abstract_subexpr` 和 `formal_verifier` 尚未构建，需要先按项目原流程构建这两个 Rust 依赖。

### 4.3 编译主库

```bash
cd $(MIRAGE_HOME)/build
make -j mirage_runtime
```

编译成功后会在 `build/` 下生成 `libmirage_runtime.a`。

## 5. 使用方法

### 5.1 作为 C++ 库使用

在 CMake 项目中链接 `mirage_runtime`：

```cmake
add_subdirectory($MIRAGE_HOME mirage)
target_link_libraries(your_target mirage_runtime)
```

### 5.2 Python 包安装

```bash
cd $(MIRAGE_HOME)
source mirageenv/bin/activate
python setup.py install
```

> 当前阶段主库 `mirage_runtime` 已可编译，但 Python 包的完整安装可能还需要进一步处理 CUDA/ROCm 运行时和 JIT 相关部分。

### 5.3 验证编译

```bash
cd $(MIRAGE_HOME)/build
make -j mirage_runtime
ls -lh libmirage_runtime.a
```

## 6. 已知限制

- 本兼容层仅覆盖主库 `mirage_runtime` 编译所需的 CUTLASS 符号。
- Blackwell/Hopper 专用任务头文件（`include/mirage/persistent_kernel/tasks/` 下的 `blackwell_*.cuh`）属于 JIT 生成代码，不参与主库编译，当前未做处理。
- 实际在 HCU 上执行生成的 kernel 还需要进一步将 CUDA kernel 代码转译为 HIP/ROCm，或依赖 DTK 的 CUDA 兼容运行时支持。

## 7. 后续工作

- [ ] 验证 Python 包能否完整安装并导入 `mirage`。
- [ ] 运行简单的 fingerprint/operator 测试，确认数值正确性。
- [ ] 评估是否需要为 HCU 实现自定义 kernel backend，替代 JIT 生成的 CUDA kernel。
