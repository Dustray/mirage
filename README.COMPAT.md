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
> 注意：DTK/HCU 环境在构建前需要先 `source /opt/dtk/cuda/env.sh`。

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
- 通用路径（`ampere/`、`tasks/common/`）中的 PTX 内联汇编已增加 HIP 分支处理，详见第 8 节。
- 在 HCU 上实际执行 megakernel 时，JIT 生成的 kernel 仍可能包含 Blackwell/Hopper 专用 PTX，需要进一步处理或限制只使用 ampere 路径。

## 7. 后续工作

- [ ] 验证 Python 包能否完整安装并导入 `mirage`。
- [ ] 运行简单的 fingerprint/operator 测试，确认数值正确性。
- [ ] 评估是否需要为 HCU 实现自定义 kernel backend，替代 JIT 生成的 CUDA kernel。

## 8. PTX/ASM 内联汇编兼容化

Mirage persistent kernel 在通用路径（`ampere` 及 `tasks/common`）中使用了少量 PTX 内联汇编。为了让这些代码在 HCU/ROCm 上能编译，对以下文件做了 `#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)` 分支：

- `include/mirage/persistent_kernel/profiler.h`
  - `sleep_cycles()` / `get_timestamp()`：NVIDIA 路径保留 `%globaltimer_lo` PTX；HIP 路径改用 `__builtin_amdgcn_s_memrealtime()`。
- `include/mirage/persistent_kernel/mpk_atoms.cuh`
  - `atom_add_release_gpu_s32/u64`、`atom_cas_release_gpu_u64`：HIP 路径改用 `atomicAdd` / `atomicCAS` + `__builtin_amdgcn_fence(..., "agent")`。
  - `ld_acquire_gpu_u64`、`ld_acquire_sys_u64`、`ld_relaxed_gpu_u64`、`st_relaxed_gpu_u64`：HIP 路径改用 `__atomic_load_n` / `__atomic_store_n` + 合适的 `__builtin_amdgcn_fence`。
  - `ld_acquire_sys_i32`、`st_release_sys_i32`：HIP 路径改用 `__atomic_load_n` / `__atomic_store_n` + `__builtin_amdgcn_fence(..., "system")`。
  - 新增 `threadfence_gpu()`：HIP 路径映射到 `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")`。
- `include/mirage/persistent_kernel/tasks/common/utils.cuh`
  - `shfl_xor_sync`：HIP 路径改用 `__shfl_xor(x, lane_mask, warpSize)`。
  - `ptx_exp2` / `ptx_log2`：HIP 路径改用标准 `exp2f` / `log2f`。
- `include/mirage/persistent_kernel/tasks/common/copy_sm80.cuh`
  - `cp.async.*` 与 `ldmatrix` 是 Ampere+ 特性，在 HCU 上 `__CUDA_ARCH__` 不会 >= 800，因此 `CP_ASYNC_SM80_ENABLED` 不会定义，这些函数在 HIP 路径下为空操作；NVIDIA 路径保持原 PTX 不变。

所有修改都保留 NVIDIA 路径的原始 PTX 代码，仅在 HIP/HCU 编译时走替代实现。
