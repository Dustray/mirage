# Mirage 海光 HCU (gfx936) 适配说明

本文档记录将 Mirage 移植到海光 HCU（AMD CDNA 架构，DTK 26.04 兼容层）后的**适配改动总览、构建方法与测试方法**。详细的适配过程（根因分析、逐阶段战役记录）见 [COMPAT_PORTING_DETAILS.md](COMPAT_PORTING_DETAILS.md)。

## 1. 背景

- **硬件/工具链**：海光 HCU gfx936 + DTK 26.04（hipcc，clang 17），Python 3.10。
- **核心问题**：Mirage 依赖 NVIDIA CUTLASS/PTX，persistent kernel 运行时依赖 CUDA 内存模型语义；gfx936 为 wave64 架构，与 NVIDIA 32-lane warp 模型差异大。
- **目标**：Qwen3-8B megakernel（persistent kernel）在 HCU 上正确且高性能运行。当前已达成：`demo_fleet` 18.34 ms/token，与 megakernel 原版（18.3 ms）持平。

## 2. 适配改动总览

### 2.1 构建层（HIP 默认路径）

- HIP 是**默认构建路径**：`CMakeLists.txt` 选项 `USE_HIP ON`，`setup.py` `_mirage_hip_build` 默认 True；设 `MIRAGE_USE_HIP=0` 回退 CUDA。
- `include/mirage_compat/cuda` 置于头文件搜索路径最前，屏蔽 cudamocker；`.cu` 文件以 hipcc `-x hip` 编译。
- CUTLASS 以 `include/mirage_compat/cutlass/` 下的 stub 头文件最小化替换（置于 `deps/cutlass/include` 之前），避免拉入 NVIDIA PTX/ASM。
- PTX 内联汇编按 `__AMDGCN__`/`__HIP_PLATFORM_AMD__` guard 双路径保留：`mpk_atoms.cuh`（原子/loads-fences）、`profiler.h`（时钟）、`tasks/common/utils.cuh`（shuffle/exp2）、`copy_sm80.cuh`（cp.async 空操作）等。

### 2.2 gfx936 wave64 运行时适配

- AMD 构建 worker CTA 使用 **256 线程 + 原生 64-lane wave**（`worker_config.h`/`persistent_kernel.cuh` 的 `NUM_THREADS=256`、`NUM_THREADS_PER_WARP=64`，`__HIP_PLATFORM_AMD__` guard），与 megakernel 一致；128 线程回退会破坏 CK GEMM 静态 4-warp tile 假设（GEMM 半输出/零输出 + 死锁）。
- `lane_id()` 修复为 `threadIdx.x & (NUM_THREADS_PER_WARP - 1)`（原 32-lane 假设导致同一 wave 的 lane 0/32 双写共享内存，argmax 丢失真 max，文本退化）。
- Warp shuffle 使用直接 warp ID 计算而非 shuffle 实现；`__shfl_sync` mask 按 wave64 语义处理。
- MFMA 指令以 inline asm 实现（DTK 缺 `mai-insts` feature），modifier 使用 `0,0,0`。
- `__shared__` 数组带非平凡默认构造时改用 defaulted 构造，避免动态初始化错误。
- HIP API 显式 `reinterpret_cast` 函数指针。

### 2.3 内存一致性修复（gfx9 vL1）

- 读侧：worker/scheduler 自旋读 `ld_acquire_gpu_u64`/`ld_relaxed_gpu_u64` → `ld_local_u64`（volatile）+ 自旋后 `fence_local`（4 处）。gfx936 分支的 `ld_nt_u64` 是裸 `global_load` 无 glc，计数器行进 vL1 后自旋永远读旧值。
- 写侧：7 处发布原子（`atom_add_release_gpu_u64`/CAS 无 release 语义）前补 `threadfence_gpu`（worker 完成、worker→sched CAS、terminate、派发 ×4）。

### 2.4 任务移植（自 megakernel，CC 93）

- **split-kv attention**：`paged_attention_split_kv_mi300` + `merge` 五件套（task_register.h/cc、graph.cc 名字分发、persistent_kernel.py CC93 elif、runtime.cc 的 task_type_to_name 与 task_metadata：`kv_idx=bid.z`、`merge_task_offset=bid.y`、`request_id=bid.x`）。
- **splitk_linear_res_atomic**（o_proj/down_proj 的 split-K + 残差原子加）：同五件套模式，枚举 `TASK_SPLITK_LINEAR_RES_ATOMIC_MI300=133`，5 输入 1 输出（input/weight/residual/workspace f32/done_counter i32 → output bf16）；设备内核复用 `tasks/mi300/linear_mi300.cuh` 现有实现。
- 对应 demo 层：`demo/qwen3/demo_fleet.py`（megakernel demo 的 mirage 版配套）与 `demo/qwen3/demo.py`（profiler 缓冲区放大 64 倍）。

### 2.5 profiling 机制

- 自 megakernel 移植 `profiling_num_iters` 编译期注入：`RuntimeConfig` 字段 + `-DMPK_PROFILING_NUM_ITERS=N`（env `MPK_PROFILING_ITERS`，persistent_kernel.py）；OFFLINE 请求完成条件改三分支（TEST_MODE / PROFILING 封顶 / 正常）。
- profiler 导出优化：`_decode_events` 连续零值早退（全量扫描 38.4M 槽位 ≈3min → 秒级）；CSV dangling BEGIN 降级为告警（terminate 时刻在途任务，良性）。
- `demo.py` profiler 缓冲区放大 64 倍（30000×1280）；`demo_fleet.py` 再放大 8 倍（240000×1280，全程 1024 迭代安全）。

## 3. 构建方法

### 3.1 一键构建

```bash
cd /public/home/panyq/yiny/projects/mirage
./build.sh                     # HIP 构建（默认），gfx936
MIRAGE_USE_HIP=0 ./build.sh    # 回退 CUDA 路径（NVIDIA 环境）
```

`build.sh` 做的事：设置 `MIRAGE_USE_HIP`/`AMDGPU_TARGETS`（无需 source DTK 环境，镜像默认 PATH 已含 hipcc、`ROCM_PATH=/opt/dtk`，setup.py/CMakeLists 的 HIP 分支均按此解析）→ `find python -name "*.pyx" -exec touch {} +`（规避 distutils 重链陷阱，改 `src/kernel/*.cc` 后必须）→ `python setup.py build_ext --inplace` → 校验产物。

### 3.2 验证产物含新代码

```bash
SO=python/mirage/core.cpython-310-x86_64-linux-gnu.so
readelf -p .rodata $SO | grep <新任务名/符号名>
```

> 不要用 `strings` 验证：`.debug_str` 含 DWARF 枚举名会产生假阳性。

### 3.3 CMake 主库构建（可选，C++ 库用途）

```bash
export MIRAGE_HOME=$(pwd)
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DZ3_CXX_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/z3/include \
  -DZ3_LIBRARIES=/usr/local/lib/python3.10/dist-packages/z3/lib/libz3.so
make -j mirage_runtime    # 生成 libmirage_runtime.a（USE_HIP 默认 ON）
```

## 4. 测试方法

### 4.1 一键测试

```bash
./test.sh                          # demo_fleet.py，1024 tokens（性能+正确性基线）
./test.sh demo.py                  # mirage 原版 demo（无 splitk linear 优化）
./test.sh demo_fleet.py 128        # 自定义生成 token 数
PROF_ITERS=32 ./test.sh            # 剖析模式，产出 perfetto-trace + csv
PROF_ITERS=1024 ./test.sh          # 剖析全程（demo_fleet 缓冲区已放大，安全）
```

### 4.2 验证点

- **EXIT=0**；非剖析模式生成 token 数 = `max_seq_length(1088) - prompt(64)` = 1024（`--max-new-tokens` 不进内核，OFFLINE 终止由 prompt_length 控制）。
- **per-token 延迟基线**：demo_fleet 18.34 ms ≈ megakernel 18.3 ms < demo.py 21.5 ms。
- **文本连贯**：`<think>` 推理块 + 结构化正文（与 torch 基线模式一致）。
- 剖析模式无文本输出属预期；`MPK_PROFILING_ITERS` 必须为正整数（0 = 无限跑，写穿缓冲区 VMFault）。

### 4.3 对照测试（megakernel）

```bash
# 容器内（同镜像、同模型）；脚本口径：megakernel 侧 unset MIRAGE_USE_HIP，
# PYTHONPATH/MIRAGE_HOME 指向 megakernel，--profiling --trace-name /tmp/mega_prof_N
docker exec mirage bash /tmp/mega_prof_32.sh    # 32 迭代，与 mirage PROF_ITERS=32 同口径
```

产物对照（宿主机 `/tmp/traces/`）：`megakernel_32.perfetto-trace`（55MB）、`mirage_fleet.perfetto-trace`（55MB）等，ui.perfetto.dev 打开。两项目 32 迭代事件分布完全对齐（SPLITK_LINEAR_RES_ATOMIC=9216/iter、SPLIT_KV=2592/iter、MERGE=288/iter）。

## 5. 已知限制

- 剖析模式 per-token 延迟含导出开销，不代表真实性能（以非剖析运行为准）。
- `demo.py` 的 profiler 缓冲区为 32 迭代容量级，剖析全程需用 `demo_fleet.py`（已放大 8 倍）。
- CK-FMHA 路径（`USE_CK_FMHA=1`）在本环境未启用，测试统一 `USE_CK_FMHA=0`。
- megakernel/mirage 的 OFFLINE PROFILING 完成条件为**替换**而非叠加正常终止条件，`MPK_PROFILING_ITERS=0` 会无限跑（详见 COMPAT_PORTING_DETAILS.md）。
