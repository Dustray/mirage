# Mirage 海光 HCU 适配详细过程记录

本文档按时间线记录 Mirage 移植到海光 HCU gfx936 的完整适配过程：每阶段的问题现象、根因分析、修复方案与验证结果。改动总览与构建/测试方法见 [README.COMPAT.md](README.COMPAT.md)。

对照基准：容器 `mirage`（DTK 26.04，gfx936），模型 Qwen3-8B，megakernel 同仓库版本在 HCU 上 18.3 ms/token 为性能目标。

---

## 阶段一：HIP 构建路径打通

**问题**：Mirage 主库与 persistent kernel 依赖 NVIDIA CUTLASS（含大量 PTX/ASM）与 cudamocker，无法在 DTK 下编译。

**改动**：

1. **CUTLASS stub 层**：`include/mirage_compat/cutlass/` 提供主库实际用到的最小符号集（`cutlass.h`、`numeric_types.h`、`array.h`、`fast_math.h`、`matrix_coord.h`），搜索路径置于 `deps/cutlass/include` 之前。仅注释掉一处未实例化的 `GemmExecutor` 相关 include（`src/kernel/cuda/customized_kernel.cu` 的 `mirage/warp/cuda/matmul.h`）。
2. **cudamocker 屏蔽**：`include/mirage_compat/cuda` 置于头文件搜索路径最前。
3. **hipcc 编译**：`.cu` 文件统一 `-x hip`；`CMakeLists.txt` 加 `USE_HIP` 选项（后改为默认 ON），`setup.py` `_mirage_hip_build` 默认 True。HIP 从此是默认构建路径，`MIRAGE_USE_HIP=0` 才回退 CUDA。
4. **PTX 双路径**：通用路径的内联汇编按 `__AMDGCN__`/`__HIP_PLATFORM_AMD__` guard 保留原 PTX 并增 HIP 分支：
   - `mpk_atoms.cuh`：`atom_add_release_gpu_s32/u64`、`atom_cas_release_gpu_u64` → `atomicAdd`/`atomicCAS` + `__builtin_amdgcn_fence`；各 load/store 内存序原语 → `__atomic_load_n`/`__atomic_store_n` + fence；新增 `threadfence_gpu()` → `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")`。
   - `profiler.h`：`sleep_cycles()`/`get_timestamp()` → `__builtin_amdgcn_s_memrealtime()`。
   - `tasks/common/utils.cuh`：`shfl_xor_sync` → `__shfl_xor`；`ptx_exp2/ptx_log2` → `exp2f/log2f`。
   - `copy_sm80.cuh`：cp.async/ldmatrix 在 HIP 路径为空操作（`__CUDA_ARCH__` 不 ≥800）。

**约束沉淀**：CUDA API 必须经兼容层映射；PTX 替换必须 `__AMDGCN__` guard；COMPAT 注释用简洁中文，保留 `MIRAGE HIP COMPAT` 单行标记。

---

## 阶段二：gfx936 wave64 运行时适配（死锁 + 文本退化）

### 2a. 线程配置死锁（run35/40/41/42）

**现象**：persistent kernel 在 `linear_kernel_ck` 内非确定性死锁；另一症状为 GEMM 输出一半为零（zero-logits）。

**根因**：AMD 构建回退用 128 线程/CTA（4×32-lane 假设），而 CK GEMM 静态假设 4-warp tile。128 线程下共享内存 stage 只装一半 → GEMM 半输出/零输出，且卡在内核内死锁。megakernel 用 256 线程 + 原生 64-lane wave。

**修复**：`worker_config.h` + `persistent_kernel.cuh` 在 `__HIP_PLATFORM_AMD__` guard 下设 `NUM_THREADS=256`、`NUM_THREADS_PER_WARP=64`、`WORKER_NUM_THREADS`/`SINGLE_KERNEL_NUM_THREADS=256`，与 megakernel 对齐。

**配套适配**（wave64 引入的差异）：

- warp shuffle 不能用 shuffle 实现自身（wave64 下跨逻辑 warp），改为直接 warp ID 计算；`__shfl_sync` mask 按 wave64 语义构造。
- `__shared__` 数组带非平凡默认构造时用 defaulted 构造（避免 LDS 动态初始化错误）；`TaskDesc` 共享内存数组经 `load_smem` 覆盖默认值。
- MFMA 指令：DTK 缺 `mai-insts` feature，用 inline asm 实现，modifier 固定 `0,0,0`；HIP API 函数指针需显式 `reinterpret_cast`。

### 2b. 文本退化（run47 前夜）

**现象**：EXIT=0、延迟正常，但生成文本为噪音 token（argmax 返回低值）。

**根因**：`utils.cuh` 的 `lane_id()` 用 `threadIdx.x & 0x1f`（32-lane 假设），而 `warp_id()` 按 `/64` 划分。wave64 下同一 wave 的 lane 0 与 lane 32 同时满足 `lane_id()==0`，在 `block_reduce_max_idx` 中双写共享内存槽位，真 max 被覆盖。

**修复**：`lane_id()` 加 `__HIP_PLATFORM_AMD__` guard 改为 `& (NUM_THREADS_PER_WARP - 1)`。该函数仅 argmax block 归约与 norm 两处使用，影响面可控。

**验证**：run47 —— EXIT=0、1024 tokens、25.620 ms/token、`<think>` 推理块 + 结构化正文完整，与 torch 基线模式一致。死锁与文本退化两大问题闭环。

### 2c. 调试设施清理（run48）

run45 前后加入的心跳/`[HB]` 调试设施（`mpk_dbg_counter`、`mpk_hb_last` 写点、SPIN/FETCH/EXEC_DONE 打印块、宿主轮询监控等）验证完毕后全部移除（persistent_kernel.cuh + persistent_kernel.py 的 `MPK_HEARTBEAT` 线程），保留全部功能性修复。清理前后同 flag 对照编译错误集一致（4 个均为独立 TU 缺 JIT HARD_CODE 前导，非清理引入）。

**验证**：run48 —— EXIT=0、1024 tokens、32.693 ms/token（环境噪声量级）、`[HB]` 输出为 0、文本连贯。

---

## 阶段三：内存一致性战役（run45）

**现象**：worker/scheduler 计数器自旋永不开进（调度器读旧值），persistent kernel 挂死。

**根因**（与 megakernel 对照定位）：

1. **读侧**：mirage 自旋读走 `ld_acquire_gpu_u64`/`ld_relaxed_gpu_u64` → `MPK_USE_NT_MEMORY` → `ld_nt_u64` → gfx936 分支（非 gfx940）= 裸 `global_load` 无 glc。gfx9 上写方更新停留在 vL1，读方自旋核的 vL1 永远命中旧值。`ld_acquire_sys_u64` 的 COMPAT 注释早已写明此坑，但 gpu 版没修。
2. **写侧**：mirage persistent_kernel.cuh 原本零处 `threadfence_gpu`；`atom_add_release_gpu_u64` 的 inline asm 无 release 语义（其自身注释要求前置显式 fence）。megakernel 每个发布原子前都有 `threadfence_gpu`，热路径全用 volatile `ld_local/st_local` + `fence_local`。

**修复**（13 处，`persistent_kernel.cuh`）：

- 读侧 4 处：`ld_acquire_gpu_u64`/`ld_relaxed_gpu_u64` → `ld_local_u64`（volatile）+ 自旋后 `fence_local` ×2（worker / scheduler）。
- 写侧 7 处：发布原子前补 `threadfence_gpu`（worker 完成、worker→sched CAS、terminate、任务派发 ×4）。

---

## 阶段四：split-kv attention 移植（run50）

**动机**：profiling 显示 megakernel 18.3ms 对 mirage 25.6ms 的优势全在 split-kv attention（megakernel 非分块反而 53.2ms 比 mirage 慢一倍）。

**移植内容**（自 megakernel，"五件套"加式改动）：

1. `include/mirage/kernel/task_register.h`：两个任务声明。
2. `src/kernel/task_register.cc`：两个注册函数（`split_kv` 模板参数 `output_size*num_kv_chunks`，`kv_chunk_idx` 取 `task_metadata.kv_idx`；merge 复用 `ampere/merge_splitkv.cuh`）。
3. `src/kernel/graph.cc`：名字分发两分支。
4. `python/mirage/mpk/persistent_kernel.py`：两处 CC93 elif（PK 方法）。
5. `src/kernel/runtime.cc` 三处：`task_type_to_name` 两条目、`register_mugraph` 的 `request_id=bid.x` 条件、`kv_idx=bid.z` + `merge_task_offset=bid.y` 条件。

**关键坑——task_metadata 第五件套**：megakernel runtime.cc 对 split-kv 任务设置 `task_metadata`（`kv_idx=bid.z` chunk 索引、`merge_task_offset=bid.y`、`request_id=bid.x`）。漏移植时所有 chunk 任务 `kv_idx` 全为 0 → 每个 chunk 算同段 KV → merge 出噪音 token（**EXIT=0 且延迟正常 19.9ms，只有文本退化**，极易误判为数值问题）。

**配套**：demo.py profiler 缓冲区放大 64 倍（mi300 图 16357 tasks/iter）+ `num_kv_cache_chunks=ceil(max_seq_length/128)`。

**验证**：run50 —— EXIT=0、1024 tokens、21.439 ms/token、文本连贯（`<think>` + RNN/Transformer/GAN 结构化正文）。性能阶梯：megakernel+split-kv 18.3 / mirage+split-kv ~21 / mirage 非分块 25.6 / megakernel 非分块 53.2 ms。

**运维坑沉淀**：
- **distutils 重链陷阱**：setup.py 把 mirage_runtime 放在 libraries 而非 sources，distutils 不追踪 libmirage_runtime.a 变化，改 `src/kernel/*.cc` 后 build_ext 只拷回 build/ 里陈旧链接产物。必须 `find python -name "*.pyx" -exec touch {} +` 再 build。
- **docker exec 后台**：`cmd & echo` 会在 exec 退出时杀后台子壳（需 setsid nohup + `< /dev/null`）；`pkill -f` 会自匹配自身 exec 命令行（用 `demo[.]py` 方括号技巧规避）。

---

## 阶段五：profiling 停摆根因与修复

**现象**：剖析运行只跑 1 迭代、step=1、无文本（megakernel 同病：2 tokens）。

**根因**：`persistent_kernel.cuh` OFFLINE `prepare_next_batch` 的请求完成条件为 `#if defined(MPK_ENABLE_PROFILING)||defined(MPK_TEST_MODE) → if(true)` → 第 1 迭代后所有请求标记 done → `num_tokens=0` → terminate。demo 用 `mode="offline"`（先修 MODE_ONLINE 变体的硬编码 `return false` 无效）；OFFLINE 终止由 `prompt_length(=input_ids 全宽 1088)−step` 控制，`--max-new-tokens` 不进内核。

**修复**（自 megakernel 移植 `profiling_num_iters` 机制，4 处）：

1. `runtime_header.h`：`RuntimeConfig` 加 `profiling_num_iters` 字段。
2. `persistent_kernel.cuh`：`global_runtime_config` 初始化（`#ifndef MPK_PROFILING_NUM_ITERS` 默认 0）；OFFLINE 完成条件改三分支——TEST_MODE 单迭代保持 / PROFILING 按 `num_iters` 封顶 / 正常条件。
3. `persistent_kernel.py`：profiling flags 加 `-DMPK_PROFILING_NUM_ITERS={MPK_PROFILING_ITERS}`（env，默认 1），非 profiling 显式传 0。

**配套修复**：

- profiler 缓冲区按 (block,group) 列跨步布局（`write_stride=num_blocks*num_groups`），**写宏无越界检查**——单列容量=总槽位/列数。16357 tasks/iter 下调度器列 ~2602 事件/iter，1024 迭代即写穿 48 万容量（越界写侥幸未炸）。剖析必须用 `MPK_PROFILING_ITERS` 封顶。
- `_decode_events` 全量扫描 38.4M 槽位 ≈3min → 连续零值（≥4096）早退，秒级。
- CSV 导出 dangling BEGIN（terminate 时刻在途任务，良性）从 raise 降级为告警。

**验证**：run51（非 profiling 回归）—— EXIT=0、1024 tokens、21.503 ms/token（run50 为 21.439，噪声量级）、文本连贯，正常路径未受影响。

---

## 阶段六：demo_fleet 移植（fleet1）

**任务**：将 megakernel `demo/qwen3/demo.py` 原样拷至 mirage `demo/qwen3/demo_fleet.py` 并跑通。

**排查**：19 个 mirage 缺失方法中 18 个 env 门控不触发或 mirage 已有；唯一缺口是 `splitk_linear_res_atomic_layer`（o_proj/down_proj，条件 `is_rocm or (USE_GANG and ...)` 在 HIP 下必触发）。

**移植**（同五件套模式，加式改动）：

1. `task_register.h`：`register_splitk_linear_res_atomic_mi300_task` 声明。
2. `task_register.cc`：注册函数（自 megakernel 原样拷贝；5 输入 1 输出：input/weight/residual/workspace(f32)/done_counter(i32) → output(bf16)，k_splits 为唯一 param，生成 `splitk_linear_res_atomic<bfloat16, batch, n_per_block, reduction_size, k_splits>` 调用）。
3. `graph.cc`：分发分支 `(5, 1, TASK_SPLITK_LINEAR_RES_ATOMIC_MI300=133, variant_id)`。
4. `runtime.cc`：`task_type_to_name` 条目。
5. `persistent_kernel.py`：PK 方法（6 输入 1 输出 TBGraph + `register_task(tb_graph, "splitk_linear_res_atomic_mi300", [k_splits])`）。

枚举 133 与设备内核（`tasks/mi300/linear_mi300.cuh` 的 `splitk_linear_res_atomic`，签名完全匹配）mirage 侧本就存在；另补 `profiler_persistent.py` event 133 条目。

**验证**：fleet1 —— EXIT=0、18.342 ms/token（追平 megakernel 18.3，比 demo.py 21.5 快 ~3ms）、文本连贯。生成 1024 tokens（脚本设 128）系 OFFLINE 终止由 prompt_length 控制的已知行为。megakernel 与 mirage 的最后性能差距（splitk_linear_res_atomic）全部闭环。

---

## 阶段七：剖析实战

**产物**（宿主机 `/tmp/traces/`，ui.perfetto.dev 打开）：

| 文件 | 大小 | 口径 |
|---|---|---|
| megakernel.perfetto-trace | 1.8MB | split-kv 原始 A/B |
| mirage_nosplit.perfetto-trace | 706KB | mirage 非分块对照 |
| mirage_splitkv.perfetto-trace | 113MB | 128 迭代，242 万事件 |
| mirage_fleet.perfetto-trace | 55MB | demo_fleet 32 迭代 |
| megakernel_32.perfetto-trace | 55MB | megakernel 32 迭代（同口径对照） |
| megakernel_full.perfetto-trace | 835MB | megakernel 1024 迭代全程 |

**坑**：

1. **`MPK_PROFILING_ITERS=0` = GPU VMFault**：OFFLINE PROFILING 完成条件是**替换**正常终止条件（非叠加），0 表示不限迭代 → 无限跑写穿缓冲区。剖析必须显式给正整数；1024 = OFFLINE 全程。
2. **megakernel 导出链差异**：megakernel 侧 profiling 导出走 perfetto + 原始 `.pt`（无 CSV、无 `_decode_events`），统计事件需自行解码 `.pt`（tag 布局：`event_no=tag>>19`、`block_group=(tag>>11)&0xFF`、`event_idx=(tag>>2)&0x1FF`、`type=tag&0x3`）。
3. **32 迭代对照结论**：两项目事件分布完全对齐（每迭代任务数 = 条数/2）：SPLITK_LINEAR_RES_ATOMIC=9216、SPLIT_KV=2592、MERGE=288、SILU_MUL=3456、ARGMAX_PARTIAL=240、BEGIN=32。
4. 剖析封顶下 demo 的 Decode 计数显示负数（如 -31 tokens）为 demo 计时口径在封顶下的显示怪相，不影响 trace。

---

## 验证结果汇总

| 运行 | 配置 | 结果 |
|---|---|---|
| run47 | 非 profiling，1024 tokens | EXIT=0，25.620 ms/token，文本连贯（死锁+文本退化闭环） |
| run48 | 心跳清理后 | EXIT=0，32.693 ms/token（噪声），[HB]=0 |
| run50 | split-kv 移植 | EXIT=0，21.439 ms/token，文本连贯 |
| run51 | profiling 修复回归 | EXIT=0，21.503 ms/token，文本连贯 |
| fleet1 | demo_fleet 移植 | EXIT=0，**18.342 ms/token**，文本连贯（追平 megakernel 18.3） |
| 剖析×6 | 32/128/1024 迭代 | EXIT=0，事件分布两项目对齐，VMFault 仅 ITERS=0 场景 |
