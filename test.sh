#!/bin/bash
# MIRAGE HIP COMPAT 测试脚本（Qwen3-8B @ gfx936）
#
# 用法:
#   ./test.sh                              # demo_fleet.py 1024 tokens（性能+正确性基线）
#   ./test.sh demo.py                      # mirage 原版 demo（无 splitk linear 优化）
#   ./test.sh demo_fleet.py 128            # 自定义生成 token 数
#   PROF_ITERS=32 ./test.sh                # 剖析模式（32 迭代封顶），产出 trace + csv
#
# 剖析说明:
# - MPK_PROFILING_ITERS=32/128/1024 封顶跑 N 个迭代；0 或不设 = 非剖析
# - 剖析封顶下无文本输出属预期（封顶步数 < prompt 区）
# - 注意: MPK_PROFILING_ITERS=0 走 -DMPK_PROFILING_NUM_ITERS=0 时 PROFILING 分支
#   会无限跑写穿 profiler 缓冲区（GPU VMFault），剖析必须显式给正整数；
#   1024 = OFFLINE 全程（max_seq_length 1088 - prompt 64）
# - demo.py 的 profiler 缓冲区未放大（32 迭代安全）；demo_fleet.py 已放大 8 倍
#   （全程 1024 迭代安全）
# - 验证点: EXIT=0、token 数、per-token 延迟、文本连贯
set -euo pipefail
cd "$(dirname "$0")"

DEMO=${1:-demo_fleet.py}
MAX_NEW=${2:-1024}
PROF_ITERS=${PROF_ITERS:-0}

MODEL=/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3-8B/snapshots/master
PROMPT="The history of artificial intelligence spans decades of research and innovation across many disciplines including computer science mathematics neuroscience and cognitive psychology from early symbolic systems through modern deep learning and large language models "

source /opt/dtk/cuda/env.sh
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/z3/lib:${LD_LIBRARY_PATH:-}
export USE_GANG=0 USE_CK_FMHA=0 AMDGPU_TARGETS=gfx936 MIRAGE_USE_HIP=1
export HIP_ALLOC_INITIALIZE=${HIP_ALLOC_INITIALIZE:-0}
export PYTHONPATH=$(pwd)/python
export MIRAGE_HOME=$(pwd)

PROF_ARGS=""
if [ "$PROF_ITERS" -gt 0 ]; then
  export MPK_PROFILING_ITERS=$PROF_ITERS
  PROF_ARGS="--profiling --trace-name /tmp/$(basename "$DEMO" .py)_prof_${PROF_ITERS}"
  MAX_NEW=128
  echo "==> 剖析模式: $PROF_ITERS 迭代封顶，trace: /tmp/$(basename "$DEMO" .py)_prof_${PROF_ITERS}.perfetto-trace"
fi

echo "==> demo=$DEMO max_new_tokens=$MAX_NEW"
cd demo/qwen3
python -u "$DEMO" \
  --use-mirage --model "$MODEL" \
  --max-num-batched-requests 1 --max-num-batched-tokens 1 \
  --max-new-tokens "$MAX_NEW" --max-seq-length 1088 \
  --max-num-pages 16 --page-size 4096 \
  --ignore-eos --temperature 0.0 \
  --split-kv-cache $PROF_ARGS \
  --prompt "$PROMPT"
