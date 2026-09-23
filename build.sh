#!/bin/bash
# MIRAGE HIP COMPAT 构建脚本（gfx936 / DTK）
#
# 用法:
#   ./build.sh                 # HIP 构建（默认）
#   MIRAGE_USE_HIP=0 ./build.sh   # 回退 CUDA 构建路径
#
# 说明:
# - HIP 是默认构建路径（setup.py _mirage_hip_build 默认 True；CMakeLists USE_HIP ON）
# - 改动 src/kernel/*.cc 后必须强制重链：distutils 重链陷阱——setup.py 把
#   mirage_runtime 放在 libraries 而非 sources，build_ext 不追踪 libmirage_runtime.a
#   的变化，会拷回陈旧链接产物；touch 全部 .pyx 强制走完整重编重链
# - 验证 .so 是否含新代码用 readelf -p .rodata（strings 会假阳性：
#   .debug_str 含 DWARF 枚举名）
set -euo pipefail
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/z3/lib:${LD_LIBRARY_PATH:-}
export MIRAGE_USE_HIP=${MIRAGE_USE_HIP:-1}
export AMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx936}

echo "==> MIRAGE_USE_HIP=$MIRAGE_USE_HIP AMDGPU_TARGETS=$AMDGPU_TARGETS"

find python -name "*.pyx" -exec touch {} +
python setup.py build_ext --inplace "$@"

SO=$(ls -t python/mirage/core.cpython-*-linux-gnu.so | head -1)
echo "==> BUILD OK: $SO"
echo "==> 验证示例: readelf -p .rodata $SO | grep <新符号名>"
