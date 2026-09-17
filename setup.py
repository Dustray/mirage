# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import shutil
from os import path
from pathlib import Path
import sys
import sysconfig
from setuptools import find_packages, setup, Command
from contextlib import contextmanager
import subprocess
import re

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

import z3

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib", "stubs"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

z3_path = path.dirname(z3.__file__)
# ==== MIRAGE HIP COMPAT: 默认走 HIP 构建，MIRAGE_USE_HIP=0 退回 CUDA ====
_mirage_hip_build = os.environ.get("MIRAGE_USE_HIP", "1") != "0"
# ==== end MIRAGE HIP COMPAT ====

# Use version.py to get package version
version_file = os.path.join(os.path.dirname(__file__), "python/mirage/version.py")
with open(version_file, "r") as f:
    exec(f.read())  # This will define __version__

def get_backend_macros(config_file):
    flags = {
        "USE_CUDA":   None,
        "USE_NKI":    None,
    }

    pattern = re.compile(r'^\s*set\s*\(\s*(USE_CUDA|USE_NKI)\s+(ON|OFF)\s*\)', re.IGNORECASE)
    
    with open(config_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                var, val = match.groups()
                flags[var] = (val.upper() == "ON")
    
    macros = []
    if flags.get("USE_CUDA"):
        macros.append(("MIRAGE_BACKEND_USE_CUDA", None))
        macros.append(("MIRAGE_FINGERPRINT_USE_CUDA", None))
    elif flags.get("USE_NKI"):
        macros.append(("MIRAGE_BACKEND_USE_NKI", None))
        macros.append(("MIRAGE_FINGERPRINT_USE_CPU", None))
    else:
        raise KeyError("Please select either USE_CUDA or USE_NKI in config.cmake file")
    return macros

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize

        ret = []
        mirage_path = ''
        config_path = path.join(mirage_path, "config.cmake")
        macros = get_backend_macros(config_path)
        cython_path = path.join(mirage_path, "python/mirage/_cython")
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "mirage.%s" % fn[:-4],
                    ["%s/%s" % (cython_path, fn)],
                    include_dirs=[
                        path.join(mirage_path, "include"),
                        path.join(mirage_path, "include", "mirage_compat"),
                        path.join(mirage_path, "deps", "json", "include"),
                        path.join(mirage_path, "deps", "cutlass", "include"),
                        path.join(mirage_path, "deps", "cutlass", "tools", "util", "include"),
                        path.join(mirage_path, "build", "abstract_subexpr", "release"),
                        path.join(mirage_path, "build", "formal_verifier", "release"),
                        path.join(z3_path, "include"),
                        cuda_include_dir,
                    ],
                    libraries=[
                        "mirage_runtime",
                        "cudadevrt",
                        "cudart_static",
                        "cudart",
                        "cuda",
                        "z3",
                        "gomp",
                        "rt",
                        "abstract_subexpr",
                        "formal_verifier",
                    ],
                    library_dirs=[
                        path.join(mirage_path, "build"),
                        path.join(z3_path, "lib"),
                        path.join(mirage_path, "build", "abstract_subexpr", "release"),
                        path.join(mirage_path, "build", "formal_verifier", "release"),
                    ]
                    + cuda_library_dirs,
                    define_macros=macros,
                    extra_compile_args=["-std=c++17", "-fopenmp"],
                    extra_link_args=[
                        "-fPIC",
                        "-fopenmp",
                        "-lrt",
                        f"-Wl,-rpath,{path.join('$ORIGIN', 'lib')}",
                        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'abstract_subexpr', 'release')}",
                        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'formal_verifier', 'release')}",
                    ],
                    language="c++",
                )
            )
        # ==== MIRAGE HIP COMPAT: pure addition, upstream line below is untouched ====
        if _mirage_hip_build:
            # The compat CUDA headers must shadow every other include dir:
            # mirage headers include <vector_types.h> and friends, which only
            # exist in the compat layer on HIP toolchains. The static
            # mirage_runtime archive also references HIP symbols, so the
            # cython extensions link the HIP runtime libraries instead of
            # the CUDA ones.
            _hip_compat_inc = path.join(
                mirage_path, "include", "mirage_compat", "cuda")
            _hip_toolkit = (
                os.environ.get("ROCM_PATH")
                or os.environ.get("DTK_ROOT")
                or "/opt/dtk"
            )
            _hip_extra_incs = [
                _hip_compat_inc,
                path.join(_hip_toolkit, "hip", "include"),
                path.join(_hip_toolkit, "include"),
            ]
            _hip_lib_dir = path.join(_hip_toolkit, "lib")
            # Copy the z3 runtime library next to the built extensions so the
            # upstream "$ORIGIN/lib" rpath resolves it: under pip build
            # isolation z3_path points into an ephemeral overlay, so an
            # rpath to it would go stale right after the build.
            _z3_lib_src = path.join(z3_path, "lib")
            _z3_lib_dst = path.join(mirage_path, "python", "mirage", "lib")
            if path.isdir(_z3_lib_src):
                os.makedirs(_z3_lib_dst, exist_ok=True)
                for _z3_so in os.listdir(_z3_lib_src):
                    if _z3_so.startswith("libz3.so"):
                        shutil.copy2(
                            path.join(_z3_lib_src, _z3_so),
                            path.join(_z3_lib_dst, _z3_so),
                        )
            for _ext in ret:
                _ext.include_dirs = _hip_extra_incs + [
                    _d for _d in _ext.include_dirs
                    if _d not in _hip_extra_incs
                ]
                _ext.extra_compile_args = _ext.extra_compile_args + [
                    # Matches the CMake HIP build: platform macro for the
                    # HIP headers and the warp-sync declaration gate that
                    # DTK's clang pipeline leaves undefined. The forced
                    # include mirrors nvcc's implicit <cuda_runtime.h>.
                    "-D__HIP_PLATFORM_AMD__=1",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                    "-include",
                    "cuda_runtime.h",
                ]
                _ext.libraries = ["amdhip64", "hipblas"] + [
                    _lib for _lib in _ext.libraries
                    if _lib not in ("cudadevrt", "cudart_static",
                                    "cudart", "cuda")
                ]
                _ext.extra_link_args = _ext.extra_link_args + [
                    # Upstream headers define some CUTLASS_HOST_DEVICE
                    # helpers (e.g. the deserialize_*_op_parameters in
                    # threadblock/serializer/*.h and get_reduction_dim in
                    # utils/cuda_helper.h) out-of-line, so several archive
                    # members carry the same definition. The nvcc build never
                    # extracts those members together; the HIP build's symbol
                    # demand does. All duplicates come from the same header,
                    # so keeping the linker's first definition is safe.
                    "-Wl,--allow-multiple-definition",
                    # -lz3 resolves against the z3 wheel's private lib dir,
                    # whose SONAME (libz3.so.4.16) is not on the default
                    # loader path; add an rpath so import works without
                    # LD_LIBRARY_PATH.
                    "-Wl,-rpath,%s" % path.join(z3_path, "lib"),
                    # The HIP archive was compiled by clang with -fopenmp,
                    # which emits Intel-runtime (__kmpc_*) symbols resolved by
                    # libomp (bundled with the DTK compiler in dcc/lib). The
                    # g++ link step below would otherwise resolve OpenMP via
                    # libgomp and leave those symbols undefined.
                    "-L%s" % path.join(_hip_toolkit, "dcc", "lib"),
                    "-lomp",
                    "-Wl,-rpath,%s" % path.join(_hip_toolkit, "dcc", "lib"),
                    # Ubuntu 的 gcc 默认 --as-needed：出现在 mirage_runtime.a
                    # 之前的 HIP 共享库（当时还没有符号引用）会被丢弃，导致
                    # hipFree 等符号未解析；在静态库之后再链一次。
                    "-Wl,--no-as-needed",
                    "-lamdhip64",
                    "-lhipblas",
                    "-Wl,-rpath,%s" % _hip_lib_dir,
                ]
                _ext.library_dirs = [_hip_lib_dir] + [
                    _d for _d in _ext.library_dirs
                    if _d != _hip_lib_dir
                ]
        # ==== end MIRAGE HIP COMPAT ====
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
    
if os.environ.get("MIRAGE_SKIP_NATIVE_BUILD") != "1":
    # Install Rust if not yet available
    try:
        # Attempt to run a Rust command to check if Rust is installed
        subprocess.check_output(['cargo', '--version'])
    except FileNotFoundError:
        print("Rust/Cargo not found, installing it...")
        # Rust is not installed, so install it using rustup
        try:
            subprocess.run("curl https://sh.rustup.rs -sSf | sh -s -- -y", shell=True, check=True)
            print("Rust and Cargo installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        # Add the cargo binary directory to the PATH
        os.environ["PATH"] = f"{os.path.join(os.environ.get('HOME', '/root'), '.cargo', 'bin')}:{os.environ.get('PATH', '')}"

    mirage_path = path.dirname(__file__)
    # z3_path = os.path.join(mirage_path, 'deps', 'z3', 'build')
    # os.environ['Z3_DIR'] = z3_path
    if mirage_path == '':
        mirage_path = '.'

    try:
        subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr')
    except subprocess.CalledProcessError as e:
        print("Failed to build abstract_subexpr Rust library, building it ...")
        try:
            subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr', check=True)
            print("Abstract_subexpr Rust library built successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to build abstract_subexpr Rust library.")
        os.environ['ABSTRACT_SUBEXPR_LIB'] = os.path.join(mirage_path,'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so')

    try:
        subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv')
    except subprocess.CalledProcessError as e:
        print("Failed to build formal_verifier Rust library, building it ...")
        try:
            subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv', check=True)
            print("formal_verifier Rust library built successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to build formal_verifier Rust library.")
        os.environ['FORMAL_VERIFIER_LIB'] = os.path.join(mirage_path,'build', 'formal_verifier', 'release', 'libformal_verifier.so')


    # build Mirage runtime library
    try:
        os.environ["CUDACXX"] = nvcc_path if nvcc_path else os.path.join(
            cuda_home, "bin", "nvcc"
        )
        mirage_path = path.dirname(__file__)
        # z3_path = os.path.join(mirage_path, 'deps', 'z3', 'build')
        # os.environ['Z3_DIR'] = z3_path
        if mirage_path == "":
            mirage_path = "."
        os.makedirs(mirage_path, exist_ok=True)
        os.chdir(mirage_path)
        build_dir = os.path.join(mirage_path, "build")

        cc_path = os.environ.get("CC") or shutil.which("gcc") or "/usr/bin/gcc"
        os.environ["CC"] = cc_path
        cxx_path = os.environ.get("CXX") or shutil.which("g++") or "/usr/bin/g++"
        os.environ["CXX"] = cxx_path
        print(f"CC: {os.environ['CC']}, CXX: {os.environ['CXX']}", flush=True)

        # Create the build directory if it does not exist
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(
            [
                "cmake",
                "..",
                "-DCMAKE_BUILD_TYPE=" + os.environ.get("CMAKE_BUILD_TYPE", "Release"),
                "-DZ3_CXX_INCLUDE_DIRS=" + z3_path + "/include/",
                "-DZ3_LIBRARIES=" + path.join(z3_path, "lib", "libz3.so"),
                '-DABSTRACT_SUBEXPR_LIB=' + path.join(mirage_path, 'build', 'abstract_subexpr', 'release'),
                '-DABSTRACT_SUBEXPR_LIBRARIES=' + path.join(mirage_path, 'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so'),
                '-DFORMAL_VERIFIER_LIB=' + path.join(mirage_path, 'build', 'formal_verifier', 'release'),
                '-DFORMAL_VERIFIER_LIBRARIES=' + path.join(mirage_path, 'build', 'formal_verifier', 'release', 'libformal_verifier.so'),
                "-DCMAKE_C_COMPILER=" + os.environ["CC"],
                "-DCMAKE_CXX_COMPILER=" + os.environ["CXX"],
            ],
            cwd=build_dir,
            env=os.environ.copy(),
        )
        subprocess.check_call(["make", "-j8"], cwd=build_dir, env=os.environ.copy())
        print("Mirage runtime library built successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to build runtime library.")
        raise SystemExit(e.returncode)
else:
    # Pre-built: just set mirage_path for config_cython()
    mirage_path = path.dirname(__file__) or "."

from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    """Copy native .so files into mirage/lib/ before the standard build_py runs."""
    def run(self):
        lib_dir = path.join("python", "mirage", "lib")
        os.makedirs(lib_dir, exist_ok=True)
        for so_path in [
            path.join("build", "abstract_subexpr", "release", "libabstract_subexpr.so"),
            path.join("build", "formal_verifier", "release", "libformal_verifier.so"),
        ]:
            if path.exists(so_path):
                shutil.copy2(so_path, lib_dir)
        super().run()

setup_args = {}

# Create requirements list from requirements.txt
with open(Path(__file__).parent / "requirements.txt", "r") as reqs_file:
    requirements = reqs_file.read().strip().split("\n")
print(f"Requirements: {requirements}")

INCLUDE_BASE = "python/mirage/include"


@contextmanager
def copy_include():
    if not path.exists(INCLUDE_BASE):
        src_dirs = [
            "include/mirage_compat",
            "deps/cutlass/include",
            "deps/json/include",
        ]
        for src_dir in src_dirs:
            shutil.copytree(src_dir, path.join(INCLUDE_BASE, src_dir))
        # copy mirage/transpiler/runtime/*
        # to python/mirage/include/mirage/transpiler/runtime/*
        # instead of python/mirage/include/include/mirage/transpiler/runtime/*
        include_mirage_dirs = [
            "include/mirage/transpiler/runtime",
            "include/mirage/triton_transpiler/runtime",
            "include/mirage/persistent_kernel",
        ]
        include_mirage_dsts = [
            path.join(INCLUDE_BASE, "mirage/transpiler/runtime"),
            path.join(INCLUDE_BASE, "mirage/triton_transpiler/runtime"),
            path.join(INCLUDE_BASE, "mirage/persistent_kernel"),
        ]
        for include_mirage_dir, include_mirage_dst in zip(
            include_mirage_dirs, include_mirage_dsts
        ):
            shutil.copytree(include_mirage_dir, include_mirage_dst)

        config_h_src = path.join(
            mirage_path, "include/mirage/config.h"
        )  # Needed by transpiler/runtime/threadblock/utils.h
        config_h_dst = path.join(INCLUDE_BASE, "mirage/config.h")
        shutil.copy(config_h_src, config_h_dst)
        yield True
    else:
        yield False
    shutil.rmtree(INCLUDE_BASE)


with copy_include() as copied:
    if not copied:
        print(
            "WARNING: include directory already exists. Not copying again. "
            f"This may cause issues. Please remove {INCLUDE_BASE} and rerun setup.py",
            flush=True,
        )

    setup(
        name="mirage-project",
        version=__version__,
        description="Mirage: A Multi-Level Superoptimizer for Tensor Algebra",
        zip_safe=False,
        install_requires=requirements,
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        package_data={"mirage": ["lib/*.so"]},
        cmdclass={"build_py": build_py},
        url="https://github.com/mirage-project/mirage",
        ext_modules=config_cython(),
        include_package_data=True,
        # **setup_args,
    )
