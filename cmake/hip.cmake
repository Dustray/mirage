#######################################################
# Locate a HIP/ROCm toolchain (e.g. Hygon DTK) and
# provide the variables needed to build mirage_runtime
# with hipcc (`dcc -x hip`) instead of nvcc.
#
# Usage:
#   include(cmake/hip.cmake)
#   find_hip()
#
# Provide variables:
#
# - HIP_FOUND
# - HIP_EXECUTABLE        (path to hipcc)
# - HIP_TOOLKIT_ROOT_DIR  (prefix containing bin/hipcc, include/hip, lib)
# - HIP_INCLUDE_DIRS
# - HIP_RUNTIME_LIBRARY   (libamdhip64.so)
# - HIPBLAS_LIBRARY       (libhipblas.so)
#
# Build recipe (the compat layer include/mirage_compat/cuda shadows the
# CUDA headers Mirage includes and forwards them to the real HIP headers):
#   . <dtk>/env.sh
#   cmake .. -DUSE_HIP=ON -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
#   make -j mirage_runtime
#
macro(find_hip)
  # --- locate hipcc ---------------------------------------------------------
  find_program(HIP_EXECUTABLE hipcc
    HINTS
      ${ROCM_PATH}
      $ENV{ROCM_PATH}
      $ENV{DTK_ROOT}
      /opt/dtk
      /opt/rocm
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH)
  if(NOT HIP_EXECUTABLE)
    # Fall back to a plain PATH search (e.g. CXX=hipcc already exported).
    find_program(HIP_EXECUTABLE hipcc)
  endif()
  if(NOT HIP_EXECUTABLE)
    message(FATAL_ERROR
      "USE_HIP is ON but hipcc was not found. Source the DTK/ROCm env.sh "
      "(which exports ROCM_PATH and puts hipcc on PATH) or set "
      "-DCMAKE_CXX_COMPILER=<path-to-hipcc> explicitly.")
  endif()
  message(STATUS "Found hipcc: ${HIP_EXECUTABLE}")

  # --- derive the toolkit root from the hipcc location ----------------------
  get_filename_component(_HIP_BIN_DIR "${HIP_EXECUTABLE}" DIRECTORY)
  get_filename_component(HIP_TOOLKIT_ROOT_DIR "${_HIP_BIN_DIR}" DIRECTORY)
  message(STATUS "HIP_TOOLKIT_ROOT_DIR=" ${HIP_TOOLKIT_ROOT_DIR})

  set(HIP_INCLUDE_DIRS ${HIP_TOOLKIT_ROOT_DIR}/include)

  # --- runtime libraries ----------------------------------------------------
  find_library(HIP_RUNTIME_LIBRARY amdhip64
    HINTS ${HIP_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
  find_library(HIPBLAS_LIBRARY hipblas
    HINTS ${HIP_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 hipblas/lib
    NO_DEFAULT_PATH)
  message(STATUS "Found HIP_RUNTIME_LIBRARY=" ${HIP_RUNTIME_LIBRARY})
  message(STATUS "Found HIPBLAS_LIBRARY=" ${HIPBLAS_LIBRARY})

  set(HIP_FOUND TRUE)
endmacro(find_hip)
