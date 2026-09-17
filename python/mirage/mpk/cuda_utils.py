from cuda.bindings import driver, nvrtc


_cuda_driver_initialized = False


def _ensure_cuda_driver_initialized():
    global _cuda_driver_initialized
    if _cuda_driver_initialized:
        return
    checkCudaErrors(driver.cuInit(0))
    _cuda_driver_initialized = True


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def _queryMulticastSupport(cu_device):
    _ensure_cuda_driver_initialized()
    # Query multicast object support
    multicast_supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        cu_device
    ))
    return bool(multicast_supported)

def queryMulticastSupport(device_id):
    _ensure_cuda_driver_initialized()
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryMulticastSupport(cu_device)

def _queryVMMsupport(cu_device):
    _ensure_cuda_driver_initialized()
    # Query virtual memory management support
    vmm_supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        cu_device
    ))
    return bool(vmm_supported)

def queryVMMsupport(device_id):
    _ensure_cuda_driver_initialized()
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryVMMsupport(cu_device)

def _queryPeerAccessSupported(cu_device_from, cu_device_to):
    _ensure_cuda_driver_initialized()
    # Query peer access support
    peer_access_supported = checkCudaErrors(driver.cuDeviceCanAccessPeer(
        cu_device_from,
        cu_device_to
    ))
    return bool(peer_access_supported)

def queryPeerAccessSupported(device_id_from, device_id_to):
    _ensure_cuda_driver_initialized()
    cu_device_from = checkCudaErrors(driver.cuDeviceGet(device_id_from))
    cu_device_to = checkCudaErrors(driver.cuDeviceGet(device_id_to))
    return _queryPeerAccessSupported(cu_device_from, cu_device_to)

def _queryHandleTypePosixFileDescriptorSupported(cu_device):
    _ensure_cuda_driver_initialized()
    # Query POSIX shared memory support
    supported = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        cu_device
    ))
    return bool(supported)

def queryHandleTypePosixFileDescriptorSupported(device_id):
    _ensure_cuda_driver_initialized()
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    return _queryHandleTypePosixFileDescriptorSupported(cu_device)

def queryComputeCapability(device_id):
    _ensure_cuda_driver_initialized()
    cu_device = checkCudaErrors(driver.cuDeviceGet(device_id))
    major = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cu_device
    ))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cu_device
    ))
    return (major, minor)

# ==== MIRAGE HIP COMPAT: pure addition, upstream code above is untouched ====
# On HIP/ROCm toolchains (e.g. Hygon DTK) the `cuda-python` package
# (`cuda.bindings`) is unavailable, so fall back to a ctypes-based shim of the
# small CUDA driver API surface used above. Re-defining the helpers and query
# functions at module level keeps callers unchanged; on NVIDIA builds the
# upstream cuda-python implementation above is kept as-is.
try:
    import cuda.bindings  # noqa: F401
except ImportError:
    import ctypes
    import os

    # Locate the DTK/CUDA driver library. The env.sh sets CUDA_PATH to the
    # cuda-12 target directory, so prefer that and fall back to the known
    # DTK installation path.
    _CUDA_LIB_PATHS = [
        os.path.join(os.environ.get("CUDA_PATH", ""), "lib64", "libcuda.so"),
        os.path.join(os.environ.get("CUDA_PATH", ""), "lib", "libcuda.so"),
        "/opt/dtk/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so",
        "/opt/dtk/cuda/cuda/lib64/libcuda.so",
        "libcuda.so",
    ]

    _libcuda = None
    for _p in _CUDA_LIB_PATHS:
        if _p and os.path.exists(_p):
            _libcuda = ctypes.CDLL(_p)
            break

    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so")
        except OSError as _e:
            raise RuntimeError(
                "Cannot load libcuda.so. Please source /opt/dtk/cuda/env.sh "
                "or set CUDA_PATH to the DTK cuda target directory."
            ) from _e


    CUresult = ctypes.c_int

    # CUdevice_attribute values from DTK's cuda.h
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132

    CUDA_SUCCESS = 0

    _libcuda.cuInit.argtypes = [ctypes.c_uint]
    _libcuda.cuInit.restype = CUresult

    _libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _libcuda.cuDeviceGet.restype = CUresult

    _libcuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    _libcuda.cuDeviceGetAttribute.restype = CUresult

    _libcuda.cuDeviceCanAccessPeer.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    _libcuda.cuDeviceCanAccessPeer.restype = CUresult

    _libcuda.cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    _libcuda.cuGetErrorName.restype = CUresult

    _libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    _libcuda.cuGetErrorString.restype = CUresult


    _cuda_driver_initialized = False


    def _ensure_cuda_driver_initialized():
        global _cuda_driver_initialized
        if _cuda_driver_initialized:
            return
        checkCudaErrors(_libcuda.cuInit(0))
        _cuda_driver_initialized = True


    def _cudaGetErrorEnum(error):
        if isinstance(error, int):
            err = error
            name_p = ctypes.c_char_p()
            name_err = _libcuda.cuGetErrorName(err, ctypes.byref(name_p))
            if name_err == CUDA_SUCCESS and name_p.value:
                return name_p.value.decode("utf-8", errors="replace")
            return "<unknown>"
        else:
            raise RuntimeError('Unknown error type: {}'.format(error))


    def checkCudaErrors(result):
        if isinstance(result, tuple):
            err = result[0]
        else:
            err = result
        if err != CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(err, _cudaGetErrorEnum(err)))
        if isinstance(result, tuple):
            if len(result) == 1:
                return None
            elif len(result) == 2:
                return result[1]
            else:
                return result[1:]
        return None


    def _queryMulticastSupport(cu_device):
        _ensure_cuda_driver_initialized()
        multicast_supported = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGetAttribute(
            ctypes.byref(multicast_supported),
            CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            cu_device,
        ))
        return bool(multicast_supported.value)


    def queryMulticastSupport(device_id):
        _ensure_cuda_driver_initialized()
        cu_device = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device), device_id))
        return _queryMulticastSupport(cu_device.value)


    def _queryVMMsupport(cu_device):
        _ensure_cuda_driver_initialized()
        vmm_supported = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGetAttribute(
            ctypes.byref(vmm_supported),
            CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            cu_device,
        ))
        return bool(vmm_supported.value)


    def queryVMMsupport(device_id):
        _ensure_cuda_driver_initialized()
        cu_device = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device), device_id))
        return _queryVMMsupport(cu_device.value)


    def _queryPeerAccessSupported(cu_device_from, cu_device_to):
        _ensure_cuda_driver_initialized()
        peer_access_supported = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceCanAccessPeer(
            ctypes.byref(peer_access_supported),
            cu_device_from,
            cu_device_to,
        ))
        return bool(peer_access_supported.value)


    def queryPeerAccessSupported(device_id_from, device_id_to):
        _ensure_cuda_driver_initialized()
        cu_device_from = ctypes.c_int()
        cu_device_to = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device_from), device_id_from))
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device_to), device_id_to))
        return _queryPeerAccessSupported(cu_device_from.value, cu_device_to.value)


    def _queryHandleTypePosixFileDescriptorSupported(cu_device):
        _ensure_cuda_driver_initialized()
        supported = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGetAttribute(
            ctypes.byref(supported),
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
            cu_device,
        ))
        return bool(supported.value)


    def queryHandleTypePosixFileDescriptorSupported(device_id):
        _ensure_cuda_driver_initialized()
        cu_device = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device), device_id))
        return _queryHandleTypePosixFileDescriptorSupported(cu_device.value)


    def queryComputeCapability(device_id):
        _ensure_cuda_driver_initialized()
        cu_device = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGet(ctypes.byref(cu_device), device_id))
        major = ctypes.c_int()
        minor = ctypes.c_int()
        checkCudaErrors(_libcuda.cuDeviceGetAttribute(
            ctypes.byref(major),
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cu_device.value,
        ))
        checkCudaErrors(_libcuda.cuDeviceGetAttribute(
            ctypes.byref(minor),
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cu_device.value,
        ))
        return (major.value, minor.value)
# ==== end MIRAGE HIP COMPAT ====
