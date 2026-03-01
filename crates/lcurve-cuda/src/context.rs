use crate::error::{check_cuda, CudaError};
use std::ptr;

// FFI declarations for CUDA memory management
extern "C" {
    fn lcuda_init(device_id: i32) -> i32;
    fn lcuda_malloc(devptr: *mut *mut u8, size: usize) -> i32;
    fn lcuda_free(devptr: *mut u8) -> i32;
    fn lcuda_memcpy_h2d(dst: *mut u8, src: *const u8, size: usize) -> i32;
    fn lcuda_memcpy_d2h(dst: *mut u8, src: *const u8, size: usize) -> i32;
    fn lcuda_sync() -> i32;
    fn lcuda_device_count() -> i32;

    // Stream management
    fn lcuda_stream_create(stream_out: *mut *mut u8) -> i32;
    fn lcuda_stream_destroy(stream: *mut u8) -> i32;
    fn lcuda_stream_sync(stream: *mut u8) -> i32;
    fn lcuda_memcpy_h2d_async(dst: *mut u8, src: *const u8, size: usize, stream: *mut u8) -> i32;
    fn lcuda_memcpy_d2h_async(dst: *mut u8, src: *const u8, size: usize, stream: *mut u8) -> i32;

    // Pinned host memory
    fn lcuda_host_alloc(ptr: *mut *mut u8, size: usize) -> i32;
    fn lcuda_host_free(ptr: *mut u8) -> i32;
}

/// RAII wrapper for a device pointer. Frees memory on drop.
pub struct DevicePtr {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for DevicePtr {}

impl DevicePtr {
    /// Allocate `size` bytes on the device.
    pub fn alloc(size: usize) -> Result<Self, CudaError> {
        if size == 0 {
            return Ok(DevicePtr {
                ptr: ptr::null_mut(),
                size: 0,
            });
        }
        let mut ptr: *mut u8 = ptr::null_mut();
        check_cuda(unsafe { lcuda_malloc(&mut ptr as *mut *mut u8, size) })?;
        Ok(DevicePtr { ptr, size })
    }

    /// Upload data from host to device (synchronous).
    pub fn upload<T>(&self, data: &[T]) -> Result<(), CudaError> {
        let byte_len = std::mem::size_of_val(data);
        if byte_len == 0 {
            return Ok(());
        }
        if byte_len > self.size {
            return Err(CudaError::Memory(format!(
                "upload: data size {} exceeds allocation {}",
                byte_len, self.size
            )));
        }
        check_cuda(unsafe {
            lcuda_memcpy_h2d(self.ptr, data.as_ptr() as *const u8, byte_len)
        })
    }

    /// Upload data from host to device (async on a stream).
    /// SAFETY: `data` must remain valid until the stream is synchronized.
    pub unsafe fn upload_async<T>(
        &self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let byte_len = std::mem::size_of_val(data);
        if byte_len == 0 {
            return Ok(());
        }
        if byte_len > self.size {
            return Err(CudaError::Memory(format!(
                "upload_async: data size {} exceeds allocation {}",
                byte_len, self.size
            )));
        }
        check_cuda(lcuda_memcpy_h2d_async(
            self.ptr,
            data.as_ptr() as *const u8,
            byte_len,
            stream.ptr,
        ))
    }

    /// Download data from device to host (synchronous).
    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<(), CudaError> {
        let byte_len = std::mem::size_of_val(out);
        if byte_len == 0 {
            return Ok(());
        }
        if byte_len > self.size {
            return Err(CudaError::Memory(format!(
                "download: request size {} exceeds allocation {}",
                byte_len, self.size
            )));
        }
        check_cuda(unsafe {
            lcuda_memcpy_d2h(out.as_mut_ptr() as *mut u8, self.ptr, byte_len)
        })
    }

    /// Download data from device to host (async on a stream).
    /// SAFETY: `out` must remain valid until the stream is synchronized.
    pub unsafe fn download_async<T: Copy>(
        &self,
        out: &mut [T],
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let byte_len = std::mem::size_of_val(out);
        if byte_len == 0 {
            return Ok(());
        }
        if byte_len > self.size {
            return Err(CudaError::Memory(format!(
                "download_async: request size {} exceeds allocation {}",
                byte_len, self.size
            )));
        }
        check_cuda(lcuda_memcpy_d2h_async(
            out.as_mut_ptr() as *mut u8,
            self.ptr,
            byte_len,
            stream.ptr,
        ))
    }

    /// Raw device pointer for passing to kernel launches.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Raw mutable device pointer.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Allocated size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { lcuda_free(self.ptr) };
        }
    }
}

/// RAII wrapper for a CUDA stream.
pub struct CudaStream {
    pub(crate) ptr: *mut u8,
}

unsafe impl Send for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self, CudaError> {
        let mut ptr: *mut u8 = ptr::null_mut();
        check_cuda(unsafe { lcuda_stream_create(&mut ptr as *mut *mut u8) })?;
        Ok(CudaStream { ptr })
    }

    /// Raw stream pointer for passing to FFI functions.
    pub(crate) fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Wait for all operations on this stream to complete.
    pub fn sync(&self) -> Result<(), CudaError> {
        check_cuda(unsafe { lcuda_stream_sync(self.ptr) })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { lcuda_stream_destroy(self.ptr) };
        }
    }
}

/// RAII wrapper for page-locked (pinned) host memory.
/// Required for async H2D/D2H transfers.
pub struct PinnedBuf {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    /// Allocate `size` bytes of pinned host memory.
    pub fn alloc(size: usize) -> Result<Self, CudaError> {
        if size == 0 {
            return Ok(PinnedBuf {
                ptr: ptr::null_mut(),
                size: 0,
            });
        }
        let mut ptr: *mut u8 = ptr::null_mut();
        check_cuda(unsafe { lcuda_host_alloc(&mut ptr as *mut *mut u8, size) })?;
        Ok(PinnedBuf { ptr, size })
    }

    /// Copy `data` into the pinned buffer. Returns a slice of the pinned memory.
    /// Panics if data doesn't fit.
    pub fn stage<T: Copy>(&mut self, data: &[T]) -> &[T] {
        let byte_len = std::mem::size_of_val(data);
        assert!(
            byte_len <= self.size,
            "PinnedBuf::stage: data {} > buf {}",
            byte_len,
            self.size
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.ptr,
                byte_len,
            );
            std::slice::from_raw_parts(self.ptr as *const T, data.len())
        }
    }

    /// Get a mutable slice for download targets.
    pub fn as_mut_slice<T: Copy>(&mut self, count: usize) -> &mut [T] {
        let byte_len = count * std::mem::size_of::<T>();
        assert!(
            byte_len <= self.size,
            "PinnedBuf::as_mut_slice: need {} > buf {}",
            byte_len,
            self.size
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, count) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { lcuda_host_free(self.ptr) };
        }
    }
}

/// Manages the CUDA device state and persistent allocations.
pub struct CudaContext {
    device_id: i32,
}

impl CudaContext {
    /// Initialize a CUDA context on the specified device.
    pub fn new(device_id: i32) -> Result<Self, CudaError> {
        let count = unsafe { lcuda_device_count() };
        if count <= 0 {
            return Err(CudaError::DeviceNotAvailable(
                "No CUDA devices found".into(),
            ));
        }
        if device_id >= count {
            return Err(CudaError::DeviceNotAvailable(format!(
                "Device {} requested but only {} devices available",
                device_id, count
            )));
        }
        check_cuda(unsafe { lcuda_init(device_id) })?;
        Ok(CudaContext { device_id })
    }

    /// Device ID this context is bound to.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Synchronize the device (wait for all kernel launches to complete).
    pub fn sync(&self) -> Result<(), CudaError> {
        check_cuda(unsafe { lcuda_sync() })
    }
}
