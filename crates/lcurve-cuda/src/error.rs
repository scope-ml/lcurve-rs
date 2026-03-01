use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA runtime error (code {code}): {msg}")]
    Runtime { code: i32, msg: String },

    #[error("CUDA device not available: {0}")]
    DeviceNotAvailable(String),

    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("CUDA memory error: {0}")]
    Memory(String),

    #[error("lcurve error: {0}")]
    Lcurve(#[from] lcurve::LcurveError),

    #[error("roche error: {0}")]
    Roche(#[from] lcurve_roche::RocheError),

    #[error("{0}")]
    Generic(String),
}

/// Convert a CUDA error code to a CudaError.
pub fn check_cuda(code: i32) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError::Runtime {
            code,
            msg: format!("cudaError_t = {}", code),
        })
    }
}
