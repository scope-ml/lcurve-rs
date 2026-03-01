pub mod error;
pub mod context;
pub mod transfer;
pub mod batch;

pub use error::CudaError;
pub use context::CudaContext;
pub use batch::chisq_batch_gpu;

// FFI declarations for kernel launch
extern "C" {
    pub(crate) fn lcuda_flux_eval(
        d_data: *const u8,
        d_fudge: *const u8,
        d_points: *const u8,
        d_params: *const u8,
        d_out_flux: *mut u8,
        n_times: i32,
    ) -> i32;

    pub(crate) fn lcuda_flux_eval_stream(
        d_data: *const u8,
        d_fudge: *const u8,
        d_points: *const u8,
        d_params: *const u8,
        d_out_flux: *mut u8,
        n_times: i32,
        stream: *mut u8,
    ) -> i32;
}
