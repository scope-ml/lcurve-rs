/**
 * flux_kernel.cu — CUDA kernel for lcurve flux evaluation.
 *
 * Design: one thread block per time point, threads within a block split the
 * grid-point loop and warp-reduce partial sums.
 *
 * Host-callable C wrappers (lcuda_*) provide the FFI interface to Rust.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Data structures — must match Rust repr(C) definitions exactly      */
/* ------------------------------------------------------------------ */

struct GpuPoint {
    float dirn_x, dirn_y, dirn_z;
    float posn_x, posn_y, posn_z;
    float flux;
    int   n_ecl;
    float ecl_ing[3];
    float ecl_eg[3];
};

struct GpuFluxParams {
    double cosi, sini;
    double xcofm, vfac;

    double ldc1_1, ldc1_2, ldc1_3, ldc1_4;
    double mucrit1;
    int    ltype1;

    double ldc2_1, ldc2_2, ldc2_3, ldc2_4;
    double mucrit2;
    int    ltype2;

    double beam_factor1, beam_factor2;
    double spin1, spin2;

    double lin_limb_disc, quad_limb_disc;

    int    has_glens;
    double rlens1;

    double gint_phase1, gint_phase2;
    double gint_scale11, gint_scale12;
    double gint_scale21, gint_scale22;

    int n_star1f, n_star1c, n_star2f, n_star2c;
    int n_disc, n_edge, n_spot;
    int _pad[1];
};

struct GpuDatum {
    double phase;
    double expose;
    double flux_obs;
    double ferr;
    double weight;
    int    ndiv;
    int    _pad;
};

struct GpuFudge {
    double slfac;
    double third;
};

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TWOPI (2.0 * M_PI)

/* Threads per block for flux kernel */
#define BLOCK_SIZE 256

/* ------------------------------------------------------------------ */
/*  Device helper functions                                            */
/* ------------------------------------------------------------------ */

/**
 * Check if a surface element is visible (not eclipsed) at the given phase.
 */
__device__ __forceinline__
int pt_visible(const GpuPoint* pt, float phase) {
    float phi = phase - floorf(phase);
    for (int i = 0; i < pt->n_ecl; i++) {
        float ing = pt->ecl_ing[i];
        float eg  = pt->ecl_eg[i];
        if ((phi >= ing && phi <= eg) || phi <= eg - 1.0f) {
            return 0;
        }
    }
    return 1;
}

/**
 * Compute limb-darkened intensity I(mu).
 * LDC evaluation in f64 for precision.
 * ltype: 0 = Poly, 1 = Claret
 */
__device__ __forceinline__
double ldc_imu(double mu, double a1, double a2, double a3, double a4, int ltype) {
    if (mu <= 0.0) return 0.0;
    if (mu > 1.0) mu = 1.0;

    double im = 1.0;
    if (ltype == 0) {
        /* Poly: I(mu) = 1 - sum_i a_i (1-mu)^i */
        double ommu = 1.0 - mu;
        im -= ommu * (a1 + ommu * (a2 + ommu * (a3 + ommu * a4)));
    } else {
        /* Claret: I(mu) = 1 - sum_i a_i (1 - mu^(i/2)) */
        im -= a1 + a2 + a3 + a4;
        double msq = sqrt(mu);
        im += msq * (a1 + msq * (a2 + msq * (a3 + msq * a4)));
    }
    return im;
}

/**
 * Compute set_earth direction vector from phase.
 * earth = (sini*cos(phi), -sini*sin(phi), cosi)
 */
__device__ __forceinline__
void set_earth(double cosi, double sini, double phase,
               double* ex, double* ey, double* ez) {
    double phi = TWOPI * phase;
    double sp, cp;
    sincos(phi, &sp, &cp);
    *ex = sini * cp;
    *ey = -sini * sp;
    *ez = cosi;
}

/**
 * Compute gravitational lensing magnification.
 */
__device__ __forceinline__
double lensing_mag(float px, float py, float pz,
                   double ex, double ey, double ez, double rlens1) {
    double sx = (double)px, sy = (double)py, sz = (double)pz;
    double d = -(sx * ex + sy * ey + sz * ez);
    if (d <= 0.0) return 1.0;

    double rx = sx + d * ex;
    double ry = sy + d * ey;
    double rz = sz + d * ez;
    double p = sqrt(rx*rx + ry*ry + rz*rz);
    double ph = p * 0.5;
    double phsq = ph * ph;
    double rd = rlens1 * d;
    double pd;
    if (phsq > 25.0 * rd) {
        pd = p + rd / p;
    } else {
        pd = ph + sqrt(phsq + rd);
    }
    return pd * pd / (pd - ph) / ph / 4.0;
}

/**
 * Compute exposure sub-division phase and weight.
 */
__device__ __forceinline__
void exposure_weight(double phase, double expose, int nd, int ndiv,
                     double* phi_out, double* wgt_out) {
    if (ndiv == 1) {
        *phi_out = phase;
        *wgt_out = 1.0;
    } else {
        *phi_out = phase + expose * ((double)nd - (double)(ndiv - 1) / 2.0) / (double)(ndiv - 1);
        *wgt_out = (nd == 0 || nd == ndiv - 1) ? 0.5 : 1.0;
    }
}

/**
 * Grid type selection (matches Ginterp::grid_type).
 * 1 = fine star1, coarse star2
 * 2 = coarse both
 * 3 = coarse star1, fine star2
 */
__device__ __forceinline__
int grid_type(double phase, double phase1, double phase2) {
    double pnorm = phase - floor(phase);
    if (pnorm <= phase1 || pnorm >= 1.0 - phase1) {
        return 1;
    } else if ((pnorm > phase1 && pnorm < phase2) ||
               (pnorm > 1.0 - phase2 && pnorm < 1.0 - phase1)) {
        return 2;
    } else {
        return 3;
    }
}

/**
 * Ginterp scale factor for star 1.
 */
__device__ __forceinline__
double gint_scale1(double phase, double phase1, double scale11, double scale12) {
    double pnorm = phase - floor(phase);
    if (pnorm <= phase1 || pnorm >= 1.0 - phase1) {
        return 1.0;
    } else {
        return (scale11 * (1.0 - phase1 - pnorm) + scale12 * (pnorm - phase1))
               / (1.0 - 2.0 * phase1);
    }
}

/**
 * Ginterp scale factor for star 2.
 */
__device__ __forceinline__
double gint_scale2(double phase, double phase2, double scale21, double scale22) {
    double pnorm = phase - floor(phase);
    if (pnorm >= phase2 && pnorm <= 1.0 - phase2) {
        return 1.0;
    } else if (pnorm < 0.5) {
        return (scale22 * (phase2 - pnorm) + scale21 * (pnorm + phase2))
               / (2.0 * phase2);
    } else {
        return (scale21 * (1.0 + phase2 - pnorm) + scale22 * (pnorm - 1.0 + phase2))
               / (2.0 * phase2);
    }
}

/* ------------------------------------------------------------------ */
/*  Warp and block reduction                                           */
/* ------------------------------------------------------------------ */

__device__ __forceinline__
double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__
double block_reduce_sum(double val) {
    __shared__ double shared[32];  /* one slot per warp */

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

/* ------------------------------------------------------------------ */
/*  Flux accumulation for a component over grid points                 */
/* ------------------------------------------------------------------ */

/**
 * Accumulate star flux with optional beam correction and lensing.
 * Returns partial sum for this thread (must be block-reduced by caller).
 */
__device__
double accumulate_star(
    const GpuPoint* __restrict__ pts, int n_pts,
    double ex, double ey, double ez, float phase_f,
    double a1, double a2, double a3, double a4, double mucrit, int ltype,
    double beam_factor, double spin, double xcofm, double vfac,
    int is_star2, int has_glens, double rlens1)
{
    double partial = 0.0;
    for (int idx = threadIdx.x; idx < n_pts; idx += blockDim.x) {
        const GpuPoint* pt = &pts[idx];
        if (!pt_visible(pt, phase_f)) continue;

        double mu = ex * (double)pt->dirn_x
                  + ey * (double)pt->dirn_y
                  + ez * (double)pt->dirn_z;
        if (mu <= mucrit) continue;

        double mag = 1.0;
        if (is_star2 && has_glens) {
            mag = lensing_mag(pt->posn_x, pt->posn_y, pt->posn_z,
                              ex, ey, ez, rlens1);
        }

        double fpt = (double)pt->flux;
        if (beam_factor != 0.0) {
            double px = is_star2 ? (double)pt->posn_x - 1.0 : (double)pt->posn_x;
            double py = (double)pt->posn_y;
            double cofm = is_star2 ? (1.0 - xcofm) : xcofm;
            double vx = -vfac * spin * py;
            double vy = vfac * (spin * px + (is_star2 ? (1.0 - cofm) : -cofm));

            /* For star2: vy = vfac * (spin2 * (posn.x - 1.0) + 1.0 - xcofm) */
            if (is_star2) {
                vx = -vfac * spin * py;
                vy = vfac * (spin * ((double)pt->posn_x - 1.0) + 1.0 - xcofm);
            } else {
                vx = -vfac * spin * py;
                vy = vfac * (spin * (double)pt->posn_x - xcofm);
            }

            double vr = -(ex * vx + ey * vy);
            double vn = (double)pt->dirn_x * vx + (double)pt->dirn_y * vy;
            double mud = mu - mu * vr - vn;
            partial += mu * mag * fpt * (1.0 - beam_factor * vr) * ldc_imu(mud, a1, a2, a3, a4, ltype);
        } else {
            partial += mu * mag * fpt * ldc_imu(mu, a1, a2, a3, a4, ltype);
        }
    }
    return partial;
}

/**
 * Accumulate disc/edge flux with quadratic limb darkening.
 */
__device__
double accumulate_disc(
    const GpuPoint* __restrict__ pts, int n_pts,
    double ex, double ey, double ez, float phase_f,
    double lin_limb, double quad_limb)
{
    double partial = 0.0;
    for (int idx = threadIdx.x; idx < n_pts; idx += blockDim.x) {
        const GpuPoint* pt = &pts[idx];
        double mu = ex * (double)pt->dirn_x
                  + ey * (double)pt->dirn_y
                  + ez * (double)pt->dirn_z;
        if (mu > 0.0 && pt_visible(pt, phase_f)) {
            double ommu = 1.0 - mu;
            partial += mu * (double)pt->flux * (1.0 - ommu * (lin_limb + quad_limb * ommu));
        }
    }
    return partial;
}

/**
 * Accumulate bright spot flux (no limb darkening).
 */
__device__
double accumulate_spot(
    const GpuPoint* __restrict__ pts, int n_pts,
    double ex, double ey, double ez, float phase_f)
{
    double partial = 0.0;
    for (int idx = threadIdx.x; idx < n_pts; idx += blockDim.x) {
        const GpuPoint* pt = &pts[idx];
        double mu = ex * (double)pt->dirn_x
                  + ey * (double)pt->dirn_y
                  + ez * (double)pt->dirn_z;
        if (mu > 0.0 && pt_visible(pt, phase_f)) {
            partial += mu * (double)pt->flux;
        }
    }
    return partial;
}

/* ------------------------------------------------------------------ */
/*  Main flux evaluation kernel                                        */
/* ------------------------------------------------------------------ */

/**
 * flux_eval_kernel — compute total flux for each time point.
 *
 * blockIdx.x  = time point index
 * threadIdx.x = worker thread within the block
 *
 * Grid layout: star1f, star1c, star2f, star2c, disc, edge, spot are packed
 * contiguously in a single device buffer. Offsets computed from params.
 */
__global__ void flux_eval_kernel(
    const GpuDatum*  __restrict__ data,
    const GpuFudge*  __restrict__ fudge,
    const GpuPoint*  __restrict__ all_points,
    const GpuFluxParams* __restrict__ params,
    double* __restrict__ out_flux,
    int n_times)
{
    int tid = blockIdx.x;
    if (tid >= n_times) return;

    const GpuFluxParams p = *params;
    const GpuDatum d = data[tid];
    const GpuFudge f = fudge[tid];

    /* Compute offsets into the packed point buffer */
    int off_star1f = 0;
    int off_star1c = off_star1f + p.n_star1f;
    int off_star2f = off_star1c + p.n_star1c;
    int off_star2c = off_star2f + p.n_star2f;
    int off_disc   = off_star2c + p.n_star2c;
    int off_edge   = off_disc + p.n_disc;
    int off_spot   = off_edge + p.n_edge;

    const GpuPoint* star1f_pts = all_points + off_star1f;
    const GpuPoint* star1c_pts = all_points + off_star1c;
    const GpuPoint* star2f_pts = all_points + off_star2f;
    const GpuPoint* star2c_pts = all_points + off_star2c;
    const GpuPoint* disc_pts   = all_points + off_disc;
    const GpuPoint* edge_pts   = all_points + off_edge;
    const GpuPoint* spot_pts   = all_points + off_spot;

    double sum = 0.0;
    int ndiv_denom = (d.ndiv > 1) ? (d.ndiv - 1) : 1;

    for (int nd = 0; nd < d.ndiv; nd++) {
        double phi, wgt;
        exposure_weight(d.phase, d.expose, nd, d.ndiv, &phi, &wgt);

        double ex, ey, ez;
        set_earth(p.cosi, p.sini, phi, &ex, &ey, &ez);

        float phase_f = (float)(phi - floor(phi));

        int ptype = grid_type(phi, p.gint_phase1, p.gint_phase2);
        const GpuPoint* star1 = (ptype == 1) ? star1f_pts : star1c_pts;
        int n_star1 = (ptype == 1) ? p.n_star1f : p.n_star1c;
        const GpuPoint* star2 = (ptype == 3) ? star2f_pts : star2c_pts;
        int n_star2 = (ptype == 3) ? p.n_star2f : p.n_star2c;

        /* Star 1 */
        double s1 = accumulate_star(
            star1, n_star1, ex, ey, ez, phase_f,
            p.ldc1_1, p.ldc1_2, p.ldc1_3, p.ldc1_4, p.mucrit1, p.ltype1,
            p.beam_factor1, p.spin1, p.xcofm, p.vfac, 0, 0, 0.0);
        s1 = block_reduce_sum(s1);

        double scale1 = gint_scale1(phi, p.gint_phase1, p.gint_scale11, p.gint_scale12);

        /* Star 2 */
        double s2 = accumulate_star(
            star2, n_star2, ex, ey, ez, phase_f,
            p.ldc2_1, p.ldc2_2, p.ldc2_3, p.ldc2_4, p.mucrit2, p.ltype2,
            p.beam_factor2, p.spin2, p.xcofm, p.vfac, 1, p.has_glens, p.rlens1);
        s2 = block_reduce_sum(s2);

        double scale2 = gint_scale2(phi, p.gint_phase2, p.gint_scale21, p.gint_scale22);

        /* Disc */
        double sd = 0.0;
        if (p.n_disc > 0) {
            sd = accumulate_disc(disc_pts, p.n_disc, ex, ey, ez, phase_f,
                                 p.lin_limb_disc, p.quad_limb_disc);
            sd = block_reduce_sum(sd);
        }

        /* Edge */
        double se = 0.0;
        if (p.n_edge > 0) {
            se = accumulate_disc(edge_pts, p.n_edge, ex, ey, ez, phase_f,
                                 p.lin_limb_disc, p.quad_limb_disc);
            se = block_reduce_sum(se);
        }

        /* Spot */
        double ss = 0.0;
        if (p.n_spot > 0) {
            ss = accumulate_spot(spot_pts, p.n_spot, ex, ey, ez, phase_f);
            ss = block_reduce_sum(ss);
        }

        if (threadIdx.x == 0) {
            sum += wgt * (scale1 * s1 + scale2 * s2 + sd + se + ss);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double flux = (sum / (double)ndiv_denom) * f.slfac + f.third;
        out_flux[tid] = flux;
    }
}

/* ------------------------------------------------------------------ */
/*  Host-callable C wrappers for Rust FFI                              */
/* ------------------------------------------------------------------ */

extern "C" {

/** Initialise CUDA device. Returns cudaError_t. */
int lcuda_init(int device_id) {
    return (int)cudaSetDevice(device_id);
}

/** Allocate device memory. Returns cudaError_t. */
int lcuda_malloc(void** devptr, size_t size) {
    return (int)cudaMalloc(devptr, size);
}

/** Free device memory. Returns cudaError_t. */
int lcuda_free(void* devptr) {
    return (int)cudaFree(devptr);
}

/** Copy host to device. Returns cudaError_t. */
int lcuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    return (int)cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

/** Copy device to host. Returns cudaError_t. */
int lcuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    return (int)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

/** Synchronise device. Returns cudaError_t. */
int lcuda_sync() {
    return (int)cudaDeviceSynchronize();
}

/**
 * Launch flux evaluation kernel.
 *
 * @param d_data      Device pointer to GpuDatum array
 * @param d_fudge     Device pointer to GpuFudge array
 * @param d_points    Device pointer to packed GpuPoint buffer
 * @param d_params    Device pointer to GpuFluxParams
 * @param d_out_flux  Device pointer to output flux array (f64)
 * @param n_times     Number of time points
 * @return cudaError_t
 */
int lcuda_flux_eval(
    const void* d_data,
    const void* d_fudge,
    const void* d_points,
    const void* d_params,
    void* d_out_flux,
    int n_times)
{
    dim3 grid(n_times);
    dim3 block(BLOCK_SIZE);

    flux_eval_kernel<<<grid, block>>>(
        (const GpuDatum*)d_data,
        (const GpuFudge*)d_fudge,
        (const GpuPoint*)d_points,
        (const GpuFluxParams*)d_params,
        (double*)d_out_flux,
        n_times);

    return (int)cudaGetLastError();
}

/** Query number of CUDA devices. Returns count, or -1 on error. */
int lcuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return -1;
    return count;
}

/* ---- Stream management ---- */

/** Create a CUDA stream. Returns cudaError_t. */
int lcuda_stream_create(void** stream_out) {
    return (int)cudaStreamCreate((cudaStream_t*)stream_out);
}

/** Destroy a CUDA stream. Returns cudaError_t. */
int lcuda_stream_destroy(void* stream) {
    return (int)cudaStreamDestroy((cudaStream_t)stream);
}

/** Synchronise a specific stream. Returns cudaError_t. */
int lcuda_stream_sync(void* stream) {
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

/** Async copy host to device on a stream. Returns cudaError_t. */
int lcuda_memcpy_h2d_async(void* dst, const void* src, size_t size, void* stream) {
    return (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

/** Async copy device to host on a stream. Returns cudaError_t. */
int lcuda_memcpy_d2h_async(void* dst, const void* src, size_t size, void* stream) {
    return (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

/**
 * Launch flux evaluation kernel on a specific stream.
 */
int lcuda_flux_eval_stream(
    const void* d_data,
    const void* d_fudge,
    const void* d_points,
    const void* d_params,
    void* d_out_flux,
    int n_times,
    void* stream)
{
    dim3 grid(n_times);
    dim3 block(BLOCK_SIZE);

    flux_eval_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const GpuDatum*)d_data,
        (const GpuFudge*)d_fudge,
        (const GpuPoint*)d_points,
        (const GpuFluxParams*)d_params,
        (double*)d_out_flux,
        n_times);

    return (int)cudaGetLastError();
}

/** Allocate page-locked (pinned) host memory for async transfers. */
int lcuda_host_alloc(void** ptr, size_t size) {
    return (int)cudaMallocHost(ptr, size);
}

/** Free page-locked host memory. */
int lcuda_host_free(void* ptr) {
    return (int)cudaFreeHost(ptr);
}

} /* extern "C" */
