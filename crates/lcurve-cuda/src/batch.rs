use crate::context::{CudaContext, CudaStream, DevicePtr, PinnedBuf};
use crate::error::{check_cuda, CudaError};
use crate::transfer::{flatten_points, GpuDatum, GpuFluxParams, GpuFudge, GpuPoint};
use lcurve::model::Model;
use lcurve::types::{Data, Ginterp, LDCType, Point};
use lcurve_subs::{sqr, C, DAY, PI, TWOPI};
use rayon::prelude::*;

/// All CPU-side data needed to launch a GPU kernel for one model.
struct PreparedModel {
    all_points: Vec<GpuPoint>,
    params: GpuFluxParams,
    gpu_data: Vec<GpuDatum>,
    gpu_fudge: Vec<GpuFudge>,
}

/// Pre-allocated GPU buffers reused across all models in a batch.
struct GpuBuffers {
    /// Double-buffered point arrays (two slots for pipelining)
    d_points: [DevicePtr; 2],
    /// Double-buffered params
    d_params: [DevicePtr; 2],
    /// Double-buffered data arrays
    d_data: [DevicePtr; 2],
    /// Double-buffered fudge arrays
    d_fudge: [DevicePtr; 2],
    /// Double-buffered output flux
    d_flux: [DevicePtr; 2],
    /// Two CUDA streams for pipelining
    streams: [CudaStream; 2],
    /// Pinned host memory for staging point uploads
    pinned_points: [PinnedBuf; 2],
    /// Pinned host memory for staging params
    pinned_params: [PinnedBuf; 2],
    /// Pinned host memory for staging data
    pinned_data: [PinnedBuf; 2],
    /// Pinned host memory for staging fudge
    pinned_fudge: [PinnedBuf; 2],
    /// Pinned host memory for downloading flux results
    pinned_flux: [PinnedBuf; 2],
}

impl GpuBuffers {
    fn new(max_points: usize, n_times: usize) -> Result<Self, CudaError> {
        let points_bytes = max_points * std::mem::size_of::<GpuPoint>();
        let params_bytes = std::mem::size_of::<GpuFluxParams>();
        let data_bytes = n_times * std::mem::size_of::<GpuDatum>();
        let fudge_bytes = n_times * std::mem::size_of::<GpuFudge>();
        let flux_bytes = n_times * std::mem::size_of::<f64>();

        Ok(GpuBuffers {
            d_points: [DevicePtr::alloc(points_bytes)?, DevicePtr::alloc(points_bytes)?],
            d_params: [DevicePtr::alloc(params_bytes)?, DevicePtr::alloc(params_bytes)?],
            d_data: [DevicePtr::alloc(data_bytes)?, DevicePtr::alloc(data_bytes)?],
            d_fudge: [DevicePtr::alloc(fudge_bytes)?, DevicePtr::alloc(fudge_bytes)?],
            d_flux: [DevicePtr::alloc(flux_bytes)?, DevicePtr::alloc(flux_bytes)?],
            streams: [CudaStream::new()?, CudaStream::new()?],
            pinned_points: [PinnedBuf::alloc(points_bytes)?, PinnedBuf::alloc(points_bytes)?],
            pinned_params: [PinnedBuf::alloc(params_bytes)?, PinnedBuf::alloc(params_bytes)?],
            pinned_data: [PinnedBuf::alloc(data_bytes)?, PinnedBuf::alloc(data_bytes)?],
            pinned_fudge: [PinnedBuf::alloc(fudge_bytes)?, PinnedBuf::alloc(fudge_bytes)?],
            pinned_flux: [PinnedBuf::alloc(flux_bytes)?, PinnedBuf::alloc(flux_bytes)?],
        })
    }
}

/// Evaluate chi-squared for a batch of parameter sets using the GPU.
///
/// **Phase 1 (parallel):** Rayon-parallel CPU grid setup + flatten for all N models.
/// **Phase 2 (pipelined):** Double-buffered GPU upload + kernel + download using two CUDA streams.
///
/// Returns `Vec<f64>` of length N with chi-squared values (NaN for failures).
pub fn chisq_batch_gpu(
    ctx: &mut CudaContext,
    base: &Model,
    data: &Data,
    param_names: &[&str],
    param_values: &[f64],
    scale: bool,
) -> Result<Vec<f64>, CudaError> {
    let ndim = param_names.len();
    assert!(
        ndim > 0 && param_values.len() % ndim == 0,
        "param_values length must be a multiple of param_names length"
    );
    let n = param_values.len() / ndim;
    let n_times = data.len();

    // Pre-compute polynomial fudge factor data (shared across all models)
    let (xmin, xmax) = data.iter().fold((data[0].time, data[0].time), |(mn, mx), d| {
        (mn.min(d.time), mx.max(d.time))
    });
    let middle = (xmin + xmax) / 2.0;
    let range = (xmax - xmin) / 2.0;

    // ================================================================
    // Phase 1: Rayon-parallel CPU grid setup for all N models
    // ================================================================
    let prepared: Vec<Option<PreparedModel>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut mdl = base.clone();
            let row = &param_values[i * ndim..(i + 1) * ndim];
            for (j, name) in param_names.iter().enumerate() {
                mdl.set_param_value(name, row[j]);
            }
            prepare_model(&mdl, data, middle, range).ok()
        })
        .collect();

    // Find max point buffer size for pre-allocation
    let max_points = prepared
        .iter()
        .filter_map(|p| p.as_ref())
        .map(|p| p.all_points.len())
        .max()
        .unwrap_or(0);

    if max_points == 0 {
        return Ok(vec![f64::NAN; n]);
    }

    // ================================================================
    // Phase 2: Double-buffered GPU pipelining
    // ================================================================
    let mut bufs = GpuBuffers::new(max_points, n_times)?;
    let mut results = vec![f64::NAN; n];

    // Collect valid models with their original indices
    let valid: Vec<(usize, &PreparedModel)> = prepared
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|m| (i, m)))
        .collect();

    if valid.is_empty() {
        return Ok(results);
    }

    // Pipeline: upload model[i] on stream[slot] while kernel runs for model[i-1] on stream[1-slot]
    // Slot 0 and slot 1 alternate.
    let mut prev_slot: Option<(usize, usize)> = None; // (original_index, slot)

    for (vi, &(orig_idx, prep)) in valid.iter().enumerate() {
        let slot = vi % 2;

        // Stage data into pinned memory and async upload on stream[slot]
        upload_model_async(prep, &mut bufs, slot, n_times)?;

        // Launch kernel on stream[slot]
        check_cuda(unsafe {
            crate::lcuda_flux_eval_stream(
                bufs.d_data[slot].as_ptr(),
                bufs.d_fudge[slot].as_ptr(),
                bufs.d_points[slot].as_ptr(),
                bufs.d_params[slot].as_ptr(),
                bufs.d_flux[slot].as_mut_ptr(),
                n_times as i32,
                bufs.streams[slot].ptr(),
            )
        })?;

        // Start async download of flux results on stream[slot]
        unsafe {
            bufs.d_flux[slot].download_async(
                bufs.pinned_flux[slot].as_mut_slice::<f64>(n_times),
                &bufs.streams[slot],
            )?;
        }

        // If there's a previous model, its stream should be done by now — sync and collect result
        if let Some((prev_idx, prev_s)) = prev_slot {
            bufs.streams[prev_s].sync()?;
            let flux = bufs.pinned_flux[prev_s].as_mut_slice::<f64>(n_times);
            match compute_chisq(data, flux, scale) {
                Ok(chisq) => results[prev_idx] = chisq,
                Err(_) => {}
            }
        }

        prev_slot = Some((orig_idx, slot));
    }

    // Collect the last model's result
    if let Some((prev_idx, prev_s)) = prev_slot {
        bufs.streams[prev_s].sync()?;
        let flux = bufs.pinned_flux[prev_s].as_mut_slice::<f64>(n_times);
        match compute_chisq(data, flux, scale) {
            Ok(chisq) => results[prev_idx] = chisq,
            Err(_) => {}
        }
    }

    Ok(results)
}

/// Stage prepared model data into pinned memory and async-upload to GPU on the given stream.
fn upload_model_async(
    prep: &PreparedModel,
    bufs: &mut GpuBuffers,
    slot: usize,
    n_times: usize,
) -> Result<(), CudaError> {
    // Stage points into pinned memory, then async upload
    let staged_points = bufs.pinned_points[slot].stage(&prep.all_points);
    unsafe {
        bufs.d_points[slot].upload_async(staged_points, &bufs.streams[slot])?;
    }

    // Stage params
    let staged_params = bufs.pinned_params[slot].stage(std::slice::from_ref(&prep.params));
    unsafe {
        bufs.d_params[slot].upload_async(staged_params, &bufs.streams[slot])?;
    }

    // Stage data
    let staged_data = bufs.pinned_data[slot].stage(&prep.gpu_data);
    unsafe {
        bufs.d_data[slot].upload_async(staged_data, &bufs.streams[slot])?;
    }

    // Stage fudge
    let staged_fudge = bufs.pinned_fudge[slot].stage(&prep.gpu_fudge);
    unsafe {
        bufs.d_fudge[slot].upload_async(staged_fudge, &bufs.streams[slot])?;
    }

    Ok(())
}

/// Phase 1: CPU grid setup + flatten + phase computation for one model.
/// This runs in parallel via Rayon — no GPU calls here.
fn prepare_model(
    mdl: &Model,
    data: &Data,
    middle: f64,
    range: f64,
) -> Result<PreparedModel, CudaError> {
    use lcurve_roche::Star;

    let (r1, mut r2) = mdl.get_r1r2();
    let rl2 = 1.0 - lcurve_roche::lagrange::xl12(mdl.q.value, mdl.spin2.value)?;
    if r2 < 0.0 {
        r2 = rl2;
    } else if r2 > rl2 {
        return Err(CudaError::Generic(
            "secondary larger than Roche lobe".into(),
        ));
    }

    let ldc1 = mdl.get_ldc1();
    let ldc2 = mdl.get_ldc2();

    let rlens1 = if mdl.glens1 {
        let gm = (1000.0 * mdl.velocity_scale.value).powi(3) * mdl.tperiod * DAY / TWOPI;
        let a = (gm / sqr(TWOPI / DAY / mdl.tperiod)).powf(1.0 / 3.0);
        4.0 * gm / (1.0 + mdl.q.value) / a / sqr(C)
    } else {
        0.0
    };

    // Generate grids
    let mut star1f = lcurve::grid::set_star_grid(mdl, Star::Primary, true)?;
    let mut star2f = lcurve::grid::set_star_grid(mdl, Star::Secondary, true)?;
    lcurve::brightness::set_star_continuum(mdl, &mut star1f, &mut star2f)?;

    let mut star1c = if mdl.nlat1f == mdl.nlat1c {
        star1f.clone()
    } else {
        lcurve::grid::set_star_grid(mdl, Star::Primary, false)?
    };

    let copy2 = mdl.nlat2f == mdl.nlat2c
        && (!mdl.npole || r1 >= r2 || (mdl.nlatfill == 0 && mdl.nlngfill == 0));

    let mut star2c = if copy2 {
        star2f.clone()
    } else {
        lcurve::grid::set_star_grid(mdl, Star::Secondary, false)?
    };

    if mdl.nlat1c != mdl.nlat1f || !copy2 {
        lcurve::brightness::set_star_continuum(mdl, &mut star1c, &mut star2c)?;
    }

    // Grid interpolation
    let mut gint = Ginterp {
        phase1: mdl.phase1,
        phase2: mdl.phase2,
        scale11: 1.0,
        scale12: 1.0,
        scale21: 1.0,
        scale22: 1.0,
    };

    if mdl.nlat1c != mdl.nlat1f {
        let ff = lcurve::flux::comp_star1(
            mdl.iangle.value, &ldc1, 0.9999999999 * mdl.phase1,
            0.0, 1, mdl.q.value, mdl.beam_factor1.value,
            mdl.velocity_scale.value as f32, &gint, &star1f, &star1c,
        );
        let fc = lcurve::flux::comp_star1(
            mdl.iangle.value, &ldc1, 1.0000000001 * mdl.phase1,
            0.0, 1, mdl.q.value, mdl.beam_factor1.value,
            mdl.velocity_scale.value as f32, &gint, &star1f, &star1c,
        );
        gint.scale11 = ff / fc;

        let ff = lcurve::flux::comp_star1(
            mdl.iangle.value, &ldc1, 1.0 - 0.9999999999 * mdl.phase1,
            0.0, 1, mdl.q.value, mdl.beam_factor1.value,
            mdl.velocity_scale.value as f32, &gint, &star1f, &star1c,
        );
        let fc = lcurve::flux::comp_star1(
            mdl.iangle.value, &ldc1, 1.0 - 1.0000000001 * mdl.phase1,
            0.0, 1, mdl.q.value, mdl.beam_factor1.value,
            mdl.velocity_scale.value as f32, &gint, &star1f, &star1c,
        );
        gint.scale12 = ff / fc;
    }

    if !copy2 {
        let ff = lcurve::flux::comp_star2(
            mdl.iangle.value, &ldc2, 1.0 - 1.0000000001 * mdl.phase2,
            0.0, 1, mdl.q.value, mdl.beam_factor2.value,
            mdl.velocity_scale.value as f32, mdl.glens1, rlens1,
            &gint, &star2f, &star2c,
        );
        let fc = lcurve::flux::comp_star2(
            mdl.iangle.value, &ldc2, 1.0 - 0.9999999999 * mdl.phase2,
            0.0, 1, mdl.q.value, mdl.beam_factor2.value,
            mdl.velocity_scale.value as f32, mdl.glens1, rlens1,
            &gint, &star2f, &star2c,
        );
        gint.scale21 = ff / fc;

        let ff = lcurve::flux::comp_star2(
            mdl.iangle.value, &ldc2, 1.0000000001 * mdl.phase2,
            0.0, 1, mdl.q.value, mdl.beam_factor2.value,
            mdl.velocity_scale.value as f32, mdl.glens1, rlens1,
            &gint, &star2f, &star2c,
        );
        let fc = lcurve::flux::comp_star2(
            mdl.iangle.value, &ldc2, 0.9999999999 * mdl.phase2,
            0.0, 1, mdl.q.value, mdl.beam_factor2.value,
            mdl.velocity_scale.value as f32, mdl.glens1, rlens1,
            &gint, &star2f, &star2c,
        );
        gint.scale22 = ff / fc;
    }

    // Disc and spot
    let mut disc: Vec<Point> = Vec::new();
    let mut edge: Vec<Point> = Vec::new();
    let mut spot: Vec<Point> = Vec::new();

    if mdl.add_disc {
        disc = lcurve::grid::set_disc_grid(mdl)?;
        edge = lcurve::grid::set_disc_edge(mdl, true)?;

        let rdisc1 = if mdl.rdisc1.value > 0.0 { mdl.rdisc1.value } else { r1 };
        let rdisc2 = if mdl.rdisc2.value > 0.0 {
            mdl.rdisc2.value
        } else {
            mdl.radius_spot.value
        };

        if mdl.opaque {
            for grids in [&mut star1f, &mut star1c, &mut star2f, &mut star2c] {
                grids.par_iter_mut().for_each(|pt| {
                    let eclipses = lcurve_roche::disc_eclipse::disc_eclipse(
                        mdl.iangle.value, rdisc1, rdisc2,
                        mdl.beta_disc.value, mdl.height_disc.value, &pt.posn,
                    );
                    for e in eclipses {
                        pt.eclipse.push(e);
                    }
                });
            }
        }

        lcurve::brightness::set_disc_continuum(
            rdisc2, mdl.temp_disc.value, mdl.texp_disc.value,
            mdl.wavelength, &mut disc,
        );
        lcurve::brightness::set_edge_continuum(
            mdl.temp_edge.value, r2, mdl.t2.value.abs(),
            mdl.absorb_edge.value, mdl.wavelength, &mut edge,
        );
    }

    if mdl.add_spot {
        spot = lcurve::grid::set_bright_spot_grid(mdl)?;
    }

    // Flatten points
    let gpu_star1f = flatten_points(&star1f);
    let gpu_star1c = flatten_points(&star1c);
    let gpu_star2f = flatten_points(&star2f);
    let gpu_star2c = flatten_points(&star2c);
    let gpu_disc = flatten_points(&disc);
    let gpu_edge = flatten_points(&edge);
    let gpu_spot = flatten_points(&spot);

    // Pack contiguously
    let total_points = gpu_star1f.len() + gpu_star1c.len() + gpu_star2f.len()
        + gpu_star2c.len() + gpu_disc.len() + gpu_edge.len() + gpu_spot.len();
    let mut all_points: Vec<GpuPoint> = Vec::with_capacity(total_points);
    all_points.extend_from_slice(&gpu_star1f);
    all_points.extend_from_slice(&gpu_star1c);
    all_points.extend_from_slice(&gpu_star2f);
    all_points.extend_from_slice(&gpu_star2c);
    all_points.extend_from_slice(&gpu_disc);
    all_points.extend_from_slice(&gpu_edge);
    all_points.extend_from_slice(&gpu_spot);

    // Build GpuFluxParams
    let ri = mdl.iangle.value.to_radians();
    let params = GpuFluxParams {
        cosi: ri.cos(),
        sini: ri.sin(),
        xcofm: mdl.q.value / (1.0 + mdl.q.value),
        vfac: mdl.velocity_scale.value / (C / 1e3),
        ldc1_1: mdl.ldc1_1.value,
        ldc1_2: mdl.ldc1_2.value,
        ldc1_3: mdl.ldc1_3.value,
        ldc1_4: mdl.ldc1_4.value,
        mucrit1: mdl.mucrit1,
        ltype1: if mdl.limb1 == LDCType::Poly { 0 } else { 1 },
        ldc2_1: mdl.ldc2_1.value,
        ldc2_2: mdl.ldc2_2.value,
        ldc2_3: mdl.ldc2_3.value,
        ldc2_4: mdl.ldc2_4.value,
        mucrit2: mdl.mucrit2,
        ltype2: if mdl.limb2 == LDCType::Poly { 0 } else { 1 },
        beam_factor1: mdl.beam_factor1.value,
        beam_factor2: mdl.beam_factor2.value,
        spin1: mdl.spin1.value,
        spin2: mdl.spin2.value,
        lin_limb_disc: mdl.lin_limb_disc.value,
        quad_limb_disc: mdl.quad_limb_disc.value,
        has_glens: if mdl.glens1 { 1 } else { 0 },
        rlens1,
        gint_phase1: gint.phase1,
        gint_phase2: gint.phase2,
        gint_scale11: gint.scale11,
        gint_scale12: gint.scale12,
        gint_scale21: gint.scale21,
        gint_scale22: gint.scale22,
        n_star1f: gpu_star1f.len() as i32,
        n_star1c: gpu_star1c.len() as i32,
        n_star2f: gpu_star2f.len() as i32,
        n_star2c: gpu_star2c.len() as i32,
        n_disc: gpu_disc.len() as i32,
        n_edge: gpu_edge.len() as i32,
        n_spot: gpu_spot.len() as i32,
        _pad: [0],
    };

    // Pre-compute phases and fudge factors
    let n_times = data.len();
    let mut gpu_data: Vec<GpuDatum> = Vec::with_capacity(n_times);
    let mut gpu_fudge: Vec<GpuFudge> = Vec::with_capacity(n_times);

    for np in 0..n_times {
        let mut phase = (data[np].time - mdl.t0.value) / mdl.period.value;
        for _ in 0..4 {
            phase -= (mdl.t0.value + phase * (mdl.period.value + mdl.pdot.value * phase)
                - data[np].time)
                / (mdl.period.value + 2.0 * mdl.pdot.value * phase);
        }
        phase += mdl.deltat.value / mdl.period.value / 2.0 * ((2.0 * PI * phase).cos() - 1.0);

        let expose = data[np].expose / mdl.period.value;
        let frac = (data[np].time - middle) / range;
        let slfac =
            1.0 + frac * (mdl.slope.value + frac * (mdl.quad.value + frac * mdl.cube.value));

        gpu_data.push(GpuDatum {
            phase,
            expose,
            flux_obs: data[np].flux,
            ferr: data[np].ferr,
            weight: data[np].weight,
            ndiv: data[np].ndiv,
            _pad: 0,
        });

        gpu_fudge.push(GpuFudge {
            slfac,
            third: mdl.third.value,
        });
    }

    Ok(PreparedModel {
        all_points,
        params,
        gpu_data,
        gpu_fudge,
    })
}

/// Compute chi-squared from flux array and data, with optional single-factor rescaling.
fn compute_chisq(data: &Data, calc: &[f64], scale: bool) -> Result<f64, CudaError> {
    if scale {
        let mut sdy = 0.0;
        let mut syy = 0.0;
        let mut wnok = 0.0;
        for (np, d) in data.iter().enumerate() {
            if d.weight > 0.0 {
                let wgt = d.weight / sqr(d.ferr);
                sdy += wgt * d.flux * calc[np];
                syy += wgt * calc[np] * calc[np];
                wnok += d.weight;
            }
        }
        if wnok > 0.0 && syy > 0.0 {
            let sfac = sdy / syy;
            let mut chisq = 0.0;
            for (np, d) in data.iter().enumerate() {
                if d.weight > 0.0 {
                    chisq += d.weight * sqr((d.flux - sfac * calc[np]) / d.ferr);
                }
            }
            Ok(chisq)
        } else {
            Ok(0.0)
        }
    } else {
        let mut chisq = 0.0;
        for (np, d) in data.iter().enumerate() {
            if d.weight > 0.0 {
                chisq += d.weight * sqr((d.flux - calc[np]) / d.ferr);
            }
        }
        Ok(chisq)
    }
}
