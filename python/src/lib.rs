use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;
use numpy::{PyArray1, IntoPyArray};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Device selection — global state for GPU/CPU dispatch
// ---------------------------------------------------------------------------

/// Which compute device to use.
#[derive(Clone, Debug)]
enum DeviceChoice {
    Cpu,
    Cuda(i32), // device_id
}

impl DeviceChoice {
    fn is_gpu(&self) -> bool {
        matches!(self, DeviceChoice::Cuda(_))
    }

    fn to_string(&self) -> String {
        match self {
            DeviceChoice::Cpu => "cpu".to_string(),
            DeviceChoice::Cuda(id) => format!("cuda:{}", id),
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_lowercase();
        if s == "cpu" {
            Ok(DeviceChoice::Cpu)
        } else if s == "cuda" || s == "gpu" {
            Ok(DeviceChoice::Cuda(0))
        } else if let Some(rest) = s.strip_prefix("cuda:") {
            let id: i32 = rest
                .parse()
                .map_err(|_| format!("invalid device id: '{}'", rest))?;
            if id < 0 {
                return Err(format!("device id must be >= 0, got {}", id));
            }
            Ok(DeviceChoice::Cuda(id))
        } else {
            Err(format!(
                "unknown device '{}': expected 'cpu', 'cuda', or 'cuda:N'",
                s
            ))
        }
    }
}

static DEVICE_CHOICE: Mutex<DeviceChoice> = Mutex::new(DeviceChoice::Cpu);

/// Wrapper around lcurve::model::Model exposed to Python.
#[pyclass]
struct Model {
    inner: lcurve::model::Model,
}

/// Result of a light curve computation.
#[pyclass]
#[derive(Clone)]
struct LcResult {
    times: Vec<f64>,
    calc: Vec<f64>,
    #[pyo3(get)]
    wdwarf: f64,
    #[pyo3(get)]
    chisq: f64,
    #[pyo3(get)]
    wnok: f64,
    #[pyo3(get)]
    logg1: f64,
    #[pyo3(get)]
    logg2: f64,
    #[pyo3(get)]
    rv1: f64,
    #[pyo3(get)]
    rv2: f64,
    sfac: Vec<f64>,
}

#[pymethods]
impl LcResult {
    /// Computed flux values as a numpy array.
    #[getter]
    fn calc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.calc.clone().into_pyarray(py)
    }

    /// Alias: computed flux values as a numpy array.
    #[getter]
    fn flux<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.calc.clone().into_pyarray(py)
    }

    /// Time values as a numpy array.
    #[getter]
    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.times.clone().into_pyarray(py)
    }

    /// Scale factors [star1, disc, edge, spot, star2] as a numpy array.
    #[getter]
    fn sfac<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.sfac.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "LcResult(npoints={}, wdwarf={:.6e}, chisq={:.4}, logg1={:.4}, logg2={:.4})",
            self.calc.len(), self.wdwarf, self.chisq, self.logg1, self.logg2
        )
    }
}

// ---------------------------------------------------------------------------
// Central macro listing all (name_str, field_ident) Pparam fields of Model.
// Every helper that needs to iterate or match on parameter names builds on
// this single source of truth.
// ---------------------------------------------------------------------------

macro_rules! for_all_pparams {
    ($macro_name:ident) => {
        $macro_name! {
            "q" => q,
            "iangle" => iangle,
            "r1" => r1,
            "r2" => r2,
            "cphi3" => cphi3,
            "cphi4" => cphi4,
            "spin1" => spin1,
            "spin2" => spin2,
            "t1" => t1,
            "t2" => t2,
            "ldc1_1" => ldc1_1,
            "ldc1_2" => ldc1_2,
            "ldc1_3" => ldc1_3,
            "ldc1_4" => ldc1_4,
            "ldc2_1" => ldc2_1,
            "ldc2_2" => ldc2_2,
            "ldc2_3" => ldc2_3,
            "ldc2_4" => ldc2_4,
            "velocity_scale" => velocity_scale,
            "beam_factor1" => beam_factor1,
            "beam_factor2" => beam_factor2,
            "t0" => t0,
            "period" => period,
            "pdot" => pdot,
            "deltat" => deltat,
            "gravity_dark1" => gravity_dark1,
            "gravity_dark2" => gravity_dark2,
            "absorb" => absorb,
            "slope" => slope,
            "quad" => quad,
            "cube" => cube,
            "third" => third,
            "rdisc1" => rdisc1,
            "rdisc2" => rdisc2,
            "height_disc" => height_disc,
            "beta_disc" => beta_disc,
            "temp_disc" => temp_disc,
            "texp_disc" => texp_disc,
            "lin_limb_disc" => lin_limb_disc,
            "quad_limb_disc" => quad_limb_disc,
            "temp_edge" => temp_edge,
            "absorb_edge" => absorb_edge,
            "radius_spot" => radius_spot,
            "length_spot" => length_spot,
            "height_spot" => height_spot,
            "expon_spot" => expon_spot,
            "epow_spot" => epow_spot,
            "angle_spot" => angle_spot,
            "yaw_spot" => yaw_spot,
            "temp_spot" => temp_spot,
            "tilt_spot" => tilt_spot,
            "cfrac_spot" => cfrac_spot,
            "stsp11_long" => stsp11_long,
            "stsp11_lat" => stsp11_lat,
            "stsp11_fwhm" => stsp11_fwhm,
            "stsp11_tcen" => stsp11_tcen,
            "stsp12_long" => stsp12_long,
            "stsp12_lat" => stsp12_lat,
            "stsp12_fwhm" => stsp12_fwhm,
            "stsp12_tcen" => stsp12_tcen,
            "stsp13_long" => stsp13_long,
            "stsp13_lat" => stsp13_lat,
            "stsp13_fwhm" => stsp13_fwhm,
            "stsp13_tcen" => stsp13_tcen,
            "stsp21_long" => stsp21_long,
            "stsp21_lat" => stsp21_lat,
            "stsp21_fwhm" => stsp21_fwhm,
            "stsp21_tcen" => stsp21_tcen,
            "stsp22_long" => stsp22_long,
            "stsp22_lat" => stsp22_lat,
            "stsp22_fwhm" => stsp22_fwhm,
            "stsp22_tcen" => stsp22_tcen,
            "uesp_long1" => uesp_long1,
            "uesp_long2" => uesp_long2,
            "uesp_lathw" => uesp_lathw,
            "uesp_taper" => uesp_taper,
            "uesp_temp" => uesp_temp,
        }
    };
}

// Derived helpers built on for_all_pparams!

fn get_pparam_mut<'a>(
    model: &'a mut lcurve::model::Model,
    name: &str,
) -> Option<&'a mut lcurve::model::Pparam> {
    macro_rules! do_match_mut {
        ($( $key:literal => $field:ident ),+ $(,)?) => {
            match name {
                $( $key => Some(&mut model.$field), )+
                _ => None,
            }
        };
    }
    for_all_pparams!(do_match_mut)
}

fn get_pparam_ref<'a>(
    model: &'a lcurve::model::Model,
    name: &str,
) -> Option<&'a lcurve::model::Pparam> {
    macro_rules! do_match_ref {
        ($( $key:literal => $field:ident ),+ $(,)?) => {
            match name {
                $( $key => Some(&model.$field), )+
                _ => None,
            }
        };
    }
    for_all_pparams!(do_match_ref)
}

fn collect_free_params(model: &lcurve::model::Model) -> Vec<(&'static str, &lcurve::model::Pparam)> {
    let mut out = Vec::new();
    macro_rules! do_collect {
        ($( $key:literal => $field:ident ),+ $(,)?) => {
            $(
                if model.$field.vary && model.$field.defined {
                    out.push(($key, &model.$field));
                }
            )+
        };
    }
    for_all_pparams!(do_collect);
    out
}

#[pymethods]
impl Model {
    /// Load a model from a parameter file.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to an lcurve model parameter file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = lcurve::model::Model::from_file(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Model { inner })
    }

    /// Get the value of a named physical parameter.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Parameter name (e.g. ``"q"``, ``"iangle"``, ``"t1"``).
    ///
    /// Returns
    /// -------
    /// float
    fn get_param(&self, name: &str) -> PyResult<f64> {
        get_pparam_ref(&self.inner, name)
            .map(|p| p.value)
            .ok_or_else(|| PyValueError::new_err(format!("unknown parameter: '{}'", name)))
    }

    /// Set the value of a named physical parameter.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Parameter name.
    /// value : float
    ///     New value.
    fn set_param(&mut self, name: &str, value: f64) -> PyResult<()> {
        get_pparam_mut(&mut self.inner, name)
            .map(|p| { p.value = value; })
            .ok_or_else(|| PyValueError::new_err(format!("unknown parameter: '{}'", name)))
    }

    /// Return the full Pparam metadata for a named parameter.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Parameter name.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Keys: ``"value"``, ``"range"``, ``"dstep"``, ``"vary"``, ``"defined"``.
    fn get_pparam<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let p = get_pparam_ref(&self.inner, name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown parameter: '{}'", name)))?;
        let d = PyDict::new(py);
        d.set_item("value", p.value)?;
        d.set_item("range", p.range)?;
        d.set_item("dstep", p.dstep)?;
        d.set_item("vary", p.vary)?;
        d.set_item("defined", p.defined)?;
        Ok(d)
    }

    /// Selectively update Pparam field(s) for a named parameter.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Parameter name.
    /// value : float, optional
    /// range : float, optional
    /// dstep : float, optional
    /// vary : bool, optional
    #[pyo3(signature = (name, *, value=None, range=None, dstep=None, vary=None))]
    fn set_pparam(
        &mut self,
        name: &str,
        value: Option<f64>,
        range: Option<f64>,
        dstep: Option<f64>,
        vary: Option<bool>,
    ) -> PyResult<()> {
        let p = get_pparam_mut(&mut self.inner, name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown parameter: '{}'", name)))?;
        if let Some(v) = value { p.value = v; }
        if let Some(r) = range { p.range = r; }
        if let Some(d) = dstep { p.dstep = d; }
        if let Some(v) = vary { p.vary = v; }
        Ok(())
    }

    /// Return metadata for all free (vary=True, defined=True) parameters.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     Each dict has keys ``"name"``, ``"value"``, ``"range"``, ``"dstep"``.
    fn get_free_params<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let free = collect_free_params(&self.inner);
        let mut out = Vec::with_capacity(free.len());
        for (name, p) in free {
            let d = PyDict::new(py);
            d.set_item("name", name)?;
            d.set_item("value", p.value)?;
            d.set_item("range", p.range)?;
            d.set_item("dstep", p.dstep)?;
            out.push(d);
        }
        Ok(out)
    }

    /// Batch-set multiple parameter values in one call.
    ///
    /// Parameters
    /// ----------
    /// names : list[str]
    ///     Parameter names.
    /// values : list[float]
    ///     Corresponding values.
    fn set_params(&mut self, names: Vec<String>, values: Vec<f64>) -> PyResult<()> {
        if names.len() != values.len() {
            return Err(PyValueError::new_err(
                format!("names and values must have same length ({} vs {})", names.len(), values.len())
            ));
        }
        for (name, val) in names.iter().zip(values.iter()) {
            get_pparam_mut(&mut self.inner, name)
                .map(|p| { p.value = *val; })
                .ok_or_else(|| PyValueError::new_err(format!("unknown parameter: '{}'", name)))?;
        }
        Ok(())
    }

    /// Deep copy of this model.
    ///
    /// Returns
    /// -------
    /// Model
    fn copy(&self) -> Self {
        Model { inner: self.inner.clone() }
    }

    // ---- Convenience properties for the most common parameters ----

    #[getter] fn q(&self) -> f64 { self.inner.q.value }
    #[setter] fn set_q(&mut self, v: f64) { self.inner.q.value = v; }
    #[getter] fn iangle(&self) -> f64 { self.inner.iangle.value }
    #[setter] fn set_iangle(&mut self, v: f64) { self.inner.iangle.value = v; }
    #[getter] fn r1(&self) -> f64 { self.inner.r1.value }
    #[setter] fn set_r1(&mut self, v: f64) { self.inner.r1.value = v; }
    #[getter] fn r2(&self) -> f64 { self.inner.r2.value }
    #[setter] fn set_r2(&mut self, v: f64) { self.inner.r2.value = v; }
    #[getter] fn t1(&self) -> f64 { self.inner.t1.value }
    #[setter] fn set_t1(&mut self, v: f64) { self.inner.t1.value = v; }
    #[getter] fn t2(&self) -> f64 { self.inner.t2.value }
    #[setter] fn set_t2(&mut self, v: f64) { self.inner.t2.value = v; }
    #[getter] fn period(&self) -> f64 { self.inner.period.value }
    #[setter] fn set_period(&mut self, v: f64) { self.inner.period.value = v; }
    #[getter] fn t0(&self) -> f64 { self.inner.t0.value }
    #[setter] fn set_t0(&mut self, v: f64) { self.inner.t0.value = v; }
    #[getter] fn velocity_scale(&self) -> f64 { self.inner.velocity_scale.value }
    #[setter] fn set_velocity_scale(&mut self, v: f64) { self.inner.velocity_scale.value = v; }

    /// Compute a light curve.
    ///
    /// Provide times in one of three ways:
    ///
    /// 1. ``time1``/``time2``/``ntime`` — generate evenly-spaced phases (default).
    /// 2. ``times`` — explicit array of times (numpy float64).
    /// 3. ``data`` — path to an lcurve data file.
    ///
    /// Parameters
    /// ----------
    /// time1 : float, optional
    ///     Start time (phase units). Default -0.2.
    /// time2 : float, optional
    ///     End time (phase units). Default 1.2.
    /// ntime : int, optional
    ///     Number of evenly-spaced points. Default 500.
    /// times : numpy.ndarray, optional
    ///     Explicit array of times. Overrides time1/time2/ntime.
    /// expose : float, optional
    ///     Exposure time per point (same units as period). Default 0.001.
    /// ndiv : int, optional
    ///     Number of sub-divisions for exposure smearing. Default 1.
    /// data : str, optional
    ///     Path to a data file. Overrides all time arguments.
    /// scale : bool, optional
    ///     Autoscale to minimize chi-squared. Default False.
    ///
    /// Returns
    /// -------
    /// LcResult
    #[pyo3(signature = (*, time1=-0.2, time2=1.2, ntime=500, times=None, expose=0.001, ndiv=1, data=None, scale=false))]
    fn light_curve(
        &self,
        py: Python<'_>,
        time1: f64,
        time2: f64,
        ntime: usize,
        times: Option<numpy::PyReadonlyArray1<'_, f64>>,
        expose: f64,
        ndiv: i32,
        data: Option<&str>,
        scale: bool,
    ) -> PyResult<LcResult> {
        let (datum_vec, rdata) = if let Some(data_path) = data {
            let d = lcurve::types::read_data(data_path)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (d, true)
        } else if let Some(time_arr) = times {
            let ts = time_arr.as_slice()
                .map_err(|e| PyValueError::new_err(format!("times must be contiguous: {}", e)))?;
            let d: Vec<lcurve::types::Datum> = ts.iter().map(|&t| lcurve::types::Datum {
                time: t, expose, flux: 0.0, ferr: 1.0, weight: 0.0, ndiv,
            }).collect();
            (d, false)
        } else {
            if ntime == 0 {
                return Err(PyValueError::new_err("ntime must be > 0"));
            }
            let d: Vec<lcurve::types::Datum> = (0..ntime).map(|i| {
                let t = if ntime == 1 {
                    (time1 + time2) / 2.0
                } else {
                    time1 + (time2 - time1) * i as f64 / (ntime - 1) as f64
                };
                lcurve::types::Datum {
                    time: t, expose, flux: 0.0, ferr: 1.0, weight: 0.0, ndiv,
                }
            }).collect();
            (d, false)
        };

        let time_vec: Vec<f64> = datum_vec.iter().map(|d| d.time).collect();

        let result = py.allow_threads(|| {
            lcurve::orchestration::light_curve_comp(&self.inner, &datum_vec, scale, rdata)
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(LcResult {
            times: time_vec,
            calc: result.calc,
            wdwarf: result.wdwarf,
            chisq: result.chisq,
            wnok: result.wnok,
            logg1: result.logg1,
            logg2: result.logg2,
            rv1: result.rv1,
            rv2: result.rv2,
            sfac: result.sfac,
        })
    }

    /// Evaluate chi-squared for a batch of parameter sets in parallel.
    ///
    /// Parameters
    /// ----------
    /// data : str
    ///     Path to an lcurve data file (read once for all evaluations).
    /// names : list[str]
    ///     Parameter names being varied.
    /// values : numpy.ndarray
    ///     2-D array of shape ``(N, len(names))`` with parameter values.
    /// scale : bool, optional
    ///     Autoscale each model to the data (default True).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     1-D array of length N with chi-squared values.
    ///     Unphysical parameter sets produce ``NaN``.
    #[pyo3(signature = (data, names, values, scale=true))]
    fn chisq_batch<'py>(
        &self,
        py: Python<'py>,
        data: &str,
        names: Vec<String>,
        values: numpy::PyReadonlyArray2<'_, f64>,
        scale: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let datum_vec = lcurve::types::read_data(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let arr = values.as_array();
        let (nsets, ncols) = (arr.nrows(), arr.ncols());
        if ncols != names.len() {
            return Err(PyValueError::new_err(format!(
                "values has {} columns but {} names were given", ncols, names.len()
            )));
        }

        // Flatten to row-major Vec<f64>
        let flat: Vec<f64> = arr.iter().copied().collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        let use_gpu = DEVICE_CHOICE.lock().map_or(false, |d| d.is_gpu());

        let result = py.allow_threads(|| {
            if use_gpu {
                chisq_batch_gpu_dispatch(&self.inner, &datum_vec, &name_refs, &flat, scale)
            } else {
                lcurve::orchestration::chisq_batch(
                    &self.inner, &datum_vec, &name_refs, &flat, scale,
                )
            }
        });

        debug_assert_eq!(result.len(), nsets);
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(q={}, iangle={}, r1={}, r2={}, t1={}, t2={})",
            self.inner.q.value, self.inner.iangle.value,
            self.inner.r1.value, self.inner.r2.value,
            self.inner.t1.value, self.inner.t2.value,
        )
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch — compiled only when the cuda feature is enabled
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn chisq_batch_gpu_dispatch(
    base: &lcurve::model::Model,
    data: &lcurve::types::Data,
    param_names: &[&str],
    param_values: &[f64],
    scale: bool,
) -> Vec<f64> {
    use std::sync::OnceLock;

    static GPU_CTX: OnceLock<Mutex<lcurve_cuda::CudaContext>> = OnceLock::new();

    let device_id = DEVICE_CHOICE
        .lock()
        .map(|d| match &*d {
            DeviceChoice::Cuda(id) => *id,
            _ => 0,
        })
        .unwrap_or(0);

    let ctx_mutex = GPU_CTX.get_or_init(|| {
        Mutex::new(
            lcurve_cuda::CudaContext::new(device_id)
                .expect("Failed to initialise CUDA context"),
        )
    });

    let mut ctx = match ctx_mutex.lock() {
        Ok(guard) => guard,
        Err(_) => {
            return lcurve::orchestration::chisq_batch(base, data, param_names, param_values, scale);
        }
    };

    match lcurve_cuda::chisq_batch_gpu(&mut ctx, base, data, param_names, param_values, scale) {
        Ok(results) => results,
        Err(_) => {
            // Graceful fallback to CPU
            lcurve::orchestration::chisq_batch(base, data, param_names, param_values, scale)
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn chisq_batch_gpu_dispatch(
    base: &lcurve::model::Model,
    data: &lcurve::types::Data,
    param_names: &[&str],
    param_values: &[f64],
    scale: bool,
) -> Vec<f64> {
    // Should not be reached — device check prevents this — but fall back gracefully
    lcurve::orchestration::chisq_batch(base, data, param_names, param_values, scale)
}

// ---------------------------------------------------------------------------
// Module-level device functions
// ---------------------------------------------------------------------------

/// Set the compute device: "cpu", "cuda", or "cuda:N".
#[pyfunction]
fn _set_device(device: &str) -> PyResult<()> {
    let choice = DeviceChoice::parse(device)
        .map_err(|e| PyValueError::new_err(e))?;

    // If requesting CUDA, check that the feature is compiled in
    if choice.is_gpu() {
        #[cfg(not(feature = "cuda"))]
        {
            return Err(PyRuntimeError::new_err(
                "lcurve_rs was built without CUDA support. \
                 Rebuild with: maturin develop --release --features cuda"
            ));
        }
    }

    let mut guard = DEVICE_CHOICE
        .lock()
        .map_err(|_| PyRuntimeError::new_err("device lock poisoned"))?;
    *guard = choice;
    Ok(())
}

/// Return current device string: "cpu" or "cuda:N".
#[pyfunction]
fn _get_device() -> PyResult<String> {
    let guard = DEVICE_CHOICE
        .lock()
        .map_err(|_| PyRuntimeError::new_err("device lock poisoned"))?;
    Ok(guard.to_string())
}

/// Return True if the CUDA feature is compiled in.
#[pyfunction]
fn _has_cuda() -> bool {
    cfg!(feature = "cuda")
}

/// lcurve_rs — Python bindings for the Rust lcurve light curve engine.
#[pymodule]
fn _lcurve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<LcResult>()?;
    m.add_function(wrap_pyfunction!(_set_device, m)?)?;
    m.add_function(wrap_pyfunction!(_get_device, m)?)?;
    m.add_function(wrap_pyfunction!(_has_cuda, m)?)?;
    Ok(())
}
