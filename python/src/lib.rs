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

    /// Create a white dwarf binary (CV) model with sensible defaults.
    ///
    /// Sets up a cataclysmic variable: WD primary + Roche-filling
    /// M-dwarf secondary, with optional accretion disc.
    ///
    /// Parameters
    /// ----------
    /// q : float
    ///     Mass ratio M2/M1 (default 0.5).
    /// iangle : float
    ///     Orbital inclination in degrees (default 82).
    /// t1 : float
    ///     WD temperature in K (default 15000).
    /// t2 : float
    ///     Secondary temperature in K (default 3500).
    /// period : float
    ///     Orbital period in days (default 0.1).
    /// r1 : float
    ///     WD radius as fraction of separation (default 0.015).
    /// disc : bool
    ///     Include accretion disc (default False).
    #[staticmethod]
    #[pyo3(signature = (q=0.5, iangle=82.0, t1=15000.0, t2=3500.0, period=0.1, r1=0.015, disc=false))]
    fn whitedwarf(
        q: f64, iangle: f64, t1: f64, t2: f64, period: f64, r1: f64, disc: bool,
    ) -> PyResult<Self> {
        // Build a model file string with CV defaults
        let disc_r1 = if disc { 0.2 } else { 0.0 };
        let disc_r2 = if disc { 0.45 } else { 0.0 };
        let disc_temp = if disc { 8000.0 } else { 0.0 };
        let model_str = format!(
            r#"q                    = {q} 0.1 0.01 0
iangle               = {iangle} 2.0 0.1 0
r1                   = {r1} 0.005 0.001 0
r2                   = -1 0.01 0.001 0
cphi3                = 0.015 0.001 0.001 0
cphi4                = 0.017 0.001 0.001 0
t1                   = {t1} 500 100 0
t2                   = {t2} 200 50 0
spin1                = 1 0.001 0.001 0
spin2                = 1 0.001 0.001 0
ldc1_1               = 0.4 0.01 0.01 0
ldc1_2               = 0.0 0.01 0.01 0
ldc1_3               = 0.0 0.01 0.01 0
ldc1_4               = 0.0 0.01 0.01 0
ldc2_1               = 0.6 0.01 0.01 0
ldc2_2               = 0.0 0.01 0.01 0
ldc2_3               = 0.0 0.01 0.01 0
ldc2_4               = 0.0 0.01 0.01 0
velocity_scale       = 0 1 1 0
beam_factor1         = 0 0.1 0.02 0
beam_factor2         = 0 0.1 0.002 0
deltat               = 0 0.001 0.001 0
t0                   = 0.0 0.0001 1e-05 0
period               = {period} 1e-06 1e-06 0
gravity_dark1        = 0.25 0.0001 0.0001 0
gravity_dark2        = 0.08 0.0001 0.0001 0
absorb               = 1.0 0.001 0.001 0
slope                = 0 0.001 0.0001 0
quad                 = 0 0.001 0.0001 0
cube                 = 0 0.001 0.0001 0
third                = 0 0.001 0.0001 0
rdisc1               = {disc_r1} 0.1 0.01 0
rdisc2               = {disc_r2} 0.1 0.01 0
height_disc          = 0.02 0.01 0.001 0
beta_disc            = 2.0 0.1 0.1 0
temp_disc            = {disc_temp} 500 100 0
texp_disc            = -0.75 0.1 0.01 0
lin_limb_disc        = 0.3 0.1 0.01 0
quad_limb_disc       = 0.0 0.1 0.01 0
temp_edge            = 0.0 500 100 0
absorb_edge          = 0.0 0.001 0.001 0
radius_spot          = 0.0 0.001 0.001 0
length_spot          = 0.0 0.001 0.001 0
height_spot          = 0.0 0.001 0.001 0
expon_spot           = 0.0 0.001 0.001 0
epow_spot            = 1.0 0.001 0.001 0
angle_spot           = 0.0 0.001 0.001 0
yaw_spot             = 0.0 0.001 0.001 0
temp_spot            = 0.0 0.001 0.001 0
tilt_spot            = 90.0 0.001 0.001 0
cfrac_spot           = 0.0 0.001 0.001 0
stsp11_long          = 0 1 1 0 0
stsp11_lat           = 0 1 1 0 0
stsp11_fwhm          = 0 1 1 0 0
stsp11_tcen          = 0 1 1 0 0
stsp12_long          = 0 1 1 0 0
stsp12_lat           = 0 1 1 0 0
stsp12_fwhm          = 0 1 1 0 0
stsp12_tcen          = 0 1 1 0 0
stsp13_long          = 0 1 1 0 0
stsp13_lat           = 0 1 1 0 0
stsp13_fwhm          = 0 1 1 0 0
stsp13_tcen          = 0 1 1 0 0
stsp21_long          = 0 1 1 0 0
stsp21_lat           = 0 1 1 0 0
stsp21_fwhm          = 0 1 1 0 0
stsp21_tcen          = 0 1 1 0 0
stsp22_long          = 0 1 1 0 0
stsp22_lat           = 0 1 1 0 0
stsp22_fwhm          = 0 1 1 0 0
stsp22_tcen          = 0 1 1 0 0
uesp_long1           = 0 1 1 0 0
uesp_long2           = 0 1 1 0 0
uesp_lathw           = 0 1 1 0 0
uesp_taper           = 0 1 1 0 0
uesp_temp            = 0 1 1 0 0
delta_phase          = -0.001
nlat1f               = 50
nlat2f               = 50
nlat1c               = 50
nlat2c               = 50
delta_phase          = -0.001
nlat1f               = 50
nlat2f               = 50
nlat1c               = 50
nlat2c               = 50
npole                = 0
nlatfill             = 4
nlngfill             = 4
lfudge               = 0.05
llo                  = 0.0
lhi                  = 0.0
wavelength           = 4700
roche1               = 1
roche2               = 1
eclipse1             = 1
eclipse2             = 1
glens1               = 0
use_radii            = 0
tperiod              = 0
gdark_bolom1         = 0
gdark_bolom2         = 0
mucrit1              = 0.0
mucrit2              = 0.0
limb1                = Claret
limb2                = Claret
mirror               = 0
add_disc             = {add_disc}
opaque               = 1
add_spot             = 0
nspot                = 0
nrad                 = 50
phase1               = -0.2
phase2               = 1.2
iscale               = 1
"#,
            q = q, iangle = iangle, t1 = t1, t2 = t2, period = period,
            r1 = r1, disc_r1 = disc_r1, disc_r2 = disc_r2,
            disc_temp = disc_temp,
            add_disc = if disc { 1 } else { 0 },
        );

        // Write to a temp file, load the model, clean up
        let tmp_path = "/tmp/_lcurve_rs_whitedwarf_tmp.dat";
        std::fs::write(tmp_path, &model_str)
            .map_err(|e| PyRuntimeError::new_err(format!("write tmpfile: {}", e)))?;
        let result = lcurve::model::Model::from_file(tmp_path);
        let _ = std::fs::remove_file(tmp_path);
        let inner = result.map_err(|e| PyValueError::new_err(format!("parse model: {}", e)))?;

        Ok(Model { inner })
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
// ---------------------------------------------------------------------------
// phoebe-rs: Eclipsing binary light curve synthesizer
// ---------------------------------------------------------------------------

/// Eclipsing binary parameters.
#[pyclass(name = "EBParams")]
#[derive(Clone)]
struct PyEBParams {
    inner: phoebe_rs::EBParams,
}

#[pymethods]
impl PyEBParams {
    /// Create a contact binary (W UMa) system.
    ///
    /// Args:
    ///     q: mass ratio M2/M1
    ///     inclination: orbital inclination in degrees (90 = edge-on)
    ///     t1: effective temperature of primary (K), default 6000
    ///     t2: effective temperature of secondary (K), default 5500
    ///     period: orbital period in days, default 0.35
    ///     ld1: limb darkening coefficient for primary, default 0.5
    ///     ld2: limb darkening coefficient for secondary, default 0.5
    ///     l3: third light fraction, default 0.0
    ///     n_grid: surface grid resolution, default 40
    #[staticmethod]
    #[pyo3(signature = (q, inclination, t1=6000.0, t2=5500.0, period=0.35, fillout=1.0, ld1=0.5, ld2=0.5, l3=0.0, phi0=0.0, n_grid=40, passband="bolometric"))]
    fn contact(q: f64, inclination: f64, t1: f64, t2: f64, period: f64,
               fillout: f64, ld1: f64, ld2: f64, l3: f64, phi0: f64,
               n_grid: usize, passband: &str) -> Self {
        let mut p = phoebe_rs::EBParams::contact(q, inclination);
        p.t_eff1 = t1;
        p.t_eff2 = t2;
        p.period = period;
        p.fillout1 = fillout;
        p.fillout2 = fillout;
        p.ld1 = ld1;
        p.ld2 = ld2;
        p.l3 = l3;
        p.phi0 = phi0;
        p.n_grid = n_grid;
        p.passband = phoebe_rs::Passband::from_str(passband)
            .unwrap_or(phoebe_rs::Passband::Bolometric);
        Self { inner: p }
    }

    /// Create a detached binary system.
    ///
    /// Args:
    ///     q: mass ratio M2/M1
    ///     inclination: orbital inclination in degrees
    ///     r1_frac: Roche lobe filling factor for primary (0-1)
    ///     r2_frac: Roche lobe filling factor for secondary (0-1)
    ///     t1, t2, period, ld1, ld2, l3, n_grid: as for contact()
    #[staticmethod]
    #[pyo3(signature = (q, inclination, r1_frac, r2_frac, t1=6000.0, t2=5500.0, period=1.0, phi0=0.0, ld1=0.5, ld2=0.5, l3=0.0, n_grid=40, passband="bolometric"))]
    fn detached(q: f64, inclination: f64, r1_frac: f64, r2_frac: f64,
                t1: f64, t2: f64, period: f64, phi0: f64,
                ld1: f64, ld2: f64, l3: f64, n_grid: usize, passband: &str) -> Self {
        let mut p = phoebe_rs::EBParams::detached(q, inclination, r1_frac, r2_frac);
        p.t_eff1 = t1;
        p.t_eff2 = t2;
        p.period = period;
        p.phi0 = phi0;
        p.ld1 = ld1;
        p.ld2 = ld2;
        p.l3 = l3;
        p.passband = phoebe_rs::Passband::from_str(passband)
            .unwrap_or(phoebe_rs::Passband::Bolometric);
        p.n_grid = n_grid;
        Self { inner: p }
    }

    fn __repr__(&self) -> String {
        format!("EBParams(q={}, i={}, T1={}, T2={}, P={})",
                self.inner.q, self.inner.inclination,
                self.inner.t_eff1, self.inner.t_eff2, self.inner.period)
    }
}

/// Compute an eclipsing binary light curve.
///
/// Args:
///     params: EBParams object
///     phases: numpy array of orbital phases (0 to 1)
///     method: "numerical" (default, full Roche mesh) or "analytic" (fast
///             Eggleton + circle-circle, ~350x faster, suitable for MCMC)
///
/// Returns:
///     dict with keys 'phases', 'flux' (and 'flux1', 'flux2' for numerical)
#[pyfunction]
#[pyo3(signature = (params, phases, method="numerical"))]
fn eb_lightcurve<'py>(
    py: Python<'py>,
    params: &PyEBParams,
    phases: numpy::PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let ph = phases.as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("phases array: {}", e)))?;

    let dict = PyDict::new(py);

    match method {
        "analytic" => {
            let lc = phoebe_rs::compute_analytic(&params.inner, ph);
            dict.set_item("phases", lc.phases.into_pyarray(py))?;
            dict.set_item("flux", lc.flux.into_pyarray(py))?;
        },
        "numerical" | _ => {
            let lc = phoebe_rs::compute_lightcurve(&params.inner, ph)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
            dict.set_item("phases", lc.phases.into_pyarray(py))?;
            dict.set_item("flux", lc.flux.into_pyarray(py))?;
            dict.set_item("flux1", lc.flux1.into_pyarray(py))?;
            dict.set_item("flux2", lc.flux2.into_pyarray(py))?;
        },
    }

    Ok(dict)
}

/// Search for best-fit mass ratio by grid search.
///
/// Computes light curves for a grid of q values and returns the
/// chi-squared for each, given observed phases, magnitudes, and errors.
///
/// Args:
///     phases: observed orbital phases
///     mags: observed magnitudes
///     errs: magnitude uncertainties
///     inclination: fixed inclination (degrees)
///     q_min: minimum mass ratio to search
///     q_max: maximum mass ratio to search
///     n_q: number of q values in grid
///     t1: primary temperature (K)
///     t2: secondary temperature (K)
///     n_grid: surface grid resolution
///
/// Returns:
///     dict with 'q_grid', 'chi2', 'best_q', 'best_chi2'
#[pyfunction]
#[pyo3(signature = (phases, mags, errs, inclination, q_min=0.05, q_max=1.0, n_q=50, t1=6000.0, t2=5500.0, n_grid=30))]
fn q_search<'py>(
    py: Python<'py>,
    phases: numpy::PyReadonlyArray1<'py, f64>,
    mags: numpy::PyReadonlyArray1<'py, f64>,
    errs: numpy::PyReadonlyArray1<'py, f64>,
    inclination: f64,
    q_min: f64,
    q_max: f64,
    n_q: usize,
    t1: f64,
    t2: f64,
    n_grid: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ph = phases.as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
    let mg = mags.as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
    let er = errs.as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

    let n = ph.len();
    if mg.len() != n || er.len() != n {
        return Err(PyValueError::new_err("phases, mags, errs must have same length"));
    }

    // Convert mags to flux (relative)
    let mag_ref = mg.iter().cloned().fold(f64::INFINITY, f64::min);
    let obs_flux: Vec<f64> = mg.iter().map(|m| 10f64.powf(-0.4 * (m - mag_ref))).collect();
    let obs_flux_err: Vec<f64> = er.iter().enumerate()
        .map(|(i, e)| obs_flux[i] * 0.4 * std::f64::consts::LN_10 * e)
        .collect();

    let q_grid: Vec<f64> = (0..n_q)
        .map(|i| q_min + (q_max - q_min) * i as f64 / (n_q - 1) as f64)
        .collect();

    let mut chi2_grid = Vec::with_capacity(n_q);
    let mut best_q = q_min;
    let mut best_chi2 = f64::INFINITY;

    for &q in &q_grid {
        let mut params = phoebe_rs::EBParams::contact(q, inclination);
        params.t_eff1 = t1;
        params.t_eff2 = t2;
        params.n_grid = n_grid;

        let lc = match phoebe_rs::compute_lightcurve(&params, ph) {
            Ok(lc) => lc,
            Err(_) => {
                chi2_grid.push(f64::INFINITY);
                continue;
            }
        };

        // Scale model flux to best match observed
        let mut sum_wxy = 0.0;
        let mut sum_wxx = 0.0;
        for i in 0..n {
            let w = 1.0 / (obs_flux_err[i] * obs_flux_err[i] + 1e-30);
            sum_wxy += w * obs_flux[i] * lc.flux[i];
            sum_wxx += w * lc.flux[i] * lc.flux[i];
        }
        let scale = if sum_wxx > 0.0 { sum_wxy / sum_wxx } else { 1.0 };

        let chi2: f64 = (0..n)
            .map(|i| {
                let resid = obs_flux[i] - scale * lc.flux[i];
                let w = 1.0 / (obs_flux_err[i] * obs_flux_err[i] + 1e-30);
                w * resid * resid
            })
            .sum();

        chi2_grid.push(chi2);
        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_q = q;
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("q_grid", q_grid.into_pyarray(py))?;
    dict.set_item("chi2", chi2_grid.into_pyarray(py))?;
    dict.set_item("best_q", best_q)?;
    dict.set_item("best_chi2", best_chi2)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// Ellipsoidal variability (ELL) analytical model
// Morris & Naftilan (1993), as used by Rowan et al. (2021)
// ---------------------------------------------------------------------------

/// Compute an analytical ellipsoidal light curve.
///
/// The model is a 3-term Fourier series with amplitudes set by the
/// tidal distortion physics (reflection, ellipsoidal, Doppler beaming):
///
///   ΔL/L̄ = A₁ cos(φ) + A₂ cos(2φ) + A₃ cos(3φ)
///
/// In the fitting mode (physical=false), A₁, A₂, A₃ are free parameters.
/// In the physical mode (physical=true), they are computed from q, i, R*/a,
/// and limb/gravity darkening coefficients u, τ.
///
/// Args:
///     phases: orbital phases (0 to 1)
///     a1, a2, a3: Fourier amplitudes (fitting mode)
///     mean_mag: mean magnitude offset
///
/// Returns: dict with 'phases', 'mags'
#[pyfunction]
#[pyo3(signature = (phases, a1, a2, a3, mean_mag=0.0))]
fn ell_lightcurve<'py>(
    py: Python<'py>,
    phases: numpy::PyReadonlyArray1<'py, f64>,
    a1: f64,
    a2: f64,
    a3: f64,
    mean_mag: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let ph = phases.as_slice()?;
    let n = ph.len();

    let mut mags = Vec::with_capacity(n);
    for &phi in ph {
        let angle = 2.0 * std::f64::consts::PI * phi;
        let flux_frac = a1 * angle.cos()
            + a2 * (2.0 * angle).cos()
            + a3 * (3.0 * angle).cos();
        // Convert fractional flux change to magnitude
        let mag = mean_mag - 2.5 * (1.0 + flux_frac).log10();
        mags.push(mag);
    }

    let dict = PyDict::new(py);
    dict.set_item("phases", ph.to_vec().into_pyarray(py))?;
    dict.set_item("mags", mags.into_pyarray(py))?;
    Ok(dict)
}

/// Fit the ELL analytical model to phase-folded magnitudes.
///
/// Fits ΔL/L̄ = A₁ cos(φ) + A₂ cos(2φ) + A₃ cos(3φ) + offset
/// using weighted least squares.
///
/// Also fits a simple cosine: m = B cos(2φ) + offset
/// and computes R = χ²_ell / χ²_cos (Rowan et al. 2021 selection criterion).
///
/// Args:
///     phases: orbital phases (0 to 1)
///     mags: observed magnitudes
///     errs: magnitude uncertainties
///
/// Returns: dict with 'a1', 'a2', 'a3', 'chi2_ell', 'chi2_cos', 'R',
///          'ell_model', 'cos_model'
#[pyfunction]
fn fit_ell<'py>(
    py: Python<'py>,
    phases: numpy::PyReadonlyArray1<'py, f64>,
    mags: numpy::PyReadonlyArray1<'py, f64>,
    errs: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ph = phases.as_slice()?;
    let mag = mags.as_slice()?;
    let err = errs.as_slice()?;
    let n = ph.len();

    if n < 5 {
        return Err(PyValueError::new_err("Need at least 5 data points"));
    }

    let pi2 = 2.0 * std::f64::consts::PI;

    // --- ELL model: m = c0 + a1*cos(φ) + a2*cos(2φ) + a3*cos(3φ) ---
    // Weighted least squares: X^T W X β = X^T W y
    // 4 parameters: c0, a1, a2, a3

    let mut xtwx = [[0.0f64; 4]; 4];
    let mut xtwy = [0.0f64; 4];

    for i in 0..n {
        let w = 1.0 / (err[i] * err[i]);
        let angle = pi2 * ph[i];
        let basis = [1.0, angle.cos(), (2.0 * angle).cos(), (3.0 * angle).cos()];

        for r in 0..4 {
            for c in 0..4 {
                xtwx[r][c] += w * basis[r] * basis[c];
            }
            xtwy[r] += w * basis[r] * mag[i];
        }
    }

    // Solve 4x4 system (Gaussian elimination)
    let mut aug = [[0.0f64; 5]; 4];
    for r in 0..4 {
        for c in 0..4 {
            aug[r][c] = xtwx[r][c];
        }
        aug[r][4] = xtwy[r];
    }

    for col in 0..4 {
        // Pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            return Err(PyRuntimeError::new_err("Singular matrix in ELL fit"));
        }

        for c in col..5 {
            aug[col][c] /= pivot;
        }
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for c in col..5 {
                aug[row][c] -= factor * aug[col][c];
            }
        }
    }

    let c0_ell = aug[0][4];
    let a1 = aug[1][4];
    let a2 = aug[2][4];
    let a3 = aug[3][4];

    // Compute chi2_ell and model
    let mut chi2_ell = 0.0;
    let mut ell_model = Vec::with_capacity(n);
    for i in 0..n {
        let angle = pi2 * ph[i];
        let pred = c0_ell + a1 * angle.cos() + a2 * (2.0 * angle).cos() + a3 * (3.0 * angle).cos();
        ell_model.push(pred);
        let resid = (mag[i] - pred) / err[i];
        chi2_ell += resid * resid;
    }

    // --- Sinusoidal model: m = c0 + b1*cos(φ) + b2*sin(φ)  (Rowan+ 2021) ---
    // 3 parameters: c0, b1, b2  (fundamental frequency with free phase)
    let mut xtwx2 = [[0.0f64; 3]; 3];
    let mut xtwy2 = [0.0f64; 3];

    for i in 0..n {
        let w = 1.0 / (err[i] * err[i]);
        let angle = pi2 * ph[i];
        let basis = [1.0, angle.cos(), angle.sin()];
        for r in 0..3 {
            for c in 0..3 {
                xtwx2[r][c] += w * basis[r] * basis[c];
            }
            xtwy2[r] += w * basis[r] * mag[i];
        }
    }

    // Solve 3x3 via Gaussian elimination
    let mut aug2 = [[0.0f64; 4]; 3];
    for r in 0..3 {
        for c in 0..3 {
            aug2[r][c] = xtwx2[r][c];
        }
        aug2[r][3] = xtwy2[r];
    }
    for col in 0..3 {
        let mut max_row = col;
        let mut max_val = aug2[col][col].abs();
        for row in (col + 1)..3 {
            if aug2[row][col].abs() > max_val {
                max_val = aug2[row][col].abs();
                max_row = row;
            }
        }
        aug2.swap(col, max_row);
        let pivot = aug2[col][col];
        if pivot.abs() < 1e-30 {
            return Err(PyRuntimeError::new_err("Singular matrix in cosine fit"));
        }
        for c in col..4 {
            aug2[col][c] /= pivot;
        }
        for row in 0..3 {
            if row == col { continue; }
            let factor = aug2[row][col];
            for c in col..4 {
                aug2[row][c] -= factor * aug2[col][c];
            }
        }
    }
    let c0_cos = aug2[0][3];
    let b1_cos = aug2[1][3];
    let b2_cos = aug2[2][3];

    let mut chi2_cos = 0.0;
    let mut cos_model = Vec::with_capacity(n);
    for i in 0..n {
        let angle = pi2 * ph[i];
        let pred = c0_cos + b1_cos * angle.cos() + b2_cos * angle.sin();
        cos_model.push(pred);
        let resid = (mag[i] - pred) / err[i];
        chi2_cos += resid * resid;
    }

    let r_ratio = if chi2_cos > 0.0 {
        chi2_ell / chi2_cos
    } else {
        f64::INFINITY
    };

    let dict = PyDict::new(py);
    dict.set_item("a1", a1)?;
    dict.set_item("a2", a2)?;
    dict.set_item("a3", a3)?;
    dict.set_item("c0_ell", c0_ell)?;
    dict.set_item("b1_cos", b1_cos)?;
    dict.set_item("b2_cos", b2_cos)?;
    dict.set_item("chi2_ell", chi2_ell)?;
    dict.set_item("chi2_cos", chi2_cos)?;
    dict.set_item("R", r_ratio)?;
    dict.set_item("ell_model", ell_model.into_pyarray(py))?;
    dict.set_item("cos_model", cos_model.into_pyarray(py))?;
    Ok(dict)
}

#[pymodule]
fn _lcurve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<LcResult>()?;
    m.add_class::<PyEBParams>()?;
    m.add_function(wrap_pyfunction!(_set_device, m)?)?;
    m.add_function(wrap_pyfunction!(_get_device, m)?)?;
    m.add_function(wrap_pyfunction!(_has_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(eb_lightcurve, m)?)?;
    m.add_function(wrap_pyfunction!(q_search, m)?)?;
    m.add_function(wrap_pyfunction!(ell_lightcurve, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ell, m)?)?;
    Ok(())
}
