# API Reference

## Python API (`lcurve_rs`)

### `lcurve_rs.Model`

```python
class Model:
    """Wrapper around the Rust light curve model."""

    def __init__(self, path: str) -> None:
        """Load a model from an lcurve parameter file.

        Parameters
        ----------
        path : str
            Path to the model parameter file.
        """

    def light_curve(
        self,
        *,
        time1: float = -0.2,
        time2: float = 1.2,
        ntime: int = 500,
        times: numpy.ndarray | None = None,
        expose: float = 0.001,
        ndiv: int = 1,
        data: str | None = None,
        scale: bool = False,
    ) -> LcResult:
        """Compute a synthetic light curve.

        Provide times in one of three ways:

        1. ``time1``/``time2``/``ntime`` — evenly-spaced phases (default).
        2. ``times`` — explicit numpy float64 array.
        3. ``data`` — path to an lcurve data file.

        Parameters
        ----------
        time1 : float
            Start time (phase units). Default -0.2.
        time2 : float
            End time (phase units). Default 1.2.
        ntime : int
            Number of evenly-spaced points. Default 500.
        times : numpy.ndarray, optional
            Explicit array of times. Overrides time1/time2/ntime.
        expose : float
            Exposure time per point. Default 0.001.
        ndiv : int
            Sub-divisions for exposure smearing. Default 1.
        data : str, optional
            Path to a data file. Overrides all time arguments.
        scale : bool
            Autoscale to minimize chi-squared. Default False.

        Returns
        -------
        LcResult
        """

    def get_param(self, name: str) -> float:
        """Get value of a named physical parameter."""

    def set_param(self, name: str, value: float) -> None:
        """Set value of a named physical parameter."""

    def get_pparam(self, name: str) -> dict:
        """Return full Pparam metadata for a named parameter.

        Returns
        -------
        dict
            Keys: "value", "range", "dstep", "vary", "defined".
        """

    def set_pparam(
        self,
        name: str,
        *,
        value: float | None = None,
        range: float | None = None,
        dstep: float | None = None,
        vary: bool | None = None,
    ) -> None:
        """Selectively update Pparam field(s) for a named parameter."""

    def get_free_params(self) -> list[dict]:
        """Return metadata for all free (vary=True) parameters.

        Returns
        -------
        list[dict]
            Each dict has keys: "name", "value", "range", "dstep".
        """

    def set_params(self, names: list[str], values: list[float]) -> None:
        """Batch-set multiple parameter values in one Rust call."""

    def copy(self) -> "Model":
        """Deep copy of this model."""

    # Properties: q, iangle, r1, r2, t1, t2, period, t0, velocity_scale
```

### `lcurve_rs.LcResult`

```python
class LcResult:
    """Result of a light curve computation."""

    times: numpy.ndarray    # Time values (float64)
    flux: numpy.ndarray     # Computed fluxes (float64), alias for calc
    calc: numpy.ndarray     # Computed fluxes (float64)
    wdwarf: float           # White dwarf contribution at phase 0.5
    chisq: float            # Weighted chi-squared
    wnok: float             # Weighted number of OK points
    logg1: float            # Flux-weighted log10(g) for star 1 (CGS)
    logg2: float            # Flux-weighted log10(g) for star 2 (CGS)
    rv1: float              # Volume-averaged radius of star 1
    rv2: float              # Volume-averaged radius of star 2
    sfac: numpy.ndarray     # Scale factors [star1, disc, edge, spot, star2]
```

## Physical Parameters

All 65 physical parameters from the lcurve model file are accessible via `get_param()`/`set_param()`. These include:

### Binary system

| Parameter | Description |
|-----------|-------------|
| `q` | Mass ratio (M2/M1) |
| `iangle` | Orbital inclination (degrees) |
| `r1` | Radius of star 1 (units of separation) |
| `r2` | Radius of star 2 (negative = Roche-filling) |
| `spin1`, `spin2` | Spin-to-orbital frequency ratios |
| `t1`, `t2` | Stellar temperatures (K); negative t2 = no irradiation from star 2 |
| `velocity_scale` | Velocity scale (km/s) |
| `period` | Orbital period (days) |
| `t0` | Ephemeris zero point |

### Limb darkening

| Parameter | Description |
|-----------|-------------|
| `ldc1_1` .. `ldc1_4` | Star 1 limb darkening coefficients |
| `ldc2_1` .. `ldc2_4` | Star 2 limb darkening coefficients |

### Disc

| Parameter | Description |
|-----------|-------------|
| `rdisc1`, `rdisc2` | Inner/outer disc radii |
| `height_disc` | Disc half-thickness |
| `beta_disc` | Disc flaring exponent |
| `temp_disc` | Disc temperature at outer edge |
| `texp_disc` | Disc temperature exponent |

### Bright spot

| Parameter | Description |
|-----------|-------------|
| `radius_spot` | Spot radius |
| `length_spot` | Spot length |
| `angle_spot` | Spot angle (degrees) |
| `temp_spot` | Spot temperature |

### Other

| Parameter | Description |
|-----------|-------------|
| `gravity_dark1`, `gravity_dark2` | Gravity darkening exponents |
| `absorb` | Irradiation absorption efficiency |
| `third` | Third-light contribution |
| `slope`, `quad`, `cube` | Polynomial fudge factors |

## Fitting API (`lcurve_rs.fitting`)

### `lcurve_rs.fitting.Prior`

```python
@dataclass
class Prior:
    name: str
    low: float
    high: float
    transform: Callable | None   # unit cube [0,1] -> param (nested sampling)
    log_prob: Callable | None    # log-probability (MCMC)

    @staticmethod
    def uniform(name: str, low: float, high: float) -> Prior: ...

    @staticmethod
    def gaussian(name: str, mean: float, sigma: float,
                 low: float, high: float) -> Prior: ...
```

### `lcurve_rs.fitting.FitResult`

```python
@dataclass
class FitResult:
    backend: str                     # "ultranest", "dynesty", or "emcee"
    param_names: list[str]
    samples: np.ndarray              # (n_samples, n_params)
    log_likelihood: np.ndarray       # (n_samples,)
    weights: np.ndarray | None       # nested sampling only
    log_evidence: float | None       # nested sampling only
    log_evidence_err: float | None
    raw_result: Any                  # backend-specific object

    @property
    def median(self) -> dict[str, float]: ...
    @property
    def best_fit(self) -> dict[str, float]: ...
    def summary(self, ci: float = 68.27) -> str: ...
```

### `lcurve_rs.fitting.Fitter`

```python
class Fitter:
    def __init__(
        self,
        model: Model,
        data: str,
        scale: bool = True,
        priors: dict[str, Prior] | None = None,
        params: list[str] | None = None,
    ) -> None: ...

    param_names: list[str]
    ndim: int

    def log_likelihood(self, theta) -> float: ...
    def prior_transform(self, cube) -> np.ndarray: ...
    def log_prior(self, theta) -> float: ...
    def log_probability(self, theta) -> float: ...

    def run_emcee(
        self,
        nwalkers: int = 0,
        nsteps: int = 5000,
        burn: int = 1000,
        **kwargs,
    ) -> FitResult: ...

    def run_ultranest(self, **kwargs) -> FitResult: ...

    def run_dynesty(self, nlive: int = 500, **kwargs) -> FitResult: ...
```

## CLI Reference

```
lroche <MODEL> <DATA> [OPTIONS]

Arguments:
  <MODEL>    Model parameter file
  <DATA>     Data file (use 'none' to generate times)

Options:
  --time1 <FLOAT>     Start time [default: -0.2]
  --time2 <FLOAT>     End time [default: 0.8]
  --ntime <INT>       Number of times [default: 500]
  --expose <FLOAT>    Exposure time [default: 0.001]
  --ndivide <INT>     Exposure sub-divisions [default: 1]
  -o, --output <FILE> Output file
  --scale             Autoscale to minimize chi-squared
```
