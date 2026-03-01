# lcurve-rs

A Rust port of Tom Marsh's C++ [lcurve](http://github.com/trmrsh/cpp-lcurve/tree/master) and associated packages (`trm.roche`, `trm.subs`) for computing synthetic light curves of eclipsing white-dwarf binaries. All Roche geometry, eclipse computation, grid generation, and model-parameter handling are faithful ports of the original C++ code, rewritten in Rust for memory safety, multi-core parallelism, and easy installation.

Includes a drop-in CLI replacement (`lroche`) and Python bindings (`lcurve_rs`).

## Features

| Feature | Description |
|---------|-------------|
| Roche geometry | Roche-lobe-filling stars with gravity darkening, limb darkening (polynomial + Claret 4-coeff) |
| Accretion disc | Opaque/transparent disc with power-law temperature profile, bright spot, disc edge |
| Star spots | Up to 3 Gaussian spots on the primary, 2 on the secondary, plus uniform equatorial spots |
| Irradiation | Mutual irradiation between stars with absorption efficiency |
| Eclipses | Primary/secondary eclipses with disc occultation |
| Exposure smearing | Trapezoid-rule integration over finite exposures |
| Gravitational lensing | Optional lensing magnification from the primary |
| Parallel computation | Rayon-based multithreading across time points and grid elements |
| Python bindings | PyO3/maturin bindings with numpy integration |
| Parameter estimation | Fitting via emcee (MCMC), UltraNest, or dynesty (nested sampling) |

## Installing

### CLI binary

Requires a [Rust toolchain](https://rustup.rs/):

```bash
cd lcurve-rs
cargo build --release
# Binary at target/release/lroche
```

### Python bindings

Requires Rust and [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin numpy
cd lcurve-rs/python
maturin develop --release
```

This builds the `lcurve_rs` Python package with Rayon multithreading.

## CLI Usage

```bash
# Generate a light curve from a model file (500 evenly-spaced phases)
lroche model.dat none --time1=-0.2 --time2=1.2 --ntime=500 --output=lc.dat

# Fit to observed data with autoscaling
lroche model.dat data.dat --scale --output=fit.dat

# Custom exposure smearing
lroche model.dat none --ntime=1000 --expose=0.01 --ndivide=5 --output=lc.dat
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--time1` | -0.2 | Start time (phase units) |
| `--time2` | 0.8 | End time (phase units) |
| `--ntime` | 500 | Number of time points |
| `--expose` | 0.001 | Exposure time per point |
| `--ndivide` | 1 | Sub-divisions for exposure smearing |
| `--output` | stdout | Output file path |
| `--scale` | off | Autoscale to minimize chi-squared |

## Python API

```python
import lcurve_rs
import numpy as np

# Load a model from a parameter file
model = lcurve_rs.Model("model.dat")
print(model)  # Model(q=0.5, iangle=82, r1=0.015, r2=-1, t1=15000, t2=3500)

# Compute a light curve with evenly-spaced times
result = model.light_curve(time1=-0.2, time2=1.2, ntime=1000)

# Access results as numpy arrays
result.times  # (1000,) array of times
result.flux   # (1000,) array of computed fluxes
result.wdwarf # white dwarf contribution at phase 0.5
result.sfac   # scale factors [star1, disc, edge, spot, star2]

# Use custom time arrays
times = np.linspace(-0.5, 1.5, 2000)
result = model.light_curve(times=times, expose=0.01, ndiv=3)

# Fit to observed data
result = model.light_curve(data="observed.dat", scale=True)
result.chisq  # weighted chi-squared

# Modify parameters on the fly
model.q = 0.3
model.iangle = 85.0
model.set_param("t1", 20000.0)
print(model.get_param("t1"))  # 20000.0
```

### Available properties

Direct attribute access for common parameters: `q`, `iangle`, `r1`, `r2`, `t1`, `t2`, `period`, `t0`, `velocity_scale`.

All 65 physical parameters are accessible via `model.get_param(name)` / `model.set_param(name, value)`.

Full Pparam metadata (value, range, dstep, vary, defined) is available via `model.get_pparam(name)` and `model.set_pparam(name, ...)`.

## Parameter Estimation

Fit model parameters to observed data using MCMC or nested sampling. Install optional backends:

```bash
pip install lcurve_rs[emcee]      # MCMC
pip install lcurve_rs[ultranest]  # nested sampling
pip install lcurve_rs[all]        # all backends
```

```python
from lcurve_rs.fitting import Fitter, Prior

model = lcurve_rs.Model("model.dat")

# Mark parameters as free
model.set_pparam("q", vary=True, range=0.1)
model.set_pparam("iangle", vary=True, range=5.0)

# Optional: override a prior
priors = {"iangle": Prior.gaussian("iangle", mean=82.0, sigma=1.5, low=70, high=90)}

# Fit
fitter = Fitter(model, data="observed.dat", scale=True, priors=priors)
result = fitter.run_emcee(nwalkers=32, nsteps=5000, burn=1000, progress=True)

print(result.summary())   # median + credible intervals
print(result.best_fit)    # max-likelihood parameters
```

UltraNest and dynesty are also supported — see the [fitting guide](docs/getting-started/fitting.md) for details.

## Performance

The main time-point loop is parallelised with [Rayon](https://github.com/rayon-rs/rayon). On multi-core hardware, wall-clock time scales near-linearly with core count.

Grid setup (star continuum, disc eclipses) is also parallelised.

Release-profile optimizations include LTO (`lto = "fat"`), single codegen unit, and `#[inline]` hints on hot-path functions (visibility checks, limb darkening, grid interpolation).

Set `RAYON_NUM_THREADS=N` to control thread count.

## Testing

### Rust tests

```bash
cargo test
```

### Regression test (Rust vs C++)

Requires the original C++ `lroche` binary and pgplot:

```bash
module load gcccore/12.3.0 pgplot/5.2.2
python3 test_data/run_regression100.py
```

Runs 100 randomly-generated models through both C++ and Rust, verifying max relative error < 1e-5. Current worst-case error is ~4e-8.

### Python tests

```bash
cd python
source .venv/bin/activate
python -c "import lcurve_rs; m = lcurve_rs.Model('../test_data/test_model.dat'); print(m.light_curve(ntime=10).flux)"
```

## CI

GitHub Actions runs tests automatically on every push and PR. See `.github/workflows/tests.yml`.

## Project Structure

```
lcurve-rs/
├── src/main.rs              # CLI binary (lroche)
├── crates/
│   ├── lcurve/              # Core light curve engine
│   │   ├── orchestration.rs # Main computation loop (parallelised)
│   │   ├── flux.rs          # Flux calculations per component
│   │   ├── grid.rs          # Roche geometry grid generation
│   │   ├── brightness.rs    # Continuum brightness setup
│   │   ├── model.rs         # Model parameter parsing
│   │   └── types.rs         # Data types (Point, Datum, LDC)
│   ├── roche/               # Roche geometry library
│   └── subs/                # Numerical utilities (Vec3, Planck, root-finding)
├── python/                  # Python bindings (PyO3 + maturin)
│   ├── src/lib.rs           # PyO3 module
│   ├── lcurve_rs/           # Python package
│   └── pyproject.toml
├── test_data/               # Test models and regression suite
└── docs/                    # MkDocs documentation
```

## Acknowledgements

This is a port of Tom Marsh's C++ [lcurve](http://github.com/trmrsh/cpp-lcurve/tree/master) package and its dependencies ([cpp-roche](https://github.com/trmrsh/cpp-roche), [cpp-subs](https://github.com/trmrsh/cpp-subs)). The Roche geometry, eclipse computation, disc/spot models, and model parameter file format are all faithful ports of the original code. Numerical results agree with the C++ version to ~4e-8 relative error across 100 randomised test models.

## License

See the original lcurve license terms.
