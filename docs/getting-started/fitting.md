# Parameter Estimation

`lcurve_rs` includes a fitting module that drives parameter estimation using
nested sampling or MCMC.  Three backends are supported:

| Backend | Method | Install |
|---------|--------|---------|
| [emcee](https://emcee.readthedocs.io/) | Affine-invariant MCMC | `pip install lcurve_rs[emcee]` |
| [UltraNest](https://johannesbuchner.github.io/UltraNest/) | Reactive nested sampling | `pip install lcurve_rs[ultranest]` |
| [dynesty](https://dynesty.readthedocs.io/) | Dynamic nested sampling | `pip install lcurve_rs[dynesty]` |

Install all three at once with `pip install lcurve_rs[all]`.

## Overview

The workflow is:

1. Load a model and mark parameters as free (`vary=True`) with prior ranges.
2. Create a `Fitter` pointing at an observed data file.
3. Call `run_emcee()`, `run_ultranest()`, or `run_dynesty()`.
4. Inspect the returned `FitResult`.

## Quick example

```python
import lcurve_rs
from lcurve_rs.fitting import Fitter, Prior

model = lcurve_rs.Model("model.dat")

# Mark parameters as free (or set vary=True in the model file)
model.set_pparam("q", vary=True, range=0.1)
model.set_pparam("iangle", vary=True, range=5.0)

# Check which parameters will be fitted
for p in model.get_free_params():
    lo = p["value"] - p["range"]
    hi = p["value"] + p["range"]
    print(f"  {p['name']:20s}  [{lo:.4f}, {hi:.4f}]")

# Create fitter
fitter = Fitter(model, data="observed.dat", scale=True)

# Run MCMC
result = fitter.run_emcee(nwalkers=32, nsteps=5000, burn=1000, progress=True)

# Print summary with 1-sigma credible intervals
print(result.summary())

# Best-fit and median parameter values
print(result.best_fit)
print(result.median)
```

## Working with parameters

### Inspecting Pparam metadata

Every physical parameter in the model file has five fields:

| Field | Type | Description |
|-------|------|-------------|
| `value` | float | Current value |
| `range` | float | Half-width of the default prior range |
| `dstep` | float | Step size (used by optimisers) |
| `vary` | bool | Whether the parameter is free |
| `defined` | bool | Whether the parameter was in the model file |

```python
info = model.get_pparam("q")
# {'value': 0.5, 'range': 0.1, 'dstep': 0.01, 'vary': False, 'defined': True}
```

### Setting parameters as free

You can mark parameters as free either in the model file (set the vary flag
to 1) or programmatically:

```python
model.set_pparam("q", vary=True, range=0.1)
model.set_pparam("iangle", vary=True, range=5.0, dstep=0.1)
```

### Listing free parameters

```python
free = model.get_free_params()
# [{'name': 'q', 'value': 0.5, 'range': 0.1, 'dstep': 0.01}, ...]
```

### Copying a model

The `Fitter` works on its own internal copy, but you can also copy models
explicitly:

```python
m2 = model.copy()
m2.q = 0.3  # does not affect the original
```

## Priors

By default, each free parameter gets a uniform prior on
`[value - range, value + range]`, taken from the model file.

Override priors for specific parameters by passing a dict:

```python
from lcurve_rs.fitting import Prior

priors = {
    # Uniform prior on [0.3, 0.7]
    "q": Prior.uniform("q", low=0.3, high=0.7),

    # Truncated Gaussian prior (requires scipy)
    "iangle": Prior.gaussian("iangle", mean=82.0, sigma=1.5, low=70, high=90),
}

fitter = Fitter(model, data="observed.dat", priors=priors)
```

### Prior.uniform(name, low, high)

Flat prior between `low` and `high`.

### Prior.gaussian(name, mean, sigma, low, high)

Truncated Gaussian with mean `mean` and width `sigma`, hard-clipped to
`[low, high]`. Requires `scipy`.

## Running samplers

### emcee (MCMC)

```python
result = fitter.run_emcee(
    nwalkers=32,    # number of walkers (default: max(32, 2*ndim+2))
    nsteps=5000,    # total MCMC steps
    burn=1000,      # burn-in steps to discard
    progress=True,  # show tqdm progress bar
)
```

Walkers are initialised as a small ball (1% of prior width) around the
midpoint of each prior.

### UltraNest (nested sampling)

```python
result = fitter.run_ultranest(
    min_num_live_points=400,
    # any other kwargs are passed to ReactiveNestedSampler.run()
)
print(f"log(Z) = {result.log_evidence:.2f} +/- {result.log_evidence_err:.2f}")
```

### dynesty (nested sampling)

```python
result = fitter.run_dynesty(
    nlive=500,
    # any other kwargs are passed to NestedSampler.run_nested()
)
```

## FitResult

All backends return a `FitResult` with a uniform interface:

| Attribute | Type | Description |
|-----------|------|-------------|
| `backend` | str | `"emcee"`, `"ultranest"`, or `"dynesty"` |
| `param_names` | list[str] | Ordered parameter names |
| `samples` | ndarray | Shape `(n_samples, n_params)` |
| `log_likelihood` | ndarray | Shape `(n_samples,)` |
| `weights` | ndarray or None | Sample weights (nested sampling only) |
| `log_evidence` | float or None | log(Z) (nested sampling only) |
| `log_evidence_err` | float or None | Uncertainty on log(Z) |
| `raw_result` | object | Backend's native result object |

### Properties

```python
result.best_fit   # dict: parameter values at max likelihood
result.median     # dict: median of each parameter's marginal posterior
```

### Summary

```python
print(result.summary(ci=68.27))
```

Prints a table of median values with credible intervals (default 1-sigma):

```
Backend: emcee  |  Samples: 1280

           Parameter        Median         Lower         Upper
--------------------------------------------------------------
                   q      0.49533       0.46541      0.522303
              iangle       81.809        81.137       82.4598
```

## Inject-and-recover example

A full worked example that generates synthetic data with known parameters,
perturbs them, and uses emcee to recover the truth:

```python
import os, tempfile
import numpy as np
import lcurve_rs
from lcurve_rs.fitting import Fitter, Prior

# 1. Load model with known truth
model_true = lcurve_rs.Model("model.dat")
TRUE_Q = model_true.q          # e.g. 0.5
TRUE_IANGLE = model_true.iangle  # e.g. 82.0

# 2. Generate synthetic data with noise
times = np.linspace(-0.25, 0.75, 80)
result = model_true.light_curve(times=times)
flux_true = np.array(result.flux)

rng = np.random.default_rng(42)
snr = 50.0
ferr = flux_true.mean() / snr
flux_obs = flux_true + rng.normal(0, ferr, len(times))

# Write to lcurve data format: time expose flux ferr weight ndiv
data_path = os.path.join(tempfile.mkdtemp(), "synthetic.dat")
with open(data_path, "w") as f:
    for i in range(len(times)):
        f.write(f"{times[i]:.12e} 0.001 {flux_obs[i]:.10e} {ferr:.10e} 1 1\n")

# 3. Perturb starting values
model_fit = model_true.copy()
model_fit.set_pparam("q", vary=True, range=0.1)
model_fit.set_pparam("iangle", vary=True, range=5.0)
model_fit.q = TRUE_Q + 0.03
model_fit.iangle = TRUE_IANGLE - 1.5

# 4. Fit
fitter = Fitter(model_fit, data=data_path, scale=True)
result = fitter.run_emcee(nwalkers=16, nsteps=100, burn=20, progress=True)

# 5. Check
print(result.summary())
print(f"True q={TRUE_Q}, recovered={result.median['q']:.4f}")
print(f"True iangle={TRUE_IANGLE}, recovered={result.median['iangle']:.2f}")
```

## Performance tips

- Each likelihood evaluation requires a full light curve computation, so
  wall time is roughly `nwalkers * nsteps * (time per light curve)`.
- Use fewer data points for faster iterations during exploratory runs.
- Set `RAYON_NUM_THREADS=1` if running multiple walkers — emcee's
  parallelism and Rayon's parallelism can compete for cores.
- For production runs, use nested sampling (UltraNest/dynesty) for
  evidence computation, or emcee with longer chains for posterior sampling.
