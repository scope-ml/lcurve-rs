# Quick Start

## CLI

Generate a synthetic light curve from a model parameter file:

```bash
lroche model.dat none --time1=-0.2 --time2=1.2 --ntime=500 --output=lc.dat
```

The output file has columns: `time  expose  flux  ferr  weight  ndiv`.

Fit to observed data with autoscaling:

```bash
lroche model.dat observed.dat --scale --output=fit.dat
```

## Python

### Load a model and compute a light curve

```python
import lcurve_rs
import numpy as np

# Load model from parameter file
model = lcurve_rs.Model("model.dat")
print(model)
# Model(q=0.5, iangle=82, r1=0.015, r2=-1, t1=15000, t2=3500)

# Compute light curve at evenly-spaced phases
result = model.light_curve(time1=-0.2, time2=1.2, ntime=1000)
```

### Access results

```python
result.times   # numpy array of time values
result.flux    # numpy array of computed fluxes (alias for result.calc)
result.wdwarf  # white dwarf contribution at phase 0.5
result.logg1   # flux-weighted log10(g) for star 1 (CGS)
result.logg2   # flux-weighted log10(g) for star 2 (CGS)
result.rv1     # volume-averaged radius of star 1
result.rv2     # volume-averaged radius of star 2
result.sfac    # scale factors [star1, disc, edge, spot, star2]
result.chisq   # weighted chi-squared (when fitting data)
```

### Modify parameters

```python
# Direct property access for common parameters
model.q = 0.3
model.iangle = 85.0

# Generic access for all 65 physical parameters
model.set_param("gravity_dark1", 0.08)
print(model.get_param("gravity_dark1"))  # 0.08

# Recompute after modification
result2 = model.light_curve(time1=-0.2, time2=1.2, ntime=1000)
```

### Custom time arrays

```python
times = np.linspace(-0.5, 1.5, 2000)
result = model.light_curve(times=times, expose=0.01, ndiv=3)
```

### Fit to observed data

```python
result = model.light_curve(data="observed.dat", scale=True)
print(f"chi-squared: {result.chisq:.2f}")
print(f"scale factors: {result.sfac}")
```

### Plot a light curve

```python
import matplotlib.pyplot as plt

result = model.light_curve(time1=-0.2, time2=1.2, ntime=1000)
plt.plot(result.times, result.flux)
plt.xlabel("Phase")
plt.ylabel("Flux")
plt.title(f"q={model.q}, i={model.iangle}")
plt.show()
```

### Parameter estimation

Fit model parameters to observed data using MCMC or nested sampling.
See the [Parameter Estimation](fitting.md) guide for full details.

```python
from lcurve_rs.fitting import Fitter

# Mark parameters as free
model.set_pparam("q", vary=True, range=0.1)
model.set_pparam("iangle", vary=True, range=5.0)

# Fit with emcee
fitter = Fitter(model, data="observed.dat", scale=True)
result = fitter.run_emcee(nwalkers=32, nsteps=5000, burn=1000, progress=True)
print(result.summary())
```
