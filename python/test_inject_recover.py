"""
Inject-and-recover test for the lcurve_rs fitting infrastructure.

1. Load the test model with known "true" parameters.
2. Generate a synthetic light curve and add Gaussian noise to create
   fake observations.
3. Perturb two parameters (q and iangle) away from their true values.
4. Run emcee to recover the injected parameters.
5. Assert the recovered medians are within 3-sigma of the truth.
"""

import os
import tempfile

import numpy as np

import lcurve_rs
from lcurve_rs.fitting import Fitter, Prior

# ── 1. True model ──────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "test_data", "test_model.dat")
model_true = lcurve_rs.Model(MODEL_PATH)

TRUE_Q = model_true.q            # 0.5
TRUE_IANGLE = model_true.iangle  # 82.0

print(f"True parameters: q={TRUE_Q}, iangle={TRUE_IANGLE}")

# ── 2. Generate synthetic observed data ────────────────────────────────

# Phase-folded times across the full orbit (keep small for speed)
ntimes = 80
times_arr = np.linspace(-0.25, 0.75, ntimes)

result_true = model_true.light_curve(times=times_arr)
flux_true = np.array(result_true.flux)

# Signal-to-noise ~ 50 (2% noise — higher noise = broader posterior = easier to recover)
rng = np.random.default_rng(42)
snr = 50.0
ferr = flux_true.mean() / snr  # constant error bar
noise = rng.normal(0, ferr, ntimes)
flux_obs = flux_true + noise

# Write to a temporary lcurve data file:
# columns: time  expose  flux  ferr  weight  ndiv
tmpdir = tempfile.mkdtemp()
data_path = os.path.join(tmpdir, "synthetic.dat")
expose = 0.001
with open(data_path, "w") as f:
    for i in range(ntimes):
        f.write(f"{times_arr[i]:.12e} {expose} {flux_obs[i]:.10e} {ferr:.10e} 1 1\n")

print(f"Synthetic data written to {data_path}")
print(f"  {ntimes} points, SNR~{snr:.0f}, ferr={ferr:.4e}")

# Verify the true model gives a sensible chisq against this data
result_check = model_true.light_curve(data=data_path, scale=True)
print(f"  True-model chisq = {result_check.chisq:.1f}  (expect ~{ntimes})")

# ── 3. Perturbed starting model ───────────────────────────────────────

model_fit = model_true.copy()

# Mark q and iangle as free, with ranges
model_fit.set_pparam("q", vary=True, range=0.1)
model_fit.set_pparam("iangle", vary=True, range=5.0)

# Offset starting values from truth
model_fit.set_param("q", TRUE_Q + 0.03)          # 0.53
model_fit.set_param("iangle", TRUE_IANGLE - 1.5)  # 80.5

print(f"\nStarting fit from: q={model_fit.q}, iangle={model_fit.iangle}")
print(f"Free parameters: {[p['name'] for p in model_fit.get_free_params()]}")

# ── 4. Run emcee ──────────────────────────────────────────────────────

priors = {
    "q":      Prior.uniform("q",      TRUE_Q - 0.1,      TRUE_Q + 0.1),
    "iangle": Prior.uniform("iangle", TRUE_IANGLE - 5.0,  TRUE_IANGLE + 5.0),
}

fitter = Fitter(model_fit, data=data_path, scale=True, priors=priors)

NWALKERS, NSTEPS, BURN = 16, 100, 20
print(f"\nRunning emcee (nwalkers={NWALKERS}, nsteps={NSTEPS}, burn={BURN}) ...")
result = fitter.run_emcee(nwalkers=NWALKERS, nsteps=NSTEPS, burn=BURN, progress=True)

print("\n" + result.summary())

# ── 5. Check recovery ────────────────────────────────────────────────

median = result.median
best = result.best_fit

print(f"\n{'':=<60}")
print("Recovery check:")
print(f"  q:      true={TRUE_Q:.4f}   median={median['q']:.4f}   best={best['q']:.4f}")
print(f"  iangle: true={TRUE_IANGLE:.2f}  median={median['iangle']:.2f}  best={best['iangle']:.2f}")

# Compute posterior std
q_std = np.std(result.samples[:, 0])
ia_std = np.std(result.samples[:, 1])
print(f"  q      posterior std = {q_std:.4f}")
print(f"  iangle posterior std = {ia_std:.4f}")

# Assert within 3 sigma
q_err = abs(median["q"] - TRUE_Q)
ia_err = abs(median["iangle"] - TRUE_IANGLE)

q_ok = q_err < 3 * q_std
ia_ok = ia_err < 3 * ia_std

print(f"\n  q      offset = {q_err:.4f} ({q_err/q_std:.1f} sigma) — {'PASS' if q_ok else 'FAIL'}")
print(f"  iangle offset = {ia_err:.4f} ({ia_err/ia_std:.1f} sigma) — {'PASS' if ia_ok else 'FAIL'}")

# Cleanup
os.remove(data_path)
os.rmdir(tmpdir)

if q_ok and ia_ok:
    print("\n  *** ALL PARAMETERS RECOVERED SUCCESSFULLY ***")
else:
    raise AssertionError("Parameter recovery failed!")
