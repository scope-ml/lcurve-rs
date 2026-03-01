"""
lcurve_rs — Python bindings for the Rust lcurve light curve engine.

Fast, parallel computation of synthetic light curves for eclipsing
white-dwarf binaries.  A Rust port of Tom Marsh's C++ lcurve/lroche code,
exposed to Python via PyO3.

Example
-------
>>> import lcurve_rs
>>> model = lcurve_rs.Model("model.dat")
>>> result = model.light_curve(time1=-0.2, time2=1.2, ntime=1000)
>>> result.flux   # numpy array of computed fluxes
"""

from lcurve_rs._lcurve_rs import Model, LcResult
from lcurve_rs.fitting import Fitter, Prior, FitResult

__all__ = ["Model", "LcResult", "Fitter", "Prior", "FitResult"]
__version__ = "0.1.0"
