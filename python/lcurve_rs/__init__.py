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
from lcurve_rs._lcurve_rs import _set_device, _get_device, _has_cuda
from lcurve_rs.fitting import Fitter, Prior, FitResult

__all__ = [
    "Model", "LcResult", "Fitter", "Prior", "FitResult",
    "set_device", "get_device", "has_cuda",
]
__version__ = "0.1.0"


def set_device(device: str = "cpu") -> None:
    """Set compute device: 'cpu', 'cuda', or 'cuda:N'.

    Examples
    --------
    >>> import lcurve_rs
    >>> lcurve_rs.set_device("cuda")     # use default GPU
    >>> lcurve_rs.set_device("cuda:1")   # use GPU 1
    >>> lcurve_rs.set_device("cpu")      # force CPU (default)
    """
    _set_device(device)


def get_device() -> str:
    """Return current device string ('cpu' or 'cuda:N')."""
    return _get_device()


def has_cuda() -> bool:
    """Return True if lcurve_rs was built with CUDA support."""
    return _has_cuda()
