"""
Atmospheric transmission.

The instrument efficiency curves in this package (GIANO and friends) are
INSTRUMENT ONLY -- they exclude the atmosphere.  ratri.py multiplies
them by a separate transmission spectrum from ESO's skycalc, and so does
this module, so the two agree.

The shipped curve is skycalc at airmass 1.0 and PWV 2.5 mm, median-
binned from 204,330 points over 0.90-2.50 um down to 3,000.

AIRMASS SCALING follows Beer-Lambert.  Transmission is exp(-tau), and
the optical depth along a slant path is X times the zenith value, so

    T(X) = T(1) ** X

exact for a non-scattering, plane-parallel atmosphere, which is a good
approximation out to the airmass 2 cut used here.

MEAN OR MEDIAN?  Both are stored and the choice matters by 2-3%:

    band    mean      median
    J       0.9564    0.9752
    H       0.9689    0.9834
    K       0.9466    0.9706

The MEAN counts every photon the atmosphere removes and is the honest
figure for total throughput.  The MEDIAN is closer to HRCCS practice,
where heavily absorbed pixels are masked before cross-correlation and
so never contribute: what survives the mask is nearer the median.  The
default is the mean, which is conservative.

WATER VAPOUR is not modelled here.  Only PWV 2.5 mm is shipped, and
scaling transmission to another PWV is not a simple power law -- the
water lines saturate.  ratri.py handles it by interpolating between
skycalc tables computed at several PWV values, which is the right
approach; generate those tables and pass them in if you need it.
"""

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_SKY = os.path.join(_DATA, 'transmission_airmass1_pwv2.5.npz')


def load_transmission(statistic='mean'):
    """
    Shipped skycalc transmission as (wl_um, transmission) at airmass 1.

    statistic='mean'    every photon the atmosphere removes (default)
    statistic='median'  closer to what survives telluric masking
    """
    if not os.path.exists(_SKY):
        raise FileNotFoundError(f'{_SKY} is missing')
    d = np.load(_SKY)
    key = 'transmission' if statistic == 'mean' else 'transmission_median'
    return d['wl_um'], d[key]


def transmission_at_airmass(airmass, statistic='mean'):
    """Beer-Lambert: T(X) = T(1) ** X."""
    wl, t1 = load_transmission(statistic)
    return wl, np.clip(t1, 1e-12, 1.0) ** float(airmass)


def combine(throughput, transmission):
    """
    Multiply an instrument curve by an atmospheric one onto a common
    grid, giving the end-to-end efficiency the SNR model expects.

    Either argument may be a scalar or a (wl_um, values) pair.  The
    result is interpolated onto whichever input is a curve; if both
    are, onto the finer of the two restricted to their overlap.
    """
    def as_curve(x):
        if np.isscalar(x):
            return None, float(x)
        if isinstance(x, dict):
            return np.asarray(x['wl_um'], float), \
                   np.asarray(x['throughput'], float)
        a, b = x
        return np.asarray(a, float), np.asarray(b, float)

    w1, v1 = as_curve(throughput)
    w2, v2 = as_curve(transmission)

    if w1 is None and w2 is None:
        return v1 * v2
    if w1 is None:
        return w2, v1 * v2
    if w2 is None:
        return w1, v1 * v2

    lo, hi = max(w1.min(), w2.min()), min(w1.max(), w2.max())
    grid = w1[(w1 >= lo) & (w1 <= hi)] if w1.size >= w2.size \
        else w2[(w2 >= lo) & (w2 <= hi)]
    if grid.size < 2:
        raise ValueError('throughput and transmission curves do not overlap')
    return grid, np.interp(grid, w1, v1) * np.interp(grid, w2, v2)
