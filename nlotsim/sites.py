"""
Observing sites.  Longitudes are degrees EAST.

Collecting areas are effective, not pi r^2: NLOT's 121 m^2 is 90
segments of 1.35 m^2 (ratri.py), equivalent to a 12.4 m filled aperture.

Throughput is the LEAST reliable input.  It is degenerate with the band
zero point, so comparing against a published SNR constrains only their
product; with the zero points pinned by Cohen et al. 2003, throughput
alone sets the absolute SNR scale.

NLOT uses the measured GIANO efficiency curve (load_giano_throughput),
as a proxy for a GIANO-like instrument -- the same proxy ratri.py uses.
Band means from that curve are 1.23% (J), 2.04% (H), 2.64% (K), against
the flat 8% this file previously assumed, which overstated K by 3x and
hence the SNR by sqrt(3).

The comparison sites still carry flat placeholder values.  They are
there to give the structure something to hold, NOT to be quoted.
"""

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_GIANO = os.path.join(_DATA, 'throughput_giano.npz')


def load_giano_throughput():   # kept for backwards compatibility
    """
    Measured GIANO efficiency as (wl_um, throughput), for use as a
    GIANO-like proxy.

    Derived from the 50 x 2048 echelle order traces used by ratri.py,
    flattened, sorted and median-binned to ~1100 points (band means
    preserved to ~1%).  Inter-order gaps are left as gaps rather than
    interpolated across, because no light is collected there; the band
    mean is taken over covered pixels only, which is what a
    per-resolution-element SNR needs.

    Spans 0.935-2.423 um, so it covers J, H and the great majority of
    K (96% of the band FWHM).
    """
    if not os.path.exists(_GIANO):
        raise FileNotFoundError(
            f'{_GIANO} is missing; falling back to a scalar throughput '
            'is the caller\'s choice')
    d = np.load(_GIANO)
    return d['wl_um'], d['throughput']


try:
    from .instruments import load as _load_instrument
    _GIANO_CURVE = _load_instrument('giano')
except Exception:                 # data file absent, or circular import
    try:
        _GIANO_CURVE = load_giano_throughput()
    except FileNotFoundError:
        _GIANO_CURVE = 0.02       # crude stand-in

SITES = {
    'NLOT': dict(
        name='NLOT (Hanle)', lat=32.7794, lon=78.9661, height=4535.0,
        area_m2=121.0, throughput=_GIANO_CURVE, t_duty_s=60.0,
        resolution=60000,
        note='Proposed. Area and duty time from ratri.py '
             '(nlot_ccd_giano); throughput is the measured GIANO '
             'efficiency curve used as a GIANO-like proxy.'),
    'MaunaKea': dict(
        name='Mauna Kea', lat=19.8255, lon=-155.4683, height=4204.0,
        area_m2=76.0, throughput=0.05, t_duty_s=40.0, resolution=70000,
        note='Generic 10 m-class reference site. Throughput is a PLACEHOLDER, not a budget.'),
    'Paranal': dict(
        name='Cerro Paranal', lat=-24.6268, lon=-70.4045, height=2635.0,
        area_m2=52.8, throughput=0.05, t_duty_s=40.0, resolution=100000,
        note='Generic VLT UT reference site. Throughput is a PLACEHOLDER, not a budget.'),
    'LaPalma': dict(
        name='Roque de los Muchachos', lat=28.7540, lon=-17.8900,
        height=2370.0, area_m2=10.0, throughput=0.05, t_duty_s=40.0,
        resolution=50000,
        note='TNG-like. Throughput is a PLACEHOLDER, not a budget.'),
}

DEFAULT_SITE = 'NLOT'


def get_site(key_or_dict):
    """Accept a key, or a dict for a user-defined site."""
    if isinstance(key_or_dict, dict):
        return key_or_dict
    if key_or_dict not in SITES:
        raise KeyError(f'unknown site {key_or_dict!r}; '
                       f'known: {", ".join(SITES)}')
    return SITES[key_or_dict]
