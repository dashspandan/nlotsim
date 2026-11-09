"""
Measured instrument efficiency curves.

All are INSTRUMENT ONLY -- they exclude the atmosphere, which
atmosphere.py supplies separately, exactly as ratri does.  Each was
derived from the echelle order traces used by ratri: flattened,
sorted, positive values only, median-binned to ~1100 points with band
means preserved to about 1%.  Inter-order gaps are left as gaps rather
than interpolated across, since no light is collected there.

Band means (instrument only):

    instrument   J        H        K        coverage
    GIANO        0.0122   0.0206   0.0265   0.935-2.422 um
    CARMENES     0.0349   0.0284   --       0.961-1.712 um
    SPIRou       0.0151   0.0379   0.0522   0.964-2.507 um

CARMENES has no K band at all, and is the best of the three in J.
SPIRou is twice GIANO in K.  Since these are efficiencies only, the
collecting area is handled separately in sites.py -- all three
instruments sit on 3.5-3.6 m telescopes, so using one as a proxy for a
10 m-class site means borrowing the EFFICIENCY, not the aperture.
"""

import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_ARCHIVE = os.path.join(_DATA, 'throughput_instruments.npz')

INSTRUMENTS = {
    'giano': dict(
        label='GIANO-B (TNG)',
        note='Proxy for a GIANO-like NLOT instrument, as in ratri.',
        telescope_m=3.58, coverage_um=(0.935, 2.422)),
    'carmenes': dict(
        label='CARMENES NIR (Calar Alto)',
        note='Covers J and H only; no K.',
        telescope_m=3.5, coverage_um=(0.961, 1.712)),
    'spirou': dict(
        label='SPIRou (CFHT)',
        note='Widest coverage of the three, and the best in K.',
        telescope_m=3.58, coverage_um=(0.964, 2.507)),
}

DEFAULT_INSTRUMENT = 'giano'


def load(name=DEFAULT_INSTRUMENT):
    """Return (wl_um, throughput) for a named instrument."""
    key = str(name).lower()
    if key not in INSTRUMENTS:
        raise KeyError(f'unknown instrument {name!r}; '
                       f'known: {", ".join(INSTRUMENTS)}')
    if not os.path.exists(_ARCHIVE):
        raise FileNotFoundError(f'{_ARCHIVE} is missing')
    d = np.load(_ARCHIVE)
    return d[f'{key}_wl'], d[f'{key}_eta']


def covers(name, band):
    """Does this instrument cover the band's full width?"""
    from .bands import BANDS
    b = BANDS[band]
    lo, hi = INSTRUMENTS[str(name).lower()]['coverage_um']
    return lo <= b.lam_um - b.width_um / 2 and hi >= b.lam_um + b.width_um / 2


def table():
    """Summary for the app and the CLI banner."""
    from .bands import BANDS, resolve_throughput
    rows = []
    for k, meta in INSTRUMENTS.items():
        curve = load(k)
        row = dict(instrument=meta['label'],
                   coverage=f"{meta['coverage_um'][0]:.2f}-"
                            f"{meta['coverage_um'][1]:.2f} um")
        for b in BANDS:
            row[b] = (round(float(resolve_throughput(curve, b)), 4)
                      if covers(k, b) else None)
        rows.append(row)
    return rows
