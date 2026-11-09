"""
Catalogue access.

The cache is COMMITTED to the repository on purpose.  A live query would
make results irreproducible: the archive revises ephemerides and
eccentricities continually, so the same input would quietly give
different answers months apart.  For a tool accompanying a paper that is
a liability, and for a hosted app it would also be re-queried on every
cold start and rate-limited quickly.

Refreshing is therefore deliberate -- fetch(refresh=True) -- and every
refresh rewrites the provenance file alongside the data.

CAUTION on pscomppars: it is a COMPOSITE table, filling each column from
the best available source independently.  A planet's period can come
from one paper and its transit epoch from another, with no guarantee the
two are mutually consistent.  Period and epoch must share a reference or
phases drift; e and omega are fitted jointly and should move together.
Radius, T_eq and magnitudes can safely be mixed.
"""

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={q}&format=csv'

COLUMNS = ['pl_name', 'hostname', 'ra', 'dec',
           'pl_orbper', 'pl_tranmid', 'pl_trandur', 'pl_orbtper',
           'pl_orbeccen', 'pl_orblper', 'pl_orbsmax', 'pl_orbincl',
           'pl_rade', 'pl_bmasse', 'pl_eqt',
           'st_rad', 'st_teff', 'st_mass',
           # NO sy_ymag: the archive carries 2MASS photometry only
           # (J, H, Ks), and Y is not a 2MASS band.  Asking for it makes
           # TAP reject the whole query with HTTP 400.  Y-band work
           # falls back to J in to_params(); see below.
           'sy_jmag', 'sy_hmag', 'sy_kmag',
           'tran_flag']

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, 'data', 'catalogue.csv')
PROV = os.path.join(_HERE, 'data', 'provenance.json')


def fetch(refresh=False, cache=CACHE, prov=PROV):
    """Load the cached catalogue; re-query only when asked."""
    if os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache)
        missing = [c for c in COLUMNS if c not in df.columns]
        if not missing:
            return df, _read_prov(prov)
        print(f'cache missing {", ".join(missing)} -- refreshing')

    query = f'select {",".join(COLUMNS)} from pscomppars'
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print('querying the NASA Exoplanet Archive ...')
    try:
        urllib.request.urlretrieve(_TAP.format(q=urllib.parse.quote(query)),
                                   cache)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            raise RuntimeError(
                'The archive rejected the query (HTTP 400).  This almost '
                'always means a column name in COLUMNS does not exist in '
                'pscomppars.  Check it against\n'
                '  https://exoplanetarchive.ipac.caltech.edu/docs/'
                'API_PS_columns.html\n'
                f'Query was:\n  {query}') from exc
        raise
    df = pd.read_csv(cache)
    meta = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(
                    timespec='seconds'),
                table='pscomppars', query=query, n_rows=int(len(df)),
                columns=COLUMNS,
                note='Composite table: columns may come from different '
                     'source papers for the same planet.')
    with open(prov, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  {len(df)} rows -> {cache}')
    return df, meta


def _read_prov(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(retrieved_utc='unknown', note='no provenance file found')


def lookup(df, name):
    """Exact match first, then case-insensitive, then substring."""
    hit = df[df.pl_name == name]
    if len(hit):
        return hit.iloc[0]
    hit = df[df.pl_name.str.lower() == str(name).lower()]
    if len(hit):
        return hit.iloc[0]
    hit = df[df.pl_name.str.contains(str(name), case=False, na=False,
                                     regex=False)]
    if len(hit) == 1:
        return hit.iloc[0]
    if len(hit) > 1:
        raise ValueError(f'{name!r} matches {len(hit)}: '
                         f'{", ".join(hit.pl_name.head(8))}')
    raise KeyError(f'{name!r} not found')


def to_params(row, band='K'):
    """
    Catalogue row -> observability() keyword arguments.

    Fallbacks are applied here rather than silently downstream, and each
    is reported in `assumed` so the caller can show the user what was
    filled in: e = 0, i = 90 deg, a from Kepler III, and the conjunction
    epoch derived from periastron when pl_tranmid is absent.
    """
    from . import physics as P

    def g(k):
        v = row.get(k, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        return v if np.isfinite(v) else np.nan

    assumed = []
    e = g('pl_orbeccen')
    if not np.isfinite(e):
        e, _ = 0.0, assumed.append('e = 0 (not in catalogue)')
    w = g('pl_orblper')
    if not np.isfinite(w):
        w, _ = 90.0, assumed.append('omega = 90 deg (not in catalogue)')
    inc = g('pl_orbincl')
    if not np.isfinite(inc):
        inc, _ = 90.0, assumed.append('inclination = 90 deg (edge-on)')

    a = g('pl_orbsmax')
    if not np.isfinite(a) and np.isfinite(g('st_mass')) and np.isfinite(g('pl_orbper')):
        a = P.semi_major_axis(g('pl_orbper'), g('st_mass'))
        assumed.append('semi-major axis from Kepler III')

    t_conj, derived = P.conjunction_epoch(g('pl_tranmid'), g('pl_orbtper'),
                                          g('pl_orbper'), e, w)
    if derived and np.isfinite(t_conj):
        assumed.append('conjunction epoch derived from periastron '
                       '(less reliable; see catalogue docstring)')

    # The archive carries 2MASS only (J, H, Ks).  A Y request is
    # prefilled from J, the nearest band it does have, and flagged so
    # the user can type a real Y magnitude over it.
    if band == 'Y':
        mag = g('sy_jmag')
        if np.isfinite(mag):
            assumed.append('no Y magnitude in the catalogue, so the J '
                           'value is shown; replace it if you have one')
    else:
        mag = g({'J': 'sy_jmag', 'H': 'sy_hmag', 'K': 'sy_kmag'}[band])
    if not np.isfinite(mag):
        mag = g('sy_kmag')
        if np.isfinite(mag):
            assumed.append(f'{band} magnitude missing; using K')

    rp = g('pl_rade')
    return dict(
        ra_deg=g('ra'), dec_deg=g('dec'), period_d=g('pl_orbper'),
        t_conj_bjd=t_conj, e=e, omega_star_deg=w,
        t14_hr=g('pl_trandur'), a_au=a, incl_deg=inc,
        rp_rearth=rp, mp_mearth=g('pl_bmasse'), rs_rsun=g('st_rad'),
        teq_k=g('pl_eqt'), teff_k=g('st_teff'), mag=mag,
        rocky=bool(np.isfinite(rp) and rp < 1.75),
        epoch_derived=derived,
    ), assumed
