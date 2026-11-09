"""
Orbital mechanics, timing and photon accounting.

This is the single copy of the physics.  The figure script, the CLI and
the web app all import from here, so a correction made once propagates
everywhere rather than being fixed in whichever copy was looked at.

ORBITAL CONVENTION  (stated explicitly, per Householder & Weiss 2023,
arXiv:2212.06966, who show that omega alone does not define an RV curve)

    Z-hat            points AWAY from the observer
    sign             positive v_p = receding = REDSHIFT
    ascending node   where the body crosses the sky plane MOVING AWAY
                     from the observer (Paddock 1913; Eastman+2013)
    omega input      omega_STAR, as held in the archive's pl_orblper;
                     converted internally to omega_p = omega_star + pi
    transit          f + omega_p = -pi/2, i.e. f + omega_star = +pi/2
    phase zero       inferior conjunction (transit)

Equivalently the RadVel / ExoFast convention written for the planet.

Verifiable without reading the code: v_p = 0 at phase 0 and 0.5 when
e = 0, peaks at +K_p at quadrature, rises just after transit (a receding
planet must), and equals -K_p e cos(omega_star) at transit for any e.
"""

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body_barycentric
import astropy.units as u

_AU_KM = 1.495978707e8
_AU_LIGHT_S = 499.004783836
_DAY_S = 86400.0
C_KMS = 299792.458
_J2000 = 2451545.0

_G_CGS = 6.67430e-8
_K_B = 1.380649e-16
_M_U = 1.66053907e-24
_M_EARTH = 5.9722e27
_R_EARTH = 6.371e8
_RSUN_RE = 109.076


# ── Kepler ────────────────────────────────────────────────────────────

def ecc_anomaly(M, e, tol=1e-10, itmax=200):
    """
    Eccentric anomaly by Newton-Raphson on E - e sin(E) - M = 0.

    Ported from ratri (Dash et al. 2024) so results agree, with one
    change: an iteration cap.  The original loops on
    `while max(abs(fn)) > tol`, which cannot terminate if any element
    fails to converge (e -> 1, or a NaN entering M).  The cap turns that
    into a bounded inaccuracy instead of a hang -- it matters when
    sweeping a whole catalogue rather than a curated target list.
    """
    M = np.asarray(M, dtype=float)
    E = M.copy()
    for _ in range(itmax):
        fn = E - e * np.sin(E) - M
        if not np.any(np.abs(fn) > tol):
            break
        E = E - fn / (1 - e * np.cos(E))
    return E


def mean_anomaly_at_conjunction(e, omega_star_deg):
    """
    Mean anomaly at inferior conjunction.

    Transit is GEOMETRIC, so it fixes the TRUE anomaly at
    f_tr = -pi/2 - omega_p.  The MEAN anomaly there is a different
    number, reached only via the eccentric anomaly.

    Dash et al. 2024 eq. 14 sets M(phi=0) = -pi/2 - omega directly,
    which assigns a true-anomaly value to the mean anomaly, and uses the
    archive omega without the pi shift.  Each error is O(e) with
    opposite sign, so THEY CANCEL to first order; the published form is
    correct to O(e) and wrong at O(e^2), about 190 e^2 km/s for
    K_p = 150 km/s.  Negligible below e ~ 0.05, ~9 km/s at e = 0.25.

    The omega convention was established empirically: of 334 planets
    carrying both pl_tranmid and pl_orbtper, f_tr = +pi/2 - omega_star
    reproduces pl_tranmid with median phase error 0.027 (56% within
    0.05), against 0.37 (6%) for f_tr = -pi/2 - omega_star.
    """
    w = np.radians(omega_star_deg if np.isfinite(omega_star_deg) else 90.0)
    w = w + np.pi                                   # star -> planet
    if not np.isfinite(e) or e == 0.0:
        return -np.pi / 2 - w                       # true == mean at e = 0
    f_tr = -np.pi / 2 - w
    E_tr = 2 * np.arctan(np.sqrt(1 - e) / np.sqrt(1 + e) * np.tan(f_tr / 2))
    return E_tr - e * np.sin(E_tr)


def planet_kp(a_au, period_d, e=0.0, incl_deg=90.0):
    """
    Planet orbital RV semi-amplitude, Dash et al. 2024 eq. 11:

        K_P = 2 pi a / (P sqrt(1 - e^2)) sin(i)

    Validated against literature: WASP-76 b 198.4 vs 196.5, WASP-33 b
    227.7 vs 230.9, HD 189733 b 151.5 vs 153 km/s.
    """
    e = 0.0 if not np.isfinite(e) else e
    incl_deg = 90.0 if not np.isfinite(incl_deg) else incl_deg
    with np.errstate(invalid='ignore', divide='ignore'):
        return (2 * np.pi * a_au * _AU_KM
                / (period_d * _DAY_S * np.sqrt(1 - e ** 2))
                * np.sin(np.radians(incl_deg)))


def semi_major_axis(period_d, st_mass_msun):
    """Kepler III fallback when pl_orbsmax is missing, in AU."""
    return (st_mass_msun * (period_d / 365.25) ** 2) ** (1.0 / 3.0)


def planet_rv(phase, kp, e=0.0, omega_star_deg=90.0, vsys=0.0, v_bary=0.0):
    """
    Planet radial velocity (km/s), Dash et al. 2024 eqs. 10-14 with
    eq. 14 and the omega convention corrected (see above).

    phase is measured from inferior conjunction.  omega_star_deg is the
    archive's pl_orblper; the pi shift happens inside, so callers pass
    the catalogue value unmodified.

    v_sys and v_bary default to zero: only the DIFFERENCE in v_p across
    a block is used downstream, so a constant cancels exactly.  v_bary
    does not cancel but varies only 0.5-0.8 km/s within one night,
    10-16% of a 5 km/s resolution element at R = 60,000.
    """
    phase = np.asarray(phase, dtype=float)
    e = 0.0 if not np.isfinite(e) else e
    w = np.radians(omega_star_deg if np.isfinite(omega_star_deg) else 90.0)
    w = w + np.pi

    M = 2 * np.pi * phase + mean_anomaly_at_conjunction(e, omega_star_deg)
    E = M if e == 0.0 else ecc_anomaly(M, e)
    f = 2 * np.arctan(np.sqrt(1 + e) / np.sqrt(1 - e) * np.tan(E / 2))
    return vsys - v_bary + kp * (np.cos(f + w) + e * np.cos(w))


def conjunction_epoch(t_tranmid, t_peri, period_d, e, omega_star_deg):
    """
    Epoch of inferior conjunction in BJD_TDB.

    pl_tranmid is already that epoch, so it is used where present.
    Otherwise invert the definition of the mean anomaly,
    M = 2 pi (t - t_peri)/P, at M = M_tr:

        t_ic = t_peri + (P / 2 pi) M_tr

    valid for non-transiting planets too, inferior conjunction being
    geometric.  t_peri is itself BJD_TDB, so the result inherits that
    scale and no conversion is needed here.

    RELIABILITY: the derived branch is less trustworthy than
    pl_tranmid.  pscomppars is a COMPOSITE table that fills each column
    from the best available source independently, so pl_tranmid can come
    from a different paper than pl_orbtper and pl_orblper with no
    guarantee of a consistent ephemeris.  Cross-checking the 334 planets
    carrying both, the derived epoch lands within 0.05 in phase for only
    56% of them.  Callers should flag results that used this branch.
    """
    if np.isfinite(t_tranmid):
        return float(t_tranmid), False
    if not (np.isfinite(t_peri) and np.isfinite(period_d)):
        return np.nan, True
    M_tr = mean_anomaly_at_conjunction(
        e if np.isfinite(e) else 0.0, omega_star_deg)
    return float(t_peri + period_d * M_tr / (2 * np.pi)), True


# ── Time and sky position ─────────────────────────────────────────────

def _gmst_rad(jd):
    """Greenwich mean sidereal time.  Meeus eq. 12.4 (IAU 1982,
    Aoki et al. 1982, A&A 105, 359).  Agrees with astropy to 3.6"."""
    jd = np.asarray(jd, dtype=float)
    d = jd - _J2000
    T = d / 36525.0
    deg = (280.46061837 + 360.98564736629 * d
           + 0.000387933 * T ** 2 - T ** 3 / 38710000.0)
    return np.radians(np.mod(deg, 360.0))


def lst_rad(jd, lon_deg):
    return np.mod(_gmst_rad(jd) + np.radians(lon_deg), 2 * np.pi)


def sun_radec_rad(jd):
    """Low-precision solar position, Astronomical Almanac Section C.
    Quoted to 0.01 deg in ecliptic longitude, 1950-2050.  Returns
    coordinates OF DATE, which is what makes it consistent with GMST of
    date -- astropy's get_sun returns GCRS on J2000 axes, and feeding
    those into an hour angle introduces the accumulated precession."""
    jd = np.asarray(jd, dtype=float)
    n = jd - _J2000
    L = np.radians(np.mod(280.460 + 0.9856474 * n, 360.0))
    g = np.radians(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    return (np.mod(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)),
                   2 * np.pi),
            np.arcsin(np.sin(eps) * np.sin(lam)))


def altitude_deg(lat_deg, lst, ra_deg, dec_deg):
    """
    Altitude from the spherical law of cosines on the pole-zenith-star
    triangle (Meeus eq. 13.6):

        sin a = sin(dec) sin(lat) + cos(dec) cos(lat) cos(H)

    Exact given RA, Dec and LST.  Neglects refraction (~2' at 30 deg),
    diurnal parallax and nutation, all far below a 30 deg altitude cut.
    """
    lat = np.radians(lat_deg)
    ra = np.radians(np.atleast_1d(np.asarray(ra_deg, float)))[:, None]
    dec = np.radians(np.atleast_1d(np.asarray(dec_deg, float)))[:, None]
    H = np.atleast_1d(lst)[None, :] - ra
    s = (np.sin(dec) * np.sin(lat)
         + np.cos(dec) * np.cos(lat) * np.cos(H))
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def sun_altitude_deg(jd, lat_deg, lon_deg):
    lst = lst_rad(jd, lon_deg)
    sra, sdec = sun_radec_rad(jd)
    lat = np.radians(lat_deg)
    s = (np.sin(sdec) * np.sin(lat)
         + np.cos(sdec) * np.cos(lat) * np.cos(lst - sra))
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def airmass_to_altitude(airmass):
    """Altitude at which the plane-parallel airmass equals `airmass`."""
    return np.degrees(np.arcsin(1.0 / float(airmass)))


def declination_reach(lat_deg, airmass_max=2.0):
    """(dec_min, dec_max) observable at the airmass limit: lat +/- (90 - a_min).
    A site with |lat| >= 30 has no cutoff at the near pole."""
    a = airmass_to_altitude(airmass_max)
    return lat_deg - (90.0 - a), lat_deg + (90.0 - a)


# ── UTC -> BJD_TDB ────────────────────────────────────────────────────

def build_barycentric(times, lat_deg, lon_deg, height_m, step_days=2.0):
    """
    Precompute what is needed to turn JD_UTC into BJD_TDB, the scale the
    archive quotes pl_tranmid and pl_orbtper in.

    Two corrections, not interchangeable:

      CLOCK        TDB - UTC, about 69.2 s in 2027 (leap seconds plus
                   32.184 s).  Target-independent, effectively constant
                   over a year (the TDB-TT periodic term is ~1.7 ms).

      ROMER DELAY  observer -> solar system barycentre, (r_obs . n)/c.
                   +/- 500 s and TARGET-DEPENDENT: 16.6 min peak-to-peak
                   over a year on the ecliptic, 6.6 min at ecliptic
                   latitude ~60 deg.

    Subtracting a raw JD_UTC from a BJD_TDB skips both, leaving errors
    of up to ~8 min -- comparable to a typical out-of-transit baseline
    requirement, so it can move marginal windows in or out.

    r_obs includes the observatory's offset from geocentre, so the
    <= 21 ms topocentric term is kept.  The ephemeris is evaluated on a
    coarse grid and each target reduced to a dot product, because
    Time.light_travel_time() costs ~2 ms per epoch and would dominate
    the runtime over tens of thousands of epochs.  Validated against it
    to 30-75 ms over a full year.
    """
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                        height=height_m * u.m)
    jd = times.jd if hasattr(times, 'jd') else np.asarray(times, float)
    grid = np.arange(jd[0], jd[-1] + step_days, step_days)
    tc = Time(grid, format='jd', scale='utc', location=loc)

    # astropy warns that Earth-orientation (IERS) data does not extend to
    # future dates and falls back to a 50-year mean, costing arcsecond
    # accuracy in polar motion.  That is irrelevant here: an arcsecond of
    # Earth orientation moves the observer by about 9 m, which is 30
    # nanoseconds of light travel time, against a correction of up to
    # 500 seconds.  Silence it rather than print it on every run.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings('ignore', message='.*polar motion.*')
        _warnings.filterwarnings('ignore', message='.*IERS.*')
        r_earth = get_body_barycentric('earth', tc).xyz.to(u.au).value.T
        r_obs = tc.location.get_gcrs_posvel(tc)[0].xyz.to(u.au).value.T
    return dict(grid=grid,
                dtdb_s=float(tc[0].tdb.jd - tc[0].utc.jd) * _DAY_S,
                r=r_earth + r_obs)


def bjd_offset_days(bary, ra_deg, dec_deg, jd):
    """BJD_TDB - JD_UTC for one target, in days."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    n = np.array([np.cos(dec) * np.cos(ra),
                  np.cos(dec) * np.sin(ra),
                  np.sin(dec)])
    off_s = bary['dtdb_s'] + (bary['r'] @ n) * _AU_LIGHT_S
    return np.interp(np.asarray(jd, float), bary['grid'], off_s) / _DAY_S


# ── Contrast ──────────────────────────────────────────────────────────

def scale_height_contrast(rp_rearth, mp_mearth, rs_rsun, teq_k, mu_amu=2.3):
    """
    Transmission signal, the change in transit depth over one scale
    height (Snellen 2025):

        SHC = 2 pi R_p H / (pi R_star^2),   H = k T_eq / (mu g)

    mu is the weak link.  2.3 suits an H/He envelope; a rocky planet's
    secondary atmosphere is 18 (water) at best and 44 for CO2, shrinking
    H by a further factor ~2.4.  Line cores probe several scale heights,
    so this UNDERSTATES per-line depth by a factor of a few -- left
    uncorrected here because the factor is poorly constrained and, being
    a flat multiplier, changes no ordering.
    """
    Rp = rp_rearth * _R_EARTH
    Mp = mp_mearth * _M_EARTH
    Rs = rs_rsun * _RSUN_RE * _R_EARTH
    with np.errstate(invalid='ignore', divide='ignore'):
        g = _G_CGS * Mp / Rp ** 2
        H = _K_B * teq_k / (mu_amu * _M_U * g)
        return 2.0 * Rp * H / Rs ** 2


def emission_contrast(rp_rearth, rs_rsun, teq_k, teff_k, lam_um):
    """Dayside planet/star flux ratio, (R_p/R_s)^2 B(T_eq)/B(T_eff)."""
    def planck(T):
        lam = lam_um * 1e-4
        with np.errstate(over='ignore', invalid='ignore'):
            return 1.0 / (lam ** 5 * (np.exp(
                _H_ERG_S_C / (lam * _K_B * T)) - 1.0))
    Rs = rs_rsun * _RSUN_RE
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rp_rearth / Rs) ** 2 * planck(teq_k) / planck(teff_k)


_H_ERG_S_C = 6.62607015e-27 * 2.99792458e10
