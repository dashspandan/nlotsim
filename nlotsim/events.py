"""
Single-planet observability: when the events are, how many, and the
optimistic planet SNR.

The whole package funnels through observability(), so the CLI, the web
app and any batch script share one implementation.

CRITERIA  (they differ by channel, deliberately)

  TRANSMISSION  a contiguous observable block covering at least
      transit_frac_min of T14 plus oot_min_minutes of out-of-transit on
      at least one side, and NO velocity requirement: planet-free
      out-of-transit frames provide the detrending basis, so even
      sub-pixel shifts are recoverable (Cheverall & Madhusudhan 2024;
      Dash et al. 2024).

  EMISSION  a contiguous block of at least emission_block_hr within the
      dayside windows FLANKING secondary eclipse (not at quadrature --
      phi = 0.25 and 0.75 are not in these windows), achieving
      |dv_pl| above the floor.  With no planet-free reference the
      velocity sweep is the whole basis for separating planet from
      tellurics.  The sweep is measured over the best-placed window of
      at most standard_block_hr, a CAP not a requirement, so a long
      stare earns no credit beyond it while a short-period planet whose
      entire dayside window is a couple of hours is not excluded.

VELOCITY FLOOR
      3 x the resolution element (Hong, Finnerty & Fitzgerald 2025,
      arXiv:2505.09781 sec. 4): below that the planet signal is removed
      by median division and detrending.  A FLOOR, not a detection
      threshold -- their 6-sigma limits are 3-7x stricter.  It scales
      with resolution, which is what makes a resolution sweep
      meaningful.

KNOWN LIMITS, stated rather than hidden
  - Frames taken DURING secondary eclipse are planet-free in exactly
    the way out-of-transit frames are, so bracketing the eclipse should
    relax the velocity requirement.  Raised in Dash et al. 2026; no
    published HRCCS detection exploits it, so it is not modelled and
    the emission numbers are conservative for planets whose eclipse is
    observable.
  - A short-period planet passes through the dayside windows several
    times per night.  Passes are counted separately, never concatenated,
    though they could be co-added within the night: the sweep would not
    improve but the SNR would, as sqrt(passes).
  - Because a dayside window spans 0.15 in phase, it lasts
    emission_block_hr only if P >= 0.556 d at the default 2 h, so
    shorter-period planets are structurally excluded from the emission
    channel.
"""

import numpy as np
import astropy.units as u
from astropy.time import Time

from . import physics as P
from .bands import BANDS, resolve_throughput
from .sites import get_site

DEFAULTS = dict(
    airmass_max=2.0,
    sun_alt_max=-18.0,
    timestep_min=10.0,
    transit_frac_min=0.50,
    oot_min_minutes=30.0,
    emission_block_hr=2.0,
    standard_block_hr=4.0,
    emission_phases=((0.30, 0.45), (0.55, 0.70)),
    dv_floor_resels=3.0,
    t_exp_max_s=300.0,
    full_well_e=1.0e5,
    peak_pix_frac=0.5,
    smear_resels=1.0,
    n_lines=1000.0,
    mu_gas_rich=2.3,
    mu_rocky=18.0,
)


# ── input validation ──────────────────────────────────────────────────

def validate(p):
    """
    Return (errors, warnings).  Errors are unphysical; warnings are
    legal but worth a second look.
    """
    err, warn = [], []

    def num(k):
        v = p.get(k)
        return v if (v is not None and np.isfinite(v)) else None

    if num('period_d') is None or p['period_d'] <= 0:
        err.append('period must be positive')
    if num('dec_deg') is None or not -90 <= p['dec_deg'] <= 90:
        err.append('declination must be between -90 and +90')
    if num('ra_deg') is None or not 0 <= p['ra_deg'] < 360:
        err.append('right ascension must be in [0, 360)')
    e = num('e')
    if e is not None and not 0 <= e < 1:
        err.append('eccentricity must satisfy 0 <= e < 1')
    for k, lab in [('rp_rearth', 'planet radius'), ('rs_rsun', 'stellar radius'),
                   ('teq_k', 'equilibrium temperature'),
                   ('teff_k', 'stellar effective temperature')]:
        v = num(k)
        if v is not None and v <= 0:
            err.append(f'{lab} must be positive')

    if e is not None and e > 0.9:
        warn.append(f'e = {e:.2f} is extreme; the Kepler solve may be '
                    'poorly conditioned')
    kp = num('kp_kms')
    if kp is not None and kp > 400:
        warn.append(f'K_p = {kp:.0f} km/s is very large; check a and P')
    t14, per = num('t14_hr'), num('period_d')
    if t14 and per and t14 / 24.0 > per / 3:
        warn.append('transit duration exceeds a third of the period; '
                    'check units (T14 is in hours)')
    if p.get('epoch_derived'):
        warn.append('conjunction epoch was derived from periastron rather '
                    'than read from pl_tranmid: pscomppars is a composite '
                    'table, so these may come from different sources. '
                    'Phases are less reliable.')
    if p.get('_edited'):
        warn.append('parameters edited: period and epoch must come from the '
                    'same reference or phases drift, and e and omega are '
                    'fitted jointly so should be changed together.')
    return err, warn


# ── exposure time ─────────────────────────────────────────────────────

def exposure_time(mag, band, site, resolution, kp_kms=None, period_d=None,
                  cfg=None, detail=False):
    """
    The smaller of three limits: a ceiling, detector saturation, and
    smearing (the planet must not drift more than smear_resels within
    one exposure -- Snellen 2025 sec. 2.4, first listed pitfall).

    Note t_exp CANCELS from the photon-limited SNR, since
    sqrt(rate t_exp) sqrt(T/t_exp) = sqrt(rate T).  It matters only
    through the duty cycle, because each exposure costs a fixed
    readout.  There it matters a lot: a bright star forced to short
    exposures spends most of the night reading out.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    b = BANDS[band]
    eta = resolve_throughput(site['throughput'], band)
    rate = b.photon_rate(mag, site['area_m2'] * 1e4, eta, resolution)

    # 1. a ceiling we would never exceed regardless
    limits = {'ceiling': float(cfg['t_exp_max_s'])}

    # 2. saturation: full well / peak-pixel photon rate
    with np.errstate(divide='ignore', invalid='ignore'):
        limits['saturation'] = float(
            cfg['full_well_e'] / (rate * cfg['peak_pix_frac']))

    # 3. smearing: the planet must not drift more than smear_resels
    #    within one exposure, else its lines blend and the effective
    #    resolving power drops (Snellen 2025 sec. 2.4, first pitfall).
    #    |dv_p/dt| ~ 2 pi K_p / P, evaluated at conjunction.
    if kp_kms and period_d and np.isfinite(kp_kms):
        dvdt = abs(kp_kms) * 2 * np.pi / (period_d * 86400.0)
        if dvdt > 0:
            limits['smearing'] = float(
                cfg['smear_resels'] * b.resel_kms(resolution) / dvdt)

    which = min(limits, key=limits.get)
    t = max(min(limits.values()), 1.0)
    if detail:
        return t, which, limits
    return t


def n_exposures(hours, t_exp_s, t_duty_s):
    """
    How many exposures fit in `hours`, following ratri.py:

        n = (T + t_duty) / (t_exp + t_duty)

    Each exposure costs its own integration plus one readout.
    """
    T = max(hours, 0.0) * 3600.0
    return int(max((T + t_duty_s) / (t_exp_s + t_duty_s), 0))


def snr_star(mag, band, site, resolution, hours, t_exp_s, cfg=None,
             per_exposure=False):
    """
    Photon-limited stellar SNR per resolution element.

    By default this is the total over `hours`, with the open-shutter
    time reduced by the duty cycle.  With per_exposure=True it is the
    SNR of a single exposure instead, so that

        SNR_total = SNR_exposure * sqrt(n_exposures)
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    eta = resolve_throughput(site['throughput'], band)
    rate = BANDS[band].photon_rate(mag, site['area_m2'] * 1e4, eta,
                                   resolution)
    if per_exposure:
        return float(np.sqrt(rate * t_exp_s))
    duty = t_exp_s / (t_exp_s + site['t_duty_s'])
    return float(np.sqrt(rate * max(hours, 0.0) * 3600.0 * duty))


# ── the main entry point ──────────────────────────────────────────────

def observability(ra_deg, dec_deg, period_d, t_conj_bjd,
                  e=0.0, omega_star_deg=90.0, t14_hr=None,
                  kp_kms=None, a_au=None, incl_deg=90.0,
                  rp_rearth=None, mp_mearth=None, rs_rsun=None,
                  teq_k=None, teff_k=None, mag=None,
                  site='NLOT', band='K', resolution=None,
                  start='2027-01-01T00:00:00', n_days=365,
                  rocky=False, cfg=None, epoch_derived=False,
                  edited=False):
    """
    Everything for one planet from one site in one band.

    Returns a dict with the event counts, their dates, the velocity
    sweeps, the SNRs, and the validation messages.  All times in and
    out are BJD_TDB; the UTC grid is converted internally.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    s = get_site(site)
    R = resolution or s['resolution']
    b = BANDS[band]

    params = dict(ra_deg=ra_deg, dec_deg=dec_deg, period_d=period_d,
                  e=e, rp_rearth=rp_rearth, rs_rsun=rs_rsun, teq_k=teq_k,
                  teff_k=teff_k, t14_hr=t14_hr, kp_kms=kp_kms,
                  epoch_derived=epoch_derived, _edited=edited)
    errors, warnings = validate(params)
    out = dict(errors=errors, warnings=warnings, site=s['name'], band=band,
               resolution=R)
    if errors:
        return out

    if kp_kms is None or not np.isfinite(kp_kms):
        if a_au is None or not np.isfinite(a_au):
            out['errors'] = ['need either K_p or the semi-major axis']
            return out
        kp_kms = P.planet_kp(a_au, period_d, e, incl_deg)
    out['kp_kms'] = float(kp_kms)

    # time grid, and the UTC -> BJD_TDB conversion
    step = cfg['timestep_min']
    n = int(round(n_days * 24 * 60 / step))
    # astropy warns if a bare number is added to a Time, since it has to
    # guess the unit.  Say "days" explicitly.
    t0 = Time(start, scale='utc')
    times = t0 + np.arange(n) * (step / 1440.0) * u.day
    jd = times.jd
    bary = P.build_barycentric(times, s['lat'], s['lon'], s['height'])
    bjd = jd + P.bjd_offset_days(bary, ra_deg, dec_deg, jd)

    # observable mask
    alt_min = P.airmass_to_altitude(cfg['airmass_max'])
    lst = P.lst_rad(jd, s['lon'])
    alt = P.altitude_deg(s['lat'], lst, [ra_deg], [dec_deg])[0]
    night = P.sun_altitude_deg(jd, s['lat'], s['lon']) < cfg['sun_alt_max']
    obs = (alt >= alt_min) & night
    out['hours_observable'] = float(obs.sum() * step / 60.0)

    floor = cfg['dv_floor_resels'] * b.resel_kms(R)
    out['dv_floor_kms'] = float(floor)
    phase = np.mod((bjd - t_conj_bjd) / period_d, 1.0)

    # ---- transmission ------------------------------------------------
    tr = dict(n_events=0, dates=[], dv_kms=[], block_hr=[])
    if t14_hr and np.isfinite(t14_hr) and t14_hr > 0:
        half = (t14_hr / 24.0) / 2.0 + cfg['oot_min_minutes'] / 1440.0
        k0 = int(np.floor((bjd[0] - t_conj_bjd) / period_d))
        k1 = int(np.ceil((bjd[-1] - t_conj_bjd) / period_d))
        for kk in range(k0, k1 + 1):
            mid = t_conj_bjd + kk * period_d
            if mid - half < bjd[0] or mid + half > bjd[-1]:
                continue
            w = (bjd >= mid - half) & (bjd <= mid + half)
            if not w.any():
                continue
            m = obs[w]
            if not m.any():
                continue
            idx = np.flatnonzero(w)
            runs = np.split(idx[m], np.flatnonzero(np.diff(idx[m]) != 1) + 1)
            best = max(runs, key=len)
            in_tr = np.abs(bjd[best] - mid) <= (t14_hr / 24.0) / 2.0
            frac = in_tr.sum() * step / (t14_hr * 60.0)
            oot = (~in_tr).sum() * step
            if frac >= cfg['transit_frac_min'] and oot >= cfg['oot_min_minutes']:
                v = P.planet_rv(phase[best[[0, -1]]], kp_kms, e, omega_star_deg)
                tr['n_events'] += 1
                tr['dates'].append(float(mid))
                tr['dv_kms'].append(float(abs(v[1] - v[0])))
                tr['block_hr'].append(float(len(best) * step / 60.0))
    out['transmission'] = tr

    # ---- emission ----------------------------------------------------
    inwin = np.zeros_like(obs)
    for lo, hi in cfg['emission_phases']:
        inwin |= (phase >= lo) & (phase <= hi)
    usable = obs & inwin
    need = int(np.ceil(cfg['emission_block_hr'] * 60 / step))
    nfix = int(round(cfg['standard_block_hr'] * 60 / step))

    em = dict(n_events=0, dates=[], dv_kms=[], block_hr=[])
    if usable.any():
        idx = np.flatnonzero(usable)
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
            if len(seg) < need:
                continue
            nwin = min(nfix, len(seg))
            best_dv = 0.0
            for a0 in range(0, len(seg) - nwin + 1):
                v = P.planet_rv(phase[seg[[a0, a0 + nwin - 1]]], kp_kms,
                                e, omega_star_deg)
                best_dv = max(best_dv, abs(v[1] - v[0]))
            if best_dv >= floor:
                em['n_events'] += 1
                em['dates'].append(float(bjd[seg[0]]))
                em['dv_kms'].append(float(best_dv))
                em['block_hr'].append(float(nwin * step / 60.0))
    out['emission'] = em

    # ---- SNR ---------------------------------------------------------
    if mag is not None and np.isfinite(mag):
        t_exp = exposure_time(mag, band, s, R, kp_kms, period_d, cfg)
        out['t_exp_s'] = float(t_exp)
        out['duty_cycle'] = float(t_exp / (t_exp + s['t_duty_s']))

        mu = cfg['mu_rocky'] if rocky else cfg['mu_gas_rich']
        shc = fpfs = np.nan
        if all(v is not None and np.isfinite(v)
               for v in [rp_rearth, mp_mearth, rs_rsun, teq_k]):
            shc = float(P.scale_height_contrast(rp_rearth, mp_mearth,
                                                rs_rsun, teq_k, mu))
        if all(v is not None and np.isfinite(v)
               for v in [rp_rearth, rs_rsun, teq_k, teff_k]):
            fpfs = float(P.emission_contrast(rp_rearth, rs_rsun, teq_k,
                                             teff_k, b.lam_um))
        out['scale_height_contrast'] = shc
        out['emission_contrast'] = fpfs

        sqrtN = np.sqrt(cfg['n_lines'])
        for ch, con in [('transmission', shc), ('emission', fpfs)]:
            blocks = out[ch]['block_hr']
            if blocks:
                hb = float(np.median(blocks))
                ss = snr_star(mag, band, s, R, hb, t_exp, cfg)
                out[ch]['snr_star'] = ss
                out[ch]['n_exposures'] = n_exposures(hb, t_exp,
                                                     s['t_duty_s'])
                out[ch]['snr_star_per_exposure'] = snr_star(
                    mag, band, s, R, hb, t_exp, cfg, per_exposure=True)
                out[ch]['block_hr_median'] = hb
                out[ch]['snr_planet'] = (float(con * ss * sqrtN)
                                         if np.isfinite(con) else np.nan)
                # optimistic sqrt(N) co-add over the year: an UPPER
                # BOUND, since tellurics vary night to night, the
                # detrending basis differs, ephemeris error smears the
                # Kp-vsys peak, and model mismatch does not average down
                out[ch]['snr_planet_year'] = (
                    out[ch]['snr_planet'] * np.sqrt(out[ch]['n_events'])
                    if np.isfinite(out[ch]['snr_planet']) else np.nan)
    return out


def snr_spectrum(mag, band, site, resolution, hours, t_exp_s=None,
                 teff_k=5800.0, cfg=None, wl_grid=None):
    """
    Stellar SNR per resolution element as a function of wavelength --
    the CRIRES+-simulator style output.

    The star's spectral shape is a BLACKBODY at teff_k, normalised so
    that at the reference band's isophotal wavelength it reproduces the
    photon rate implied by `mag`.  That is a real approximation and
    should be read as such: a blackbody has no absorption lines, and it
    departs from a true stellar SED most at the blue end.  Across J to
    K for a cool star it is adequate for showing where the instrument
    and the atmosphere let light through, which is what this plot is
    for.  It is NOT a substitute for a synthetic spectrum.

    Returns (wl_um, snr_per_resel).  The wavelength grid is the
    throughput curve's own, so inter-order gaps appear as gaps.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    b = BANDS[band]
    eta = site['throughput']
    if np.isscalar(eta):
        raise ValueError('snr_spectrum needs a throughput CURVE, not a '
                         'scalar: a flat efficiency would make the plot '
                         'meaningless')
    if isinstance(eta, dict):
        wl, tp = np.asarray(eta['wl_um'], float),                  np.asarray(eta['throughput'], float)
    else:
        wl, tp = (np.asarray(x, float) for x in eta)
    if wl_grid is not None:
        tp = np.interp(wl_grid, wl, tp)
        wl = np.asarray(wl_grid, float)

    if t_exp_s is None:
        t_exp_s = exposure_time(mag, band, site, resolution, cfg=cfg)

    # photon rate at the reference wavelength, from the magnitude
    ref_rate = b.photon_rate(mag, site['area_m2'] * 1e4,
                             resolve_throughput(site['throughput'], band),
                             resolution)

    # blackbody PHOTON spectral shape, B_lambda / (hc/lambda), relative
    # to the reference wavelength
    def nphot(lam_um):
        h, c, k = 6.62607015e-27, 2.99792458e10, 1.380649e-16
        lam = np.asarray(lam_um, float) * 1e-4
        with np.errstate(over='ignore'):
            return 1.0 / (lam ** 4 * (np.exp(h * c / (lam * k * teff_k)) - 1))

    shape = nphot(wl) / nphot(b.lam_um)
    # per-resolution-element width scales with wavelength, and the
    # throughput at the reference was already folded into ref_rate
    width = (wl / b.lam_um)
    eta_rel = tp / resolve_throughput(site['throughput'], band)

    duty = t_exp_s / (t_exp_s + site['t_duty_s'])
    t_open = max(hours, 0.0) * 3600.0 * duty
    return wl, np.sqrt(ref_rate * shape * eta_rel * width * t_open)


def dates_to_iso(bjd_list):
    """BJD_TDB -> ISO strings, for display."""
    if not bjd_list:
        return []
    return [t.isot for t in Time(np.asarray(bjd_list), format='jd',
                                 scale='tdb')]
