"""
Photometric bands and their Vega zero points.

ZERO POINTS
    J, H, Ks from Cohen, Wheaton & Megeath 2003, AJ 126, 1090, Table 2
    (the 2MASS absolute calibration).  Their F_lambda is quoted in
    W cm^-2 um^-1 -- per CM^2, not per m^2, which is an easy and
    expensive mistake:

        F_lambda [erg s^-1 cm^-2 A^-1] = F_lambda [W cm^-2 um^-1] x 1e3

    Y is omitted: it is not a 2MASS band, so the archive has no
    magnitude to prefill from.  See the note below BANDS for how to
    restore it if you need He I 10830 A.

WHICH BAND TO USE
    For EMISSION the contrast B(T_p)/B(T_star) climbs steeply with
    wavelength, because the star is near Rayleigh-Jeans across the
    near-IR while the planet is still on the Wien side.  Relative to K,
    the net planet SNR (contrast x stellar photons) is roughly

        T_p = 1500 K :  J 0.14,  H 0.47,  K 1.00
        T_p = 2900 K :  J 0.65,  H 0.89,  K 1.00

    so K wins at every temperature, decisively for cool planets.  J and
    H do deliver more stellar photons (1.3x and 1.2x in SNR for a G
    dwarf) but nowhere near enough to compensate.

    For TRANSMISSION the scale-height contrast 2 R_p H / R_star^2 is
    geometric and carries no explicit wavelength dependence, so the
    choice is driven by which species are accessible and how bad the
    tellurics are, not by this calculation.

    L band would beat K for emission (2.7x at 1500 K, 10.7x at 800 K)
    but lies outside a GIANO-like 0.95-2.45 um range, and the thermal
    background assumption below fails there.
"""

import numpy as np

_H_ERG_S = 6.62607015e-27
_C_CM_S = 2.99792458e10


class Band:
    """A photometric band: isophotal wavelength and Vega zero point."""

    def __init__(self, name, lam_um, f_lam_w_cm2_um, source, width_um=None):
        self.name = name
        self.lam_um = lam_um
        self.lam_a = lam_um * 1e4
        self.width_um = width_um
        self.source = source
        # W cm^-2 um^-1 -> erg s^-1 cm^-2 A^-1
        self.f_lam_cgs = f_lam_w_cm2_um * 1e7 * 1e-4
        self.e_photon = _H_ERG_S * _C_CM_S / (lam_um * 1e-4)
        # photons s^-1 cm^-2 A^-1 at zero magnitude
        self.zp_photons = self.f_lam_cgs / self.e_photon

    def photon_rate(self, mag, area_cm2, throughput, resolution):
        """Photons per second per resolution element from a star."""
        res_el_a = self.lam_a / float(resolution)
        return (self.zp_photons * 10.0 ** (-np.asarray(mag, float) / 2.5)
                * area_cm2 * throughput * res_el_a)

    def resel_kms(self, resolution):
        return 299792.458 / float(resolution)

    def __repr__(self):
        return (f'<Band {self.name}: {self.lam_um:.3f} um, '
                f'zp {self.zp_photons:.2f} ph/s/cm2/A>')


BANDS = {
    # Hewett et al. 2006: lambda_eff 1.0305 um, F_nu 2026 Jy,
    # F_lambda 5.71e-9 W m^-2 um^-1 = 5.71e-13 W cm^-2 um^-1.
    # (An earlier version had 2.026e-13 here, which was the 2026 Jy
    # figure with an exponent stuck on it: low by a factor 2.82, so Y
    # signal to noise was low by 1.68.)
    'Y': Band('Y', 1.0305, 5.71e-13, width_um=0.102,
              source='Hewett et al. 2006, MNRAS 367, 454 (UKIRT/WFCAM)'),
    'J': Band('J', 1.235, 3.129e-13, width_um=0.162,
              source='Cohen et al. 2003, AJ 126, 1090 (2MASS)'),
    'H': Band('H', 1.662, 1.133e-13, width_um=0.251,
              source='Cohen et al. 2003, AJ 126, 1090 (2MASS)'),
    'K': Band('K', 2.159, 4.283e-14, width_um=0.262,
              source='Cohen et al. 2003, AJ 126, 1090 (2MASS Ks)'),
}

DEFAULT_BAND = 'K'

# A NOTE ON Y.
#
# The instrument curves here all reach shortward of 0.98 um, so Y is
# covered on the instrument side.  What the NASA Exoplanet Archive does
# NOT carry is a Y magnitude: it holds 2MASS photometry only (J, H, Ks).
# So a Y calculation is prefilled from J and flagged, and the user can
# type a real Y magnitude over it.  For a solar-type star Y - J is a few
# tenths of a magnitude.
#
# Y is the weakest band for emission, roughly 3.6% of K's contrast at
# 1500 K, but it contains He I 10830 A, the standard probe of escaping
# atmospheres, and for transmission the scale-height contrast carries no
# wavelength penalty.
#
# UKIRT/WFCAM is a different photometric system from 2MASS, so a Y and a
# J magnitude for one star are not on a common scale to better than a
# few per cent.  There is also a definitional difference: the 2MASS
# response curves include the atmosphere and detector QE, while the
# WFCAM ones do not.  Since this package applies atmospheric
# transmission separately, the Y figures are marginally cleaner in that
# respect than J, H and K, which carry a little atmosphere twice.  The
# effect is small, a few per cent, but it is there.


def resolve_throughput(throughput, band, mode='band_mean'):
    """
    Turn a throughput specification into a single number for `band`.

    `throughput` may be either

      a float                 used as-is, wavelength-independent
      (wl_um, values)         two array-likes, a measured curve
      {'wl_um': [...], 'throughput': [...]}   the same as a dict

    A measured curve is much better than a scalar: real throughput
    varies strongly across the near-IR, and a flat value silently
    over- or under-states whole bands.  With the zero points fixed by
    Cohen+2003 this is the remaining free parameter, so it sets the
    absolute SNR scale outright.

    mode='band_mean'  average over the band's full width.  The honest
                      default: SNR here is quoted per resolution
                      element but is meant to represent the band.
    mode='isophotal'  interpolate at the isophotal wavelength only.
                      Consistent with the single-wavelength treatment
                      used elsewhere, but blind to structure across the
                      band.

    NOTE the curve must be end-to-end from the top of the atmosphere --
    telescope, instrument, fibre, detector QE and mean atmospheric
    transmission.  Mixing an instrument-only curve with these zero
    points overstates the SNR, by a large factor for a fibre-fed
    single-mode system where injection losses dominate.
    """
    if throughput is None:
        raise ValueError('throughput is required')
    if np.isscalar(throughput):
        return float(throughput)

    if isinstance(throughput, dict):
        wl = np.asarray(throughput['wl_um'], float)
        tp = np.asarray(throughput['throughput'], float)
    else:
        wl, tp = (np.asarray(x, float) for x in throughput)
    if wl.shape != tp.shape or wl.size < 2:
        raise ValueError('throughput curve needs matching arrays of '
                         'length >= 2')
    order = np.argsort(wl)
    wl, tp = wl[order], tp[order]

    b = BANDS[band]
    if mode == 'isophotal' or b.width_um is None:
        return float(np.interp(b.lam_um, wl, tp))

    lo, hi = b.lam_um - b.width_um / 2, b.lam_um + b.width_um / 2
    if wl[0] > lo or wl[-1] < hi:
        # curve does not span the band: fall back rather than
        # extrapolate, and say so
        import warnings as _w
        _w.warn(f'throughput curve spans {wl[0]:.3f}-{wl[-1]:.3f} um but '
                f'band {band} needs {lo:.3f}-{hi:.3f} um; using the '
                'isophotal value instead of a band mean')
        return float(np.interp(b.lam_um, wl, tp))

    grid = np.linspace(lo, hi, 200)
    return float(np.trapezoid(np.interp(grid, wl, tp), grid)
                 / (hi - lo))


def band_table():
    """Readable summary, for the app sidebar and the CLI banner."""
    rows = []
    for k in BANDS:
        b = BANDS[k]
        rows.append(dict(band=k, lam_um=b.lam_um,
                         zp_photons=round(b.zp_photons, 2),
                         source=b.source))
    return rows
