"""
nlotsim web app.

    streamlit run app.py

Every setting that changes the answer is a visible control, with its
default shown and its source in the tooltip.
"""

import io
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from nlotsim import atmosphere, catalogue, events, instruments
from nlotsim.bands import BANDS, band_table, resolve_throughput
from nlotsim.events import DEFAULTS
from nlotsim.sites import SITES

st.set_page_config(page_title='nlotsim', layout='wide')
st.title('nlotsim')
st.write('How often you can observe a planet, when, and how strong the '
         'signal would be.')


@st.cache_data
def load():
    return catalogue.fetch()


df, prov = load()
st.caption(f"Catalogue: {prov.get('table', '?')}, "
           f"{prov.get('n_rows', '?')} planets, downloaded "
           f"{prov.get('retrieved_utc', 'unknown')}. It is kept as a file "
           "rather than fetched live, so the same input always gives the "
           "same answer.")

# ── sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Telescope')
    site_key = st.selectbox('Site', list(SITES),
                            format_func=lambda k: SITES[k]['name'])
    st.caption(SITES[site_key]['note'])
    band = st.selectbox('Band', list(BANDS), index=list(BANDS).index('K'))
    resolution = st.number_input('Resolving power R', 5000, 300000,
                                 int(SITES[site_key]['resolution']), 5000)
    area_m2 = st.number_input(
        'Collecting area (m^2)', 1.0, 2000.0,
        float(SITES[site_key]['area_m2']), 1.0,
        help='The real light-collecting area, not pi r squared. NLOT is '
             '121 m2, made of 90 segments of 1.35 m2 each. That collects '
             'as much light as a single 12.4 m mirror.')

    st.divider()
    st.header('How much light gets through')
    inst_key = st.selectbox(
        'Instrument', list(instruments.INSTRUMENTS) + ['flat'],
        format_func=lambda k: (instruments.INSTRUMENTS[k]['label']
                               if k != 'flat' else 'flat value'),
        help='Measured efficiency curves. These cover the instrument '
             'only. Mirror size is set above, so choosing an instrument '
             'here borrows its efficiency, not its telescope.')
    if inst_key == 'flat':
        instr = st.number_input('Instrument efficiency', 0.0001, 1.0,
                                0.02, 0.005, format='%.4f')
    else:
        instr = instruments.load(inst_key)
        st.caption(instruments.INSTRUMENTS[inst_key]['note'])
        if not instruments.covers(inst_key, band):
            st.warning(f'{instruments.INSTRUMENTS[inst_key]["label"]} does '
                       f'not reach across all of the {band} band. Numbers '
                       'outside its range are estimated.')
        st.caption(f'Instrument alone, average over {band}: '
                   f'{resolve_throughput(instr, band) * 100:.2f}%')

    apply_atm = st.checkbox(
        'Include the Earth atmosphere', True,
        help='The instrument curves leave the atmosphere out. This adds a '
             'measured sky transmission spectrum, the same way ratri.py '
             'does.')
    airmass_tr = st.slider(
        'Airmass for the sky transmission', 1.0, 2.0, 1.2, 0.1,
        disabled=not apply_atm,
        help='Looking further from overhead means more air to see '
             'through. Transmission is scaled as T(1) to the power of '
             'the airmass. The stored curve has 2.5 mm of water vapour.')
    tel_stat = st.radio(
        'Count absorbed light as', ['mean', 'median'], horizontal=True,
        disabled=not apply_atm,
        help='Mean counts every photon the sky absorbs. Median is closer '
             'to real practice, where the most strongly absorbed pixels '
             'are thrown away before analysis. The two differ by 2 to 3 '
             'per cent.')

    eta = instr
    if apply_atm:
        eta = atmosphere.combine(
            instr, atmosphere.transmission_at_airmass(airmass_tr, tel_stat))
        st.caption(f'Instrument and sky together, average over {band}: '
                   f'{resolve_throughput(eta, band) * 100:.2f}%')

    st.divider()
    st.header('Exposures')
    st.caption('Three things cap how long one exposure can be. The '
               'shortest of the three is used. Note that exposure length '
               'does not change the total signal to noise, because taking '
               'twice as many half-length exposures collects the same '
               'light. It matters only through the time lost to reading '
               'out the detector.')
    t_exp_max = st.number_input('Longest exposure you would use (s)',
                                1.0, 3600.0,
                                float(DEFAULTS['t_exp_max_s']), 10.0)
    t_duty = st.number_input(
        'Time lost reading out, per exposure (s)', 0.0, 600.0,
        float(SITES[site_key]['t_duty_s']), 5.0,
        help='ratri.py uses 60 s for a GIANO-like setup.')
    full_well = st.number_input(
        'Detector full well (electrons)', 1e3, 1e7,
        float(DEFAULTS['full_well_e']), 1e4, format='%.0f',
        help='How much charge a pixel holds before it saturates. This is '
             'a generic value, so replace it with the real detector '
             'number when you have it.')
    peak_pix = st.slider(
        'Share of the light landing in the brightest pixel', 0.1, 1.0,
        float(DEFAULTS['peak_pix_frac']), 0.05,
        help='With the full well above, this decides how bright a star '
             'has to be before exposures must be shortened.')
    smear = st.slider(
        'Allowed planet drift during one exposure', 0.25, 4.0,
        float(DEFAULTS['smear_resels']), 0.25,
        help='In resolution elements. If an exposure runs too long the '
             'planet moves, its lines blur together, and the effective '
             'resolution drops (Snellen 2025).')

    st.divider()
    st.header('Signal strength')
    n_lines = st.number_input(
        'Number of usable lines in the template', 10, 100000, 1000, 10,
        help='The planet signal grows as the square root of this, so it '
             'scales every answer up or down together without changing '
             'which planets rank highest. Published work ranges from '
             'about 56 lines for CO in the K band up to about 5000 for a '
             'methane template, so a factor of ten in signal to noise.')
    airmass_max = st.slider(
        'Highest airmass you would observe at', 1.5, 3.0,
        float(DEFAULTS['airmass_max']), 0.1,
        help='Airmass 2 means the target is 30 degrees above the '
             'horizon.')
    year = st.number_input('Start year', 2025, 2040, 2027)
    n_days = st.number_input('Days to cover', 30, 730, 365, 5)

# ── target ────────────────────────────────────────────────────────────

name = st.text_input('Planet', 'WASP-76 b')
prefill = st.checkbox('Fill the boxes from the catalogue', True)

params, assumed = None, []
if prefill:
    try:
        params, assumed = catalogue.to_params(catalogue.lookup(df, name), band)
    except (KeyError, ValueError) as exc:
        st.error(str(exc))

if params is None:
    params = dict(ra_deg=0.0, dec_deg=0.0, period_d=3.0,
                  t_conj_bjd=2460000.0, e=0.0, omega_star_deg=90.0,
                  t14_hr=3.0, a_au=0.04, incl_deg=90.0, rp_rearth=12.0,
                  mp_mearth=300.0, rs_rsun=1.0, teq_k=1500.0,
                  teff_k=5800.0, mag=9.0, rocky=False, epoch_derived=False)

if assumed:
    st.info('Some values were not in the catalogue, so these were '
            'assumed:\n\n' + '\n'.join(f'- {a}' for a in assumed))

FIELDS = [
    ('ra_deg', 'RA (deg)', '%.6f'), ('dec_deg', 'Dec (deg)', '%.6f'),
    ('period_d', 'Period (d)', '%.6f'),
    ('t_conj_bjd', 'Transit epoch (BJD_TDB)', '%.6f'),
    ('e', 'Eccentricity', '%.4f'),
    ('omega_star_deg', 'omega of the star (deg)', '%.2f'),
    ('t14_hr', 'Transit duration (hr)', '%.3f'),
    ('a_au', 'Orbit size a (AU)', '%.5f'),
    ('incl_deg', 'Inclination (deg)', '%.3f'),
    ('rp_rearth', 'Planet radius (Earth radii)', '%.3f'),
    ('mp_mearth', 'Planet mass (Earth masses)', '%.3f'),
    ('rs_rsun', 'Star radius (Solar radii)', '%.4f'),
    ('teq_k', 'Planet temperature (K)', '%.1f'),
    ('teff_k', 'Star temperature (K)', '%.1f'),
    ('mag', f'{band} magnitude', '%.3f'),
]

st.subheader('Numbers for this planet')
st.caption('You can change any of these. Two pairs need care. Period and '
           'transit epoch should come from the same paper, or the '
           'predicted times slowly drift. Eccentricity and omega are '
           'measured together, so change them together. Radius, '
           'temperature and magnitude can safely be taken from anywhere.')

# A box with a fixed key keeps its old value when the page reloads, so
# switching planets would leave the previous numbers on screen.  Tying
# the key to the planet gives each target its own boxes.
tag = f'{name.strip().lower()}|{band}|{int(prefill)}'

edited, vals = [], {}
cols = st.columns(3)
for i, (key, label, fmt) in enumerate(FIELDS):
    base = params.get(key)
    base = 0.0 if base is None or not np.isfinite(base) else float(base)
    v = cols[i % 3].number_input(label, value=base, format=fmt,
                                 key=f'{key}@{tag}')
    vals[key] = v
    if abs(v - base) > 1e-12:
        edited.append(label)

if edited:
    st.warning('You changed: ' + ', '.join(edited) +
               '. The results below no longer match the catalogue entry.')

# ── run ───────────────────────────────────────────────────────────────

if st.button('Calculate', type='primary'):
    site_cfg = dict(SITES[site_key], throughput=eta,
                    area_m2=float(area_m2), t_duty_s=float(t_duty))
    cfg = dict(n_lines=float(n_lines), airmass_max=float(airmass_max),
               t_exp_max_s=float(t_exp_max), full_well_e=float(full_well),
               peak_pix_frac=float(peak_pix), smear_resels=float(smear))

    res = events.observability(
        **vals, site=site_cfg, band=band, resolution=int(resolution),
        start=f'{int(year)}-01-01T00:00:00', n_days=int(n_days),
        rocky=params.get('rocky', False), cfg=cfg,
        epoch_derived=params.get('epoch_derived', False),
        edited=bool(edited))

    if res['errors']:
        for e_ in res['errors']:
            st.error(e_)
        st.stop()
    for w_ in res['warnings']:
        st.warning(w_)

    t_exp, which, limits = events.exposure_time(
        vals['mag'], band, site_cfg, int(resolution), res['kp_kms'],
        vals['period_d'], cfg, detail=True)
    plain = {'ceiling': 'the longest exposure you allowed',
             'saturation': 'the detector filling up',
             'smearing': 'the planet moving during the exposure'}

    a, b, c = st.columns(3)
    a.metric('Planet orbital speed K_p (km/s)', f"{res['kp_kms']:.1f}")
    b.metric('Hours the target is up and dark',
             f"{res['hours_observable']:.0f}")
    c.metric('Speed change needed (km/s)', f"{res['dv_floor_kms']:.1f}")
    a.metric('Exposure length (s)', f'{t_exp:.0f}')
    b.metric('Fraction of time collecting light',
             f'{t_exp / (t_exp + t_duty):.2f}')
    c.metric('One resolution element (km/s)',
             f'{BANDS[band].resel_kms(resolution):.1f}')
    st.caption(f'Exposure length is set by {plain[which]}. The three caps '
               'are: ' + ', '.join(f'{k} {v:.0f} s'
                                   for k, v in limits.items()) +
               '. The shortest one wins.')

    tables = {}
    for ch, title in [('transmission', 'Starlight through the atmosphere '
                                       '(transit)'),
                      ('emission', 'Light from the planet dayside')]:
        d = res[ch]
        st.subheader(f'{title}: {d["n_events"]} chances per year')
        if not d['n_events']:
            st.write('None in this period.')
            continue

        x, y, z = st.columns(3)
        x.metric('Exposures in one observation',
                 f"{d.get('n_exposures', 0)}")
        y.metric('Star signal to noise, one exposure',
                 f"{d.get('snr_star_per_exposure', 0):.0f}")
        z.metric('Star signal to noise, whole observation',
                 f"{d.get('snr_star', 0):.0f}")
        x.metric('Planet signal to noise, one observation',
                 f"{d.get('snr_planet', float('nan')):.2f}")
        y.metric('Planet signal to noise, a year of them',
                 f"{d.get('snr_planet_year', float('nan')):.1f}")
        z.metric('Typical length of one observation (hr)',
                 f"{d.get('block_hr_median', 0):.1f}")
        st.caption('A whole observation is worth the square root of the '
                   'number of exposures more than a single one. The '
                   'yearly figure assumes every chance is taken and they '
                   'add up perfectly, so treat it as a best case.')

        tbl = pd.DataFrame({
            'date (UTC)': events.dates_to_iso(d['dates']),
            'BJD_TDB': np.round(d['dates'], 5),
            'speed change (km/s)': np.round(d['dv_kms'], 1),
            'length (hr)': np.round(d['block_hr'], 2)})
        tables[ch] = tbl
        st.dataframe(tbl, width='stretch', hide_index=True)

    st.caption('Comparing planets with each other is far safer than '
               'trusting the exact numbers. The line count and the '
               'efficiency both scale every answer at once.')

    # ---- figure ------------------------------------------------------
    fig = None
    if not np.isscalar(eta):
        st.subheader('Across the wavelength range')
        block = (res['emission'].get('block_hr_median')
                 or res['transmission'].get('block_hr_median') or 3.0)
        wl_s, snr_s = events.snr_spectrum(
            vals['mag'], band, site_cfg, int(resolution), hours=block,
            t_exp_s=t_exp, teff_k=vals['teff_k'], cfg=cfg)

        fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                               height_ratios=[2, 1])
        ax[0].plot(wl_s, snr_s, lw=0.7, color='#4878CF')
        bb = BANDS[band]
        ax[0].axvspan(bb.lam_um - bb.width_um / 2,
                      bb.lam_um + bb.width_um / 2,
                      color='#C44E52', alpha=0.10, label=f'{band} band')
        ax[0].set_ylabel(f'Star signal to noise\nper resolution element'
                         f'\n({block:.1f} hr, R = {resolution:,})')
        ax[0].legend(fontsize=8, loc='upper left')
        ax[0].grid(lw=0.3, alpha=0.35, ls='--')
        if not np.isscalar(instr):
            ax[1].plot(instr[0], instr[1] * 100, lw=0.7, color='#55A868',
                       label='instrument')
        ax[1].plot(eta[0], eta[1] * 100, lw=0.7, color='#333333',
                   label='with the sky' if apply_atm else 'total')
        ax[1].set_ylabel('light getting through (%)')
        ax[1].set_xlabel('wavelength (micron)')
        ax[1].legend(fontsize=8)
        ax[1].grid(lw=0.3, alpha=0.35, ls='--')
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            'This is the signal to noise on the STAR, per resolution '
            'element, for one observation of the length given above. '
            'The star is treated as a simple blackbody at '
            f'{vals["teff_k"]:.0f} K scaled to its {band} magnitude, so '
            'it has no absorption lines and is least accurate at short '
            'wavelengths. The gaps are between the detector orders, '
            'where no light is recorded. The two deep dips near 1.4 and '
            '1.9 micron are water in the Earth atmosphere.')

    # ---- downloads ---------------------------------------------------
    st.subheader('Download')
    summary = dict(
        planet=name, site=SITES[site_key]['name'], band=band,
        instrument=inst_key, resolution=int(resolution),
        area_m2=float(area_m2), atmosphere_included=bool(apply_atm),
        airmass_for_transmission=float(airmass_tr) if apply_atm else None,
        telluric_statistic=tel_stat if apply_atm else None,
        band_mean_efficiency=float(resolve_throughput(eta, band)),
        exposure_s=float(t_exp), exposure_limited_by=which,
        readout_s=float(t_duty), n_lines=float(n_lines),
        airmass_max=float(airmass_max), start_year=int(year),
        days=int(n_days), parameters=vals, edited_fields=edited,
        assumed=assumed, warnings=res['warnings'],
        catalogue_retrieved=prov.get('retrieved_utc'),
        results={k: {kk: vv for kk, vv in res[k].items()
                     if not isinstance(vv, list)}
                 for k in ['transmission', 'emission']})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('settings_and_results.json',
                   json.dumps(summary, indent=2, default=str))
        for ch, tbl in tables.items():
            z.writestr(f'{ch}_dates.csv', tbl.to_csv(index=False))
        if fig is not None:
            png = io.BytesIO()
            fig.savefig(png, format='png', dpi=200, bbox_inches='tight')
            z.writestr('snr_vs_wavelength.png', png.getvalue())
            pdf = io.BytesIO()
            fig.savefig(pdf, format='pdf', bbox_inches='tight')
            z.writestr('snr_vs_wavelength.pdf', pdf.getvalue())
            z.writestr('snr_vs_wavelength.csv',
                       pd.DataFrame({'wavelength_um': np.round(wl_s, 6),
                                     'snr_per_resolution_element':
                                         np.round(snr_s, 2)}
                                    ).to_csv(index=False))
    st.download_button(
        'Download everything as a zip', buf.getvalue(),
        file_name=f"nlotsim_{name.replace(' ', '_')}_{band}.zip",
        mime='application/zip',
        help='Dates, the figure as PNG and PDF, the plotted numbers as '
             'CSV, and a JSON record of every setting used.')

with st.expander('Instrument efficiencies'):
    st.dataframe(pd.DataFrame(instruments.table()), hide_index=True)
    st.caption('Averages over each band, instrument only, with the sky '
               'left out. Blank means the instrument does not reach '
               'across that band.')

with st.expander('Brightness scale'):
    st.dataframe(pd.DataFrame(band_table()), hide_index=True)
    st.caption('Zero points from Cohen, Wheaton and Megeath 2003, '
               'AJ 126, 1090, the 2MASS calibration. Their flux is given '
               'per square centimetre, not per square metre.')
