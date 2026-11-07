#!/usr/bin/env python3
"""
nlotsim command line.

    python cli.py "WASP-76 b"
    python cli.py "HAT-P-11 b" --site MaunaKea --band H --days 365
    python cli.py "GJ 3470 b" --period 3.33665 --ecc 0.017
    python cli.py --refresh-catalogue
"""

import argparse
import sys

import numpy as np

from nlotsim import atmosphere, catalogue, events
from nlotsim.bands import BANDS
from nlotsim.sites import SITES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('planet', nargs='?')
    ap.add_argument('--site', default='NLOT', choices=list(SITES))
    ap.add_argument('--band', default='K', choices=list(BANDS))
    ap.add_argument('--resolution', type=int)
    ap.add_argument('--year', type=int, default=2027)
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--events', type=int, default=5,
                    help='how many event dates to print (0 for none)')
    ap.add_argument('--refresh-catalogue', action='store_true')
    ap.add_argument('--n-lines', type=float, default=1000.0,
                    help='usable lines in the template; enters SNR_planet '
                         'as sqrt(N), so it sets the absolute scale but '
                         'changes no ordering (literature ~56 to ~5000)')
    ap.add_argument('--throughput', type=float,
                    help='flat instrument throughput, overriding the '
                         'site curve')
    ap.add_argument('--no-atmosphere', action='store_true',
                    help='skip the skycalc transmission (the instrument '
                         'curves exclude it, so this overstates the SNR)')
    ap.add_argument('--airmass-transmission', type=float, default=1.2,
                    help='airmass for the transmission curve, '
                         'Beer-Lambert scaled (default 1.2)')
    ap.add_argument('--telluric', choices=['mean', 'median'], default='mean',
                    help='mean counts every photon removed; median is '
                         'closer to what survives telluric masking')
    for k in ['period', 'ecc', 'omega', 't14', 'mag', 'teq', 'rp']:
        ap.add_argument(f'--{k}', type=float,
                        help=f'override {k} from the catalogue')
    a = ap.parse_args(argv)

    if a.refresh_catalogue:
        catalogue.fetch(refresh=True)
        if not a.planet:
            return 0

    if not a.planet:
        ap.error('give a planet name, or --refresh-catalogue')

    df, prov = catalogue.fetch()
    print(f"catalogue: {prov.get('table','?')}, "
          f"retrieved {prov.get('retrieved_utc','unknown')}")
    try:
        row = catalogue.lookup(df, a.planet)
    except (KeyError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    params, assumed = catalogue.to_params(row, a.band)
    over = dict(period='period_d', ecc='e', omega='omega_star_deg',
                t14='t14_hr', mag='mag', teq='teq_k', rp='rp_rearth')
    edited = []
    for flag, key in over.items():
        v = getattr(a, flag)
        if v is not None:
            params[key] = v
            edited.append(key)

    print(f"\n{row.pl_name}  |  {SITES[a.site]['name']}  |  {a.band} band"
          f"  |  {a.days} d from {a.year}")
    if assumed:
        print('\nassumed:')
        for s in assumed:
            print(f'  - {s}')
    if edited:
        print(f"\noverridden: {', '.join(edited)}")

    rocky = params.pop('rocky', False)
    derived = params.pop('epoch_derived', False)
    site_cfg = SITES[a.site]
    eta = a.throughput if a.throughput is not None else site_cfg['throughput']
    if not a.no_atmosphere:
        eta = atmosphere.combine(
            eta, atmosphere.transmission_at_airmass(a.airmass_transmission,
                                                    a.telluric))
    site_cfg = dict(site_cfg, throughput=eta)
    from nlotsim.bands import resolve_throughput
    print(f"  end-to-end throughput ({a.band}) "
          f"{resolve_throughput(eta, a.band)*100:8.2f} %"
          + ('' if a.no_atmosphere else
             f'  (incl. atmosphere at X = {a.airmass_transmission})'))
    res = events.observability(**params, site=site_cfg, band=a.band,
                               resolution=a.resolution,
                               cfg=dict(n_lines=a.n_lines),
                               start=f'{a.year}-01-01T00:00:00',
                               n_days=a.days, rocky=rocky,
                               epoch_derived=derived, edited=bool(edited))

    if res['errors']:
        for e in res['errors']:
            print(f'error: {e}', file=sys.stderr)
        return 1
    for w in res['warnings']:
        print(f'\nwarning: {w}')

    print(f"\n  K_p               {res['kp_kms']:8.1f} km/s")
    print(f"  observable hours  {res['hours_observable']:8.0f}")
    print(f"  dv floor          {res['dv_floor_kms']:8.1f} km/s")
    if 't_exp_s' in res:
        te, binds, lims = events.exposure_time(
            params['mag'], a.band, site_cfg, a.resolution or
            site_cfg['resolution'], res['kp_kms'], params['period_d'],
            dict(n_lines=a.n_lines), detail=True)
        print(f"  exposure / duty   {res['t_exp_s']:8.0f} s  "
              f"/ {res['duty_cycle']:.2f}   (limited by {binds})")
        print('                    limits: ' +
              ',  '.join(f'{k} {v:.0f} s' for k, v in lims.items()))

    for ch in ['transmission', 'emission']:
        d = res[ch]
        print(f"\n  {ch.upper()}: {d['n_events']} opportunities")
        if not d['n_events']:
            continue
        print(f"    median |dv|     {np.median(d['dv_kms']):8.1f} km/s")
        if 'snr_star' in d:
            print(f"    SNR_star        {d['snr_star']:8.0f}")
            print(f"    SNR_planet      {d['snr_planet']:8.2f}")
            print(f"    SNR_planet/yr   {d['snr_planet_year']:8.1f}"
                  "   (optimistic sqrt(N) co-add)")
        if a.events:
            print(f"    first {min(a.events, d['n_events'])} dates (UTC):")
            for iso, dv in zip(events.dates_to_iso(d['dates'][:a.events]),
                               d['dv_kms'][:a.events]):
                print(f"      {iso[:19]}   |dv| = {dv:6.1f} km/s")

    print('\n  SNR_planet depends on an assumed line count and on the site '
          'throughput,\n  which sets the absolute scale and is the least '
          'reliable input.\n  Relative comparisons are far more '
          'trustworthy than absolute values.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
