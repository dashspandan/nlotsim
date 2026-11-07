# nlotsim

Works out how often you can observe an exoplanet from a given telescope,
when those chances fall, and how strong the signal would be. Covers both
transits and dayside emission, in the J, H and K bands.

Companion tool to the NLOT high-resolution spectroscopy paper.

## Install

A conda environment is optional but keeps things tidy:

```bash
conda create -n nlotsim python=3.12
conda activate nlotsim
```

Then:

```bash
git clone https://github.com/dashspandan/nlotsim.git
cd nlotsim
pip install -r requirements.txt
```

The planet catalogue ships with the repository, so there is nothing to
download. To update it to the latest version from the NASA Exoplanet
Archive:

```bash
python cli.py --refresh-catalogue
```

## Run the web app

```bash
streamlit run app.py
```

This opens in your browser. Type a planet name, check the numbers that
appear, change any of them if you want, and press Calculate.

## Run from the command line

```bash
python cli.py "WASP-76 b"
python cli.py "HAT-P-11 b" --site MaunaKea --band H
python cli.py "GJ 3470 b" --days 365 --n-lines 5000
```

Useful options:

| Option | What it does |
| --- | --- |
| `--site` | NLOT, MaunaKea, Paranal or LaPalma |
| `--band` | J, H or K |
| `--resolution` | resolving power |
| `--days`, `--year` | period to cover |
| `--n-lines` | lines in the template |
| `--throughput` | flat efficiency instead of the measured curve |
| `--no-atmosphere` | leave out the Earth's atmosphere |
| `--period`, `--ecc`, `--omega`, `--t14`, `--mag`, `--teq`, `--rp` | override a catalogue value |

`python cli.py --help` lists them all.

## Use it in your own code

```python
from nlotsim.events import observability

result = observability(
    ra_deg=26.633, dec_deg=2.700, period_d=1.809882,
    t_conj_bjd=2458080.626165, t14_hr=3.694, a_au=0.033,
    rp_rearth=20.8, mp_mearth=284, rs_rsun=1.756,
    teq_k=2228, teff_k=6329, mag=8.24,
    site='NLOT', band='K')

print(result['emission']['n_events'])
print(result['emission']['snr_planet'])
```

## A note on the numbers

The efficiency and the assumed number of spectral lines both scale every
result up or down together. Comparing planets against each other is much
more reliable than any single absolute value. The web app shows both
settings and lets you change them.

## References

- Cohen, Wheaton & Megeath 2003, AJ 126, 1090
- Dash et al. 2024, MNRAS 530, 3100
- Hong, Finnerty & Fitzgerald 2025, arXiv:2505.09781
- Cheverall & Madhusudhan 2024, arXiv:2403.18894
- Householder & Weiss 2023, arXiv:2212.06966
- Meeus 1998, *Astronomical Algorithms*
