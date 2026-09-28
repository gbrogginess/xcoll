# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""
Thin-gas-target benchmark of the Xcoll proton beam-gas models against Geant4.

Runs ``beamgas_benchmark`` (see ``beamgas_benchmark.cc``, Geant4 FTFP_BERT)
for protons at 450 GeV/c and 6.8 TeV/c on hydrogen, nitrogen and argon, and
compares with the models of ``xcoll.beamgas.proton_cross_sections``:

1. inelastic and elastic cross sections (from the interaction counts);
2. the distribution of |t| in hadronic elastic scattering;
3. the leading-proton spectrum of inelastic events near x_F = 1, against
   the quasi-elastic plus single-diffractive model, from which the value of
   ``sd_scale`` that reproduces Geant4 (FTF) is inferred;
4. the tail of single-step energy losses to knock-on electrons;
5. the tail of the scattering angle of protons without hadronic interaction
   (single Coulomb scattering off nuclei and electrons).

Usage::

    python run_benchmark.py path/to/beamgas_benchmark outdir [n_events]

The Geant4 environment (``geant4.sh``) must be sourced. The summary is
printed and written to ``outdir/summary.json``, with plots in ``outdir``.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xcoll.beamgas import proton_cross_sections as pcs


CONFIGS = [(Z, p) for p in (450., 6800.) for Z in (1, 7, 18)]
SYMBOL = {1: 'H', 7: 'N', 18: 'Ar'}
P_INELASTIC = 0.05       # interaction probability of the target
N_JOBS_PER_CONFIG = 20
MB_TO_CM2 = 1e-27
DELTAS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.15)


def column_density(Z, p):
    """Column density [atoms/cm^2] giving P_INELASTIC."""
    xs = pcs.ProtonNucleusCrossSections(Z, p*1e9)
    return P_INELASTIC/(xs.inelastic/pcs.MILLIBARN*MB_TO_CM2)


def run_geant4(executable, outdir, n_events):
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for Z, p in CONFIGS:
        n_l = column_density(Z, p)
        for jj in range(N_JOBS_PER_CONFIG):
            out = outdir / f'g4_Z{Z}_p{p:.0f}_{jj}.bin'
            if out.exists():
                continue
            seed = 1000*Z + int(p) + jj
            jobs.append([str(executable), str(Z), str(p), f'{n_l:.8e}',
                         str(n_events//N_JOBS_PER_CONFIG), str(seed),
                         str(out)])

    def run(cmd):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor() as pool:
        list(pool.map(run, jobs))


def load(outdir, Z, p):
    files = sorted(outdir.glob(f'g4_Z{Z}_p{p:.0f}_*.bin'))
    return np.concatenate([np.fromfile(ff).reshape(-1, 9) for ff in files])


def model_survivor_fractions(Z, p, rng, n=400_000):
    """
    Fraction of the inelastic cross section in which the beam proton keeps
    more than (1 - Delta) of its momentum, for sd_scale = 1 (so that the
    value matching Geant4 is an absolute scale), split into the
    quasi-elastic and single-diffractive parts.
    """
    xs = pcs.ProtonNucleusCrossSections(Z, p*1e9)
    zeros = np.zeros(n)
    out = {}
    qe = pcs.ProtonQuasiElasticCalculator(Z, p*1e9, cross_sections=xs)
    sd = pcs.ProtonDiffractionCalculator(Z, p*1e9, sd_scale=1.0,
                                         cross_sections=xs)
    loss_qe = -qe.sample_deflections(zeros, zeros, zeros, rng).delta
    loss_sd = -sd.sample_deflections(zeros, zeros, zeros, rng).delta
    for dd in DELTAS:
        out[dd] = (qe.compute_xsec()*np.mean(loss_qe < dd)/xs.inelastic,
                   sd.compute_xsec()*np.mean(loss_sd < dd)/xs.inelastic)
    return out


def analyse(outdir):
    rng = np.random.default_rng(1)
    summary = {}
    fig_t, axes_t = plt.subplots(2, 3, figsize=(13, 7))
    fig_x, axes_x = plt.subplots(2, 3, figsize=(13, 7))
    for ic, (Z, p) in enumerate(CONFIGS):
        key = f'{SYMBOL[Z]}_{p:.0f}GeV'
        data = load(outdir, Z, p)
        n = len(data)
        n_l = column_density(Z, p)
        xs = pcs.ProtonNucleusCrossSections(Z, p*1e9)
        res = {'n_events': n, 'column_density_cm2': n_l}

        # 1. Cross sections from the first hadronic interaction
        first = data[:, 0]
        n_el, n_in = np.sum(first == 1), np.sum(first == 2)
        p_had = (n_el + n_in)/n
        sigma_had = -np.log1p(-p_had)/n_l/MB_TO_CM2
        for name, count, model in (('inelastic', n_in, xs.inelastic),
                                   ('elastic', n_el, xs.elastic)):
            sigma = sigma_had*count/(n_el + n_in)
            res[f'sigma_{name}_g4_mb'] = sigma
            res[f'sigma_{name}_g4_err_mb'] = sigma/np.sqrt(count)
            res[f'sigma_{name}_model_mb'] = model/pcs.MILLIBARN

        # 2. Elastic |t|
        t_g4 = np.sort(data[first == 1, 1])
        chips = pcs.ChipsElasticDistribution(Z, xs.N, p)
        cdf = np.array([chips.probability(0, tt) for tt in t_g4[::10]])
        emp = np.arange(0, len(t_g4), 10)/len(t_g4)
        res['elastic_t_ks_distance'] = float(np.max(np.abs(cdf - emp)))
        res['elastic_t_ks_1sigma'] = float(0.5/np.sqrt(len(t_g4)))
        res['elastic_mean_t_g4'] = float(np.mean(t_g4))
        tt = chips.sample(200_000, rng)
        res['elastic_mean_t_model'] = float(np.mean(tt))
        ax = axes_t.flat[ic]
        bins = np.linspace(0, np.quantile(t_g4, 0.995), 60)
        ax.hist(t_g4, bins=bins, density=True, histtype='step',
                label='Geant4 FTFP_BERT')
        ax.hist(tt, bins=bins, density=True, histtype='step',
                label='Xcoll (CHIPS port)')
        ax.set_yscale('log')
        ax.set_xlabel('|t| [GeV$^2$]')
        ax.set_title(key)
        ax.legend(fontsize=8)

        # 3. Leading proton of inelastic events
        inel = data[first == 2]
        x_f = inel[:, 2]/p
        fractions = model_survivor_fractions(Z, p, rng)
        res['leading_proton'] = {}
        for dd in DELTAS:
            f_g4 = float(np.mean(x_f > 1 - dd))
            f_qe, f_sd = fractions[dd]
            scale = (f_g4 - f_qe)/f_sd if f_sd > 0 else np.nan
            res['leading_proton'][f'{dd:g}'] = {
                'fraction_g4': f_g4,
                'fraction_g4_err': float(np.sqrt(f_g4*(1 - f_g4)/len(x_f))),
                'fraction_model_qe': f_qe,
                'fraction_model_sd': f_sd,
                'sd_scale_matching_g4': float(scale)}
        ax = axes_x.flat[ic]
        bins = np.geomspace(1e-5, 1, 61)
        ax.hist(1 - x_f[(x_f > 0) & (x_f < 1)], bins=bins,
                weights=np.full(np.sum((x_f > 0) & (x_f < 1)), 1/len(x_f)),
                histtype='step', label='Geant4 FTF, leading p')
        zeros = np.zeros(200_000)
        for calc, label in (
                (pcs.ProtonQuasiElasticCalculator(Z, p*1e9), 'Xcoll QE'),
                (pcs.ProtonDiffractionCalculator(Z, p*1e9),
                 f'Xcoll SD (sd_scale={pcs.DEFAULT_SD_SCALE})')):
            loss = -calc.sample_deflections(zeros, zeros, zeros, rng).delta
            ax.hist(loss, bins=bins,
                    weights=np.full(loss.size, calc.compute_xsec()
                                    / xs.inelastic/loss.size),
                    histtype='step', label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('1 - x_F')
        ax.set_ylabel('fraction of inelastic events')
        ax.set_title(key)
        ax.legend(fontsize=8)

        # 4. Knock-on electrons: single-step energy losses
        none = data[first == 0]
        energy = np.sqrt(p**2 + pcs._MP**2)
        ko = pcs.ProtonKnockOnCalculator(Z, p*1e9, cut=1e-6)
        res['knock_on'] = {}
        for vv in (1e-4, 1e-3, 1e-2):
            mask = (none[:, 7] == 1) & (none[:, 6] > vv*energy)
            n_g4 = int(mask.sum())
            p_model = n_l*MB_TO_CM2*pcs._integrate_log(
                ko.dxsec_dT, vv*energy, ko.t_kin_max)
            res['knock_on'][f'{vv:g}'] = {
                'probability_g4': n_g4/len(none),
                'probability_g4_err': np.sqrt(max(n_g4, 1))/len(none),
                'probability_model': p_model}

        # 5. Scattering angle of the protons without hadronic interaction
        msc = 13.6e-3/p*np.sqrt(n_l*pcs.proton_data.NIST_ATOMIC_MASS[Z]
                                * 1.66054e-24/{1: 63.04, 7: 37.99,
                                               18: 19.55}[Z])
        coul = pcs.ProtonElasticCalculator(Z, p*1e9, nuclear=False,
                                           theta_lim=(1e-8, 1e-2))
        n_sample = 1_000_000
        zeros = np.zeros(n_sample)
        c_sample = coul.sample_deflections(zeros, zeros, zeros, rng)
        k_sample = ko.sample_deflections(zeros, zeros, zeros, rng)
        res['angle'] = {'msc_rms_rad': msc}
        for factor in (10, 30, 100):
            theta0 = factor*msc
            p_g4 = float(np.mean(none[:, 4] > theta0))
            p_nuc = n_l*MB_TO_CM2*coul.compute_xsec()/pcs.MILLIBARN*np.sum(
                c_sample.weight*(c_sample.theta > theta0))/n_sample
            p_ele = n_l*MB_TO_CM2*ko.compute_xsec()/pcs.MILLIBARN*np.sum(
                k_sample.weight*(k_sample.theta > theta0))/n_sample
            res['angle'][f'{theta0:.3g}'] = {
                'probability_g4': p_g4,
                'probability_g4_err': float(np.sqrt(
                    max(p_g4, 1/len(none))/len(none))),
                'probability_model_nucleus': float(p_nuc),
                'probability_model_electrons': float(p_ele)}
        summary[key] = res

    fig_t.tight_layout()
    fig_t.savefig(outdir / 'elastic_t.png', dpi=120)
    fig_x.tight_layout()
    fig_x.savefig(outdir / 'leading_proton.png', dpi=120)
    return summary


def print_summary(summary):
    for key, res in summary.items():
        print(f"\n=== {key} ({res['n_events']} events)")
        for name in ('inelastic', 'elastic'):
            g4 = res[f'sigma_{name}_g4_mb']
            err = res[f'sigma_{name}_g4_err_mb']
            model = res[f'sigma_{name}_model_mb']
            print(f"  sigma_{name:9s}: Geant4 {g4:8.2f} +- {err:5.2f} mb, "
                  f"Xcoll {model:8.2f} mb ({(g4 - model)/err:+.1f} sigma)")
        print(f"  elastic |t|: KS distance {res['elastic_t_ks_distance']:.4f}"
              f" (1 sigma ~ {res['elastic_t_ks_1sigma']:.4f}); <|t|> Geant4 "
              f"{res['elastic_mean_t_g4']:.4f}, Xcoll "
              f"{res['elastic_mean_t_model']:.4f} GeV^2")
        print("  leading proton, fraction of inelastic with x_F > 1-Delta:")
        for dd, vv in res['leading_proton'].items():
            print(f"    Delta={dd:>6s}: Geant4 {vv['fraction_g4']:.4f} +- "
                  f"{vv['fraction_g4_err']:.4f}; Xcoll QE "
                  f"{vv['fraction_model_qe']:.4f} + SD "
                  f"{vv['fraction_model_sd']:.4f} (sd_scale=1) -> sd_scale "
                  f"matching Geant4 {vv['sd_scale_matching_g4']:.2f}")
        print("  knock-on, P(single hIoni step > v E):")
        for vv, rr in res['knock_on'].items():
            print(f"    v={vv:>6s}: Geant4 {rr['probability_g4']:.3e} +- "
                  f"{rr['probability_g4_err']:.1e}, Xcoll "
                  f"{rr['probability_model']:.3e}")
        print(f"  angle, P(theta > theta0), MCS rms "
              f"{res['angle']['msc_rms_rad']:.2e} rad:")
        for th, rr in res['angle'].items():
            if th == 'msc_rms_rad':
                continue
            model = (rr['probability_model_nucleus']
                     + rr['probability_model_electrons'])
            print(f"    theta0={th:>9s}: Geant4 {rr['probability_g4']:.3e} +- "
                  f"{rr['probability_g4_err']:.1e}, Xcoll {model:.3e} "
                  f"(nucleus {rr['probability_model_nucleus']:.2e}, "
                  f"electrons {rr['probability_model_electrons']:.2e})")


if __name__ == '__main__':
    executable = Path(sys.argv[1]).resolve()
    outdir = Path(sys.argv[2]).resolve()
    n_events = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
    run_geant4(executable, outdir, n_events)
    summary = analyse(outdir)
    print_summary(summary)
    with open(outdir / 'summary.json', 'w') as fid:
        json.dump(summary, fid, indent=2, default=float)
