// copyright ############################### //
// This file is part of the Xcoll package.   //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

/*
 *  thermal_compton.h - Compton scattering of the stored beam on the thermal
 *                      (blackbody) photons of the vacuum chamber (C99,
 *                      header-only).
 *
 *  Overview
 *  --------
 *  Monte Carlo event generator for Compton scattering of ultra-relativistic
 *  leptons on the blackbody photon gas radiated by the (warm) vacuum chamber.
 *  Trial scattering angles are drawn from the Thomson angular distribution and
 *  accepted with the Klein-Nishina / Thomson ratio, so that accepted events
 *  follow the Klein-Nishina distribution while the absolute normalisation stays
 *  fixed by the (constant) Thomson cross section.  This is the proposal method
 *  of H. Burkhardt [1].
 *
 *  Provenance
 *  ----------
 *  The physics model and the absolute rate normalisation follow:
 *    - H. Burkhardt, "Monte Carlo simulation of scattering of beam particles
 *      and thermal photons", CERN SL/Note 93-73 (OP) (1993),
 *      https://cds.cern.ch/record/703373
 *    - A. Di Domenico, "Inverse Compton Scattering of Thermal Radiation at LEP
 *      and LEP-200", Particle Accelerators 39 (1992) 137.
 *    - V. Telnov, "Scattering of electrons on thermal radiation photons in
 *      electron-positron storage rings", NIM A 260 (1987) 304.
 *    - A. Natochii, "Thermal Compton Scattering of Electron Beams on Blackbody
 *      Photons: A Monte Carlo Event Generator for Multi-Turn Tracking at the
 *      Electron-Ion Collider", BNL-229489-2026-TECH, EIC-ADD-TN-159 (2026),
 *      and the accompanying public code https://github.com/eic/thermal-compton-mc
 *      (MIT licence), which is the direct inspiration for this module.
 *
 *  Summary of modifications with respect to the reference implementation
 *  --------------------------------------------------------------------
 *  1. The incoming photon direction is sampled from the *flux-weighted*
 *     distribution p(mu) = (1 - beta*mu)/2 instead of the isotropic one
 *     (mu = cosine of the angle between the photon and the lepton momentum in
 *     the laboratory frame).  The Moeller flux factor (1 - beta*mu) is part of
 *     the interaction rate; averaged over an isotropic gas it integrates to
 *     unity, so the *total* rate is unchanged, but the *shape* of the event
 *     sample is not: head-on collisions (mu -> -1, flux factor 1+beta ~ 2) are
 *     the ones producing the large momentum deviations that lead to losses.
 *     With isotropic sampling the mean energy transfer comes out as
 *     gamma^2 <k> instead of the correct (4/3) gamma^2 <k>, and the loss-
 *     relevant tail is underestimated by 25-45% (increasing with the threshold).
 *  2. The Thomson polar angle is drawn with the exact inverse CDF of
 *     (1 + cos^2 theta) obtained from the closed-form root of the cubic
 *     c^3 + 3c + 4 - 8u = 0, which consumes a single random number.
 *  3. The blackbody photon energy is drawn from the exact Gamma mixture
 *     without any truncation of the series (the mixture index is drawn by
 *     walking the 1/n^3 series, ~1.2 iterations on average) and the Gamma(3)
 *     variate is built from three exponentials.
 *  4. Events are selected against the *local momentum acceptance* (LMA) of the
 *     lattice: a scattered lepton is retained only if its resulting delta falls
 *     outside [delta_neg, delta_pos].
 *  5. An *exact* early veto rejects, before any angular sampling or boost, the
 *     trials that cannot possibly push the lepton out of the acceptance window
 *     (the maximum reachable laboratory momentum transfer is bounded by
 *     k + gamma*(1+beta)*k_star).  No event is lost by this veto.
 *  6. The kernel is written as an xobjects per-particle kernel, i.e. the
 *     macro-particles of the local beam distribution are processed in parallel
 *     (OpenMP / GPU) with independent RNG streams.
 *  7. Xsuite coordinates are used throughout (px = Px/p0c, py = Py/p0c,
 *     delta = (p - p0)/p0), not the slopes x' = Px/Pz used in the reference.
 *
 *  Units
 *  -----
 *  Energies and momenta are in eV (times c), lengths in m, rates in 1/s.
 */

#ifndef XCOLL_THERMAL_COMPTON_H
#define XCOLL_THERMAL_COMPTON_H

#include "xtrack/headers/track.h"
#include "xtrack/random/random_src/uniform_accurate.h"

#include <math.h>
#include <stdint.h>

#ifndef PI
    #define PI 3.14159265358979323846
#endif
#ifndef POW2
    #define POW2(X) ((X)*(X))
#endif

/* Boltzmann constant [eV/K] (CODATA 2018, exact by definition of k_B and e). */
#define XC_KB_EV 8.617333262145e-5
/* Apery's constant zeta(3), normalisation of the sum_n 1/n^3 mixture. */
#define XC_ZETA3 1.2020569031595943
/* Safety cap of the mixture index (P(n > 4096) < 1e-8, and those photons carry
 * an energy ~ 3 kT / n, i.e. they are irrelevant for the loss tail anyway). */
#define XC_NMAX_MIXTURE 4096


/*gpufun*/
double xc_uniform_positive(LocalParticle* part){
    /* Uniform in (0, 1), strictly positive, safe for log(). */
    double u = RandomUniformAccurate_generate(part);
    if (u <= 0.) u = 1e-300;
    return u;
}


/*gpufun*/
double xc_blackbody_photon_energy(LocalParticle* part, double kT){
    /*
     * Sample a photon energy from the blackbody *photon-number* spectrum
     *
     *      p(E) dE  ~  E^2 / (exp(E/kT) - 1) dE .
     *
     * Using 1/(exp(x)-1) = sum_{n>=1} exp(-n x) the distribution becomes an
     * exact mixture of Gamma(shape=3, scale=1/n) laws with weights ~ 1/n^3:
     *   - draw n with P(n) = (1/n^3)/zeta(3) by walking the series,
     *   - draw x ~ Gamma(3, 1/n) as the sum of three exponentials,
     *   - return E = x*kT.
     * The mean energy is 3*zeta(4)/zeta(3)*kT = 2.7011*kT.
     */
    double u = RandomUniformAccurate_generate(part)*XC_ZETA3;
    double acc = 1.;              /* n = 1 term */
    double n = 1.;
    while (u > acc && n < XC_NMAX_MIXTURE){
        n += 1.;
        acc += 1./(n*n*n);
    }
    double const x = -(log(xc_uniform_positive(part))
                     + log(xc_uniform_positive(part))
                     + log(xc_uniform_positive(part)))/n;
    return x*kT;
}


/*gpufun*/
double xc_sample_flux_weighted_mu(LocalParticle* part, double beta){
    /*
     * Cosine of the angle between the incoming photon and the lepton momentum,
     * drawn from the flux-weighted (Moeller) distribution
     *
     *      p(mu) = (1 - beta*mu)/2 ,   mu in [-1, 1] ,
     *
     * i.e. the relative-velocity factor of the collision rate
     * dR = n_gamma sigma c (1 - beta*mu) dOmega/(4 pi).  Its normalisation is
     * unity for an isotropic photon gas, so the total rate is preserved.
     * Exact inverse CDF: mu = [1 - sqrt((1+beta)^2 - 4 beta u)]/beta.
     */
    if (beta < 1e-12) return 2.*RandomUniformAccurate_generate(part) - 1.;
    double const u = RandomUniformAccurate_generate(part);
    double arg = POW2(1. + beta) - 4.*beta*u;
    if (arg < 0.) arg = 0.;
    double mu = (1. - sqrt(arg))/beta;
    if (mu >  1.) mu =  1.;
    if (mu < -1.) mu = -1.;
    return mu;
}


/*gpufun*/
double xc_sample_thomson_costheta(LocalParticle* part){
    /*
     * Polar angle of the Thomson proposal, dsigma_T/dOmega ~ 1 + cos^2(theta).
     * The CDF is F(c) = (c^3 + 3c + 4)/8, so c solves c^3 + 3c - q = 0 with
     * q = 8u - 4.  The (unique) real root is c = t - 1/t with
     * t = cbrt(q/2 + sqrt(q^2/4 + 1)).  Exact, branchless, one random number.
     */
    double const q = 8.*RandomUniformAccurate_generate(part) - 4.;
    double const t = cbrt(0.5*q + sqrt(0.25*q*q + 1.));
    double c = t - 1./t;
    if (c >  1.) c =  1.;
    if (c < -1.) c = -1.;
    return c;
}


/*gpufun*/
void xc_orthonormal_basis(double const* n, double* u, double* v){
    /* Build two unit vectors completing n (unit) into a right-handed triad. */
    double a[3];
    if (fabs(n[0]) < 0.9){ a[0] = 1.; a[1] = 0.; a[2] = 0.; }
    else                 { a[0] = 0.; a[1] = 1.; a[2] = 0.; }
    /* u = n x a, normalised */
    u[0] = n[1]*a[2] - n[2]*a[1];
    u[1] = n[2]*a[0] - n[0]*a[2];
    u[2] = n[0]*a[1] - n[1]*a[0];
    double const nu = sqrt(POW2(u[0]) + POW2(u[1]) + POW2(u[2]));
    u[0] /= nu; u[1] /= nu; u[2] /= nu;
    /* v = n x u */
    v[0] = n[1]*u[2] - n[2]*u[1];
    v[1] = n[2]*u[0] - n[0]*u[2];
    v[2] = n[0]*u[1] - n[1]*u[0];
}


/*gpufun*/
double xc_kn_over_thomson(double a, double costheta, double* x_out){
    /*
     * Ratio of the Klein-Nishina to the Thomson differential cross section in
     * the electron rest frame, for a = k_star/(m c^2):
     *
     *   x  = k_out_star/k_in_star = 1/(1 + a (1 - cos theta))
     *   R  = x^2 (x + 1/x - sin^2 theta)/(1 + cos^2 theta)  in (0, 1].
     *
     * R <= 1 for every angle, so it is a valid acceptance probability, and its
     * average over the Thomson angular distribution is sigma_KN/sigma_T.
     */
    double const x = 1./(1. + a*(1. - costheta));
    double const c2 = costheta*costheta;
    double R = x*x*(x + 1./x - (1. - c2))/(1. + c2);
    if (R > 1.) R = 1.;
    if (R < 0.) R = 0.;
    *x_out = x;
    return R;
}


/*gpufun*/
double xc_sigma_kn_over_sigma_t_small(double a){
    /* Small-a expansion of sigma_KN/sigma_T, error O(a^3).  Only used for the
     * (very soft) trials removed by the exact tail veto, for which
     * a < (acceptance window)/2, i.e. the expansion is accurate to ~1e-7. */
    return 1. - 2.*a + 5.2*a*a;
}


/* The element is passive during tracking: all the physics is in the
 * ThermalComptonScatter kernel below, called explicitly by scatter(). */
/*gpufun*/
void ThermalComptonScattering_track_local_particle(
        ThermalComptonScatteringData el, LocalParticle* part0){
    (void) el;
    (void) part0;
    return;
}


/*
 * Per-particle kernel.  Each "particle" of `part0` is one macro-lepton of the
 * local beam distribution; it is given `n_trials` independent scattering trials
 * representing the section length, and writes the events falling outside the
 * local momentum acceptance (delta < delta_neg or delta > delta_pos) into its
 * own slice of the output arrays [islot*max_events, (islot+1)*max_events).
 */
void ThermalComptonScatter(ThermalComptonScatteringData el,
                           LocalParticle* part0,
                           /*gpuglmem*/ double* x_out,
                           /*gpuglmem*/ double* px_out,
                           /*gpuglmem*/ double* y_out,
                           /*gpuglmem*/ double* py_out,
                           /*gpuglmem*/ double* zeta_out,
                           /*gpuglmem*/ double* delta_out,
                           /*gpuglmem*/ double* photon_energy_out,
                           /*gpuglmem*/ double* weight_out,
                           /*gpuglmem*/ int64_t* n_events_out,
                           /*gpuglmem*/ int64_t* n_dropped_out,
                           /*gpuglmem*/ int64_t* n_outside_out,
                           /*gpuglmem*/ double* rate_scattering_out,
                           /*gpuglmem*/ double* energy_loss_rate_out,
                           double weight_per_trial,
                           int64_t max_events){

    double  const p0c             = ThermalComptonScatteringData_get_p0c(el);
    double  const mass0           = ThermalComptonScatteringData_get_mass0(el);
    double  const temperature     = ThermalComptonScatteringData_get_temperature(el);
    double  const delta_neg       = ThermalComptonScatteringData_get_delta_neg(el);
    double  const delta_pos       = ThermalComptonScatteringData_get_delta_pos(el);
    int64_t const n_trials        = ThermalComptonScatteringData_get_n_trials(el);
    int64_t const tail_veto       = ThermalComptonScatteringData_get_enable_tail_veto(el);

    double const kT = XC_KB_EV*temperature;

    //start_per_particle_block (part0->part)

        int64_t const islot = LocalParticle_get_particle_id(part);
        int64_t const base  = islot*max_events;

        /* ---- incoming lepton in the laboratory frame ---------------------- */
        double const x_in    = LocalParticle_get_x(part);
        double const y_in    = LocalParticle_get_y(part);
        double const zeta_in = LocalParticle_get_zeta(part);
        double const px_in   = LocalParticle_get_px(part);      /* Px/p0c */
        double const py_in   = LocalParticle_get_py(part);      /* Py/p0c */
        double const delta   = LocalParticle_get_delta(part);

        double const p_in  = p0c*(1. + delta);                  /* |p| [eV] */
        double const Px_in = px_in*p0c;
        double const Py_in = py_in*p0c;
        double Pz2 = POW2(p_in) - POW2(Px_in) - POW2(Py_in);
        if (Pz2 < 0.) Pz2 = 0.;
        double const Pz_in = sqrt(Pz2);

        double const E_in  = sqrt(POW2(p_in) + POW2(mass0));
        double const beta  = p_in/E_in;
        double const gamma = E_in/mass0;
        double const ehat[3] = {Px_in/p_in, Py_in/p_in, Pz_in/p_in};

        /* Bound on the laboratory energy transfer, used by the exact veto:
         * the outgoing photon energy cannot exceed gamma (1+beta) k_star, and
         * the momentum change cannot exceed k + that bound. */
        double const boost_max = gamma*(1. + beta);
        /* Momentum transfer needed to push this lepton out of the local
         * momentum acceptance window.  The selection is on the *absolute*
         * outgoing delta, so the distances are measured from the incoming
         * delta and expressed in momentum with p0c (never p_in), which keeps
         * the veto below strictly conservative. */
        double const dp_needed_neg = (delta - delta_neg)*p0c;
        double const dp_needed_pos = (delta_pos - delta)*p0c;
        double const p_required = fmin(dp_needed_neg, dp_needed_pos);

        /* A lepton drawn already outside the acceptance would be selected (and
         * weighted) without any physical scattering.  The longitudinal
         * sampling cutoff is reduced by the Python layer to make this
         * impossible; the case is only flagged here (and its trials skipped)
         * for safety.  No `continue` is used inside the per-particle block. */
        int64_t const outside_window = (p_required <= 0.) ? 1 : 0;
        int64_t const n_trials_eff = outside_window ? 0 : n_trials;

        /* Fixed triad around the lepton direction (constant over the trials). */
        double u1[3], u2[3];
        xc_orthonormal_basis(ehat, u1, u2);

        int64_t n_events  = 0;
        int64_t n_dropped = 0;
        double  rate_all  = 0.;
        double  eloss     = 0.;

        for (int64_t it = 0; it < n_trials_eff; it++){

            /* ---- thermal photon in the laboratory frame ------------------ */
            double const k = xc_blackbody_photon_energy(part, kT);
            double const mu = xc_sample_flux_weighted_mu(part, beta);
            double const smu = sqrt(fmax(0., 1. - mu*mu));
            double const phi_g = 2.*PI*RandomUniformAccurate_generate(part);

            double const cg = cos(phi_g);
            double const sg = sin(phi_g);
            double const nin[3] = {
                mu*ehat[0] + smu*(cg*u1[0] + sg*u2[0]),
                mu*ehat[1] + smu*(cg*u1[1] + sg*u2[1]),
                mu*ehat[2] + smu*(cg*u1[2] + sg*u2[2])};

            /* ---- boost the photon to the lepton rest frame (ERF) --------- */
            double const kstar = gamma*k*(1. - beta*mu);
            double const a = kstar/mass0;

            /* ---- exact veto of the trials that cannot reach the tail ----- */
            if (tail_veto && (k + boost_max*kstar < p_required)){
                /* Such a trial would be accepted by Klein-Nishina with
                 * probability sigma_KN/sigma_T(a) and would never enter the
                 * tail sample: account for its (soft) contribution to the
                 * total scattering rate and move on. */
                double const sig = xc_sigma_kn_over_sigma_t_small(a);
                rate_all += weight_per_trial*sig;
                /* Vetoed trials are deep in the Thomson regime, where the mean
                 * outgoing photon energy is <k_out> = gamma*k_star exactly
                 * (the Thomson angular distribution has <cos theta> = 0), so
                 * the energy-loss diagnostic stays complete. */
                eloss += weight_per_trial*sig*(gamma*kstar - k);
                continue;
            }

            /* ERF direction of the incoming photon (Lorentz transform of the
             * photon momentum with the lepton velocity beta*ehat):
             *   k_star_vec = k*nin + k*[(gamma-1) mu - gamma beta] ehat   */
            double const cf = k*((gamma - 1.)*mu - gamma*beta);
            double kvs[3] = {k*nin[0] + cf*ehat[0],
                             k*nin[1] + cf*ehat[1],
                             k*nin[2] + cf*ehat[2]};
            double const kvs_norm = sqrt(POW2(kvs[0]) + POW2(kvs[1]) + POW2(kvs[2]));
            if (kvs_norm <= 0.) continue;
            double const nstar[3] = {kvs[0]/kvs_norm, kvs[1]/kvs_norm, kvs[2]/kvs_norm};

            /* ---- Thomson proposal + Klein-Nishina acceptance ------------- */
            double const ct = xc_sample_thomson_costheta(part);
            double x_shift;
            double const R = xc_kn_over_thomson(a, ct, &x_shift);
            if (RandomUniformAccurate_generate(part) >= R) continue;

            rate_all += weight_per_trial;

            /* ---- outgoing photon in the ERF ------------------------------ */
            double const st = sqrt(fmax(0., 1. - ct*ct));
            double const phi = 2.*PI*RandomUniformAccurate_generate(part);
            double v1[3], v2[3];
            xc_orthonormal_basis(nstar, v1, v2);
            double const cp = cos(phi);
            double const sp = sin(phi);
            double const kout_star = x_shift*kstar;
            double const kvo[3] = {
                kout_star*(ct*nstar[0] + st*(cp*v1[0] + sp*v2[0])),
                kout_star*(ct*nstar[1] + st*(cp*v1[1] + sp*v2[1])),
                kout_star*(ct*nstar[2] + st*(cp*v1[2] + sp*v2[2]))};

            /* ---- boost the scattered photon back to the laboratory ------- */
            double const edotk = ehat[0]*kvo[0] + ehat[1]*kvo[1] + ehat[2]*kvo[2];
            double const Eg_out = gamma*(kout_star + beta*edotk);
            double const cb = (gamma - 1.)*edotk + gamma*beta*kout_star;
            double const pg_out[3] = {kvo[0] + cb*ehat[0],
                                      kvo[1] + cb*ehat[1],
                                      kvo[2] + cb*ehat[2]};

            /* ---- scattered lepton from 4-momentum conservation ----------- */
            double const Px = Px_in + k*nin[0] - pg_out[0];
            double const Py = Py_in + k*nin[1] - pg_out[1];
            double const Pz = Pz_in + k*nin[2] - pg_out[2];

            eloss += weight_per_trial*(Eg_out - k);

            if (Pz <= 0.){        /* backward lepton: outside the ring frame */
                n_dropped++;
                continue;
            }

            double const p_out = sqrt(POW2(Px) + POW2(Py) + POW2(Pz));
            double const delta_new = p_out/p0c - 1.;

            /* Keep only the leptons that end up outside the local momentum
             * acceptance, i.e. the candidates for a loss. */
            if (delta_new > delta_neg && delta_new < delta_pos) continue;

            if (n_events >= max_events){
                n_dropped++;
                continue;
            }

            int64_t const idx = base + n_events;
            x_out[idx]      = x_in;
            px_out[idx]     = Px/p0c;
            y_out[idx]      = y_in;
            py_out[idx]     = Py/p0c;
            zeta_out[idx]   = zeta_in;
            delta_out[idx]  = delta_new;
            photon_energy_out[idx] = Eg_out;
            weight_out[idx] = weight_per_trial;
            n_events++;
        }

        n_events_out[islot]         = n_events;
        n_dropped_out[islot]        = n_dropped;
        n_outside_out[islot]        = outside_window;
        rate_scattering_out[islot]  = rate_all;
        energy_loss_rate_out[islot] = eloss;

    //end_per_particle_block
}

#endif /* XCOLL_THERMAL_COMPTON_H */