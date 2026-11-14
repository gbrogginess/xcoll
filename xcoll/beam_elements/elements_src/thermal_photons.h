// copyright ############################### #
// This file is part of the Xcoll package.   #
// Copyright (c) CERN, 2026.                 #
// ######################################### #

#ifndef XCOLL_THERMAL_PHOTONS_H
#define XCOLL_THERMAL_PHOTONS_H

#ifdef XO_CONTEXT_CPU
#include <math.h>
#include <stdint.h>  // for int64_t etc
#endif  // XO_CONTEXT_CPU

// A ThermalPhotonMarker element is a passive marker during tracking: the
// scatterings of the beam particles on the thermal photons of the vacuum
// chamber are generated up front by ThermalPhotonGenerator.generate(), which
// produces a weighted macro-particle sample that is then tracked through the
// line. Keeping the element passive (instead of collective) means that
// installing thermal-photon scattering centres in a line costs nothing in
// tracking speed and does not split the line into collective chunks.
//
// The scattering itself is done by ThermalPhotonMarker_scatter, a
// per-particle kernel called by the generator: each particle it is given
// undergoes exactly one Compton scattering on a photon of the blackbody
// radiation at the temperature of the marker, with a relative energy loss
// above the cut of the marker. The physics, the variables and the sampling
// scheme are those of xcoll/thermal_photons/cross_sections.py, where they are
// documented; ThermalComptonCalculator.sample_deflections there is the numpy
// implementation of the same algorithm.

// Boltzmann constant [eV/K] (exact in the SI)
#define XC_THERMAL_PHOTONS_BOLTZMANN_EV_K   8.617333262145179e-5
// Riemann zeta(3)
#define XC_THERMAL_PHOTONS_ZETA_3           1.2020569031595942
#define XC_THERMAL_PHOTONS_TWO_PI           6.283185307179586
// Value of xi_min above which xi is sampled from the truncated exponential
// envelope, instead of discarding the draws of the untruncated distribution
#define XC_THERMAL_PHOTONS_XI_SWITCH        1.0
// The index of the series xi L(xi) = sum_m xi exp(-m xi) / m is not searched
// beyond this value (reached with a probability of 2.5e-8)
#define XC_THERMAL_PHOTONS_MAX_SERIES_INDEX 4096.
// Maximum number of trials of the rejection sampling of one event
#define XC_THERMAL_PHOTONS_MAX_TRIALS       10000000
// Number of values recorded per event
#define XC_THERMAL_PHOTONS_RECORD_SIZE      7


/*gpufun*/
void ThermalPhotonMarker_track_local_particle(ThermalPhotonMarkerData el, LocalParticle* part0){
    (void) el;
    (void) part0;
    return;
}


// L(x) = -ln(1 - exp(-x)) for x >= 0, without cancellation. L is its own
// inverse: L(L(x)) = x.
/*gpufun*/
double thermal_photons_log1mexp(double x){
    if (x > 0.6931471805599453){
        return -log1p(-exp(-x));
    }
    return -log(-expm1(-x));
}


// L(x) exp(x) = -ln(1 - z)/z with z = exp(-x), which tends to one at large x
/*gpufun*/
double thermal_photons_log1mexp_times_exp(double x){
    double const z = exp(-x);
    if (z < 1.e-8){
        return 1. + 0.5*z;
    }
    return -log1p(-z)/z;
}


// Uniform random number in (0, 1], safe as the argument of a logarithm
/*gpufun*/
double thermal_photons_uniform_open(LocalParticle* part){
    return 1. - RandomUniformAccurate_generate(part);
}


// Cosine distributed as 1 + cos^2(theta) on [-1, 1], from a uniform random
// number u: the cumulative distribution 1/2 + (3c + c^3)/8 = u is the cubic
// c^3 + 3c - 2h = 0 with h = 4u - 2, whose real root is c = w - 1/w with
// w^3 = h + sqrt(h^2 + 1) (Cardano).
/*gpufun*/
double thermal_photons_thomson_cos_theta(double u){
    double const h = 4.*u - 2.;
    double const root = sqrt(h*h + 1.);
    // h + root, written without cancellation for negative h
    double const w = cbrt(h >= 0. ? h + root : 1./(root - h));
    double c = w - 1./w;
    if (c > 1.){ c = 1.; }
    if (c < -1.){ c = -1.; }
    return c;
}


// One Compton scattering on a thermal photon for each particle.
//
// record holds XC_THERMAL_PHOTONS_RECORD_SIZE values per particle, at the
// position of its particle_id:
//   0: relative energy loss y = 1 - k*'/k* (Lorentz invariant)
//   1: energy of the scattered photon in the laboratory frame [eV]
//   2: angle of the scattered photon to the incoming particle [rad]
//   3: energy of the thermal photon in the laboratory frame [eV]
//   4: cosine of the angle between the thermal photon and the particle
//   5: cosine of the scattering angle in the rest frame of the particle
//   6: number of trials of the rejection sampling; negative if it did not
//      converge, in which case the particle is left untouched
/*gpufun*/
void ThermalPhotonMarker_scatter(ThermalPhotonMarkerData el, LocalParticle* part0,
                                 /*gpuglmem*/ double* record){

    double const kT  = XC_THERMAL_PHOTONS_BOLTZMANN_EV_K
                     * ThermalPhotonMarkerData_get_temperature(el);
    double const cut = ThermalPhotonMarkerData_get_energy_loss_cut(el);
    // The energy loss is above the cut if eps*(1 - cos_theta) >= a
    double const a   = cut / (1. - cut);

    //start_per_particle_block (part0->part)
        int64_t const slot = XC_THERMAL_PHOTONS_RECORD_SIZE
                           * LocalParticle_get_particle_id(part);

        // Incoming particle
        double const p0c        = LocalParticle_get_p0c(part);
        double const mass_ratio = LocalParticle_get_charge_ratio(part)
                                / LocalParticle_get_chi(part);
        double const mass       = LocalParticle_get_mass0(part) * mass_ratio;
        double const px         = LocalParticle_get_px(part);
        double const py         = LocalParticle_get_py(part);
        double const delta      = LocalParticle_get_delta(part);
        double const pc         = p0c * mass_ratio * (1. + delta);
        double const energy     = sqrt(pc*pc + mass*mass);
        double const gamma      = energy / mass;
        double const beta       = pc / energy;

        // Rest-frame energy, in units of the particle mass, of a head-on
        // photon of laboratory energy kT
        double const eps_T = (1. + beta) * gamma * kT / mass;
        // (1 + beta)/(1 - beta), with 1 - beta = m^2 / (E (E + pc))
        double const gamma2_ratio = (1. + beta) * energy * (energy + pc)
                                  / (mass*mass);
        double const xi_min = a / (2.*eps_T);

        // Scaled rest-frame photon energy xi and rest-frame scattering angle,
        // drawn together by rejection
        double xi = 0.;
        double cos_theta = 0.;
        int64_t n_trials = 0;
        int8_t accepted = 0;
        while (n_trials < XC_THERMAL_PHOTONS_MAX_TRIALS){
            n_trials++;

            if (xi_min <= XC_THERMAL_PHOTONS_XI_SWITCH){
                // xi L(xi) is the mixture sum_m (1/m^3) m^2 xi exp(-m xi):
                // a Gamma(2) variable of rate m, with m drawn with
                // probability 1/(m^3 zeta(3)); discarded below xi_min
                double const target = RandomUniformAccurate_generate(part)
                                    * XC_THERMAL_PHOTONS_ZETA_3;
                double index = 1.;
                double cumulative = 1.;
                while (target > cumulative
                       && index < XC_THERMAL_PHOTONS_MAX_SERIES_INDEX){
                    index += 1.;
                    cumulative += 1./(index*index*index);
                }
                xi = -log(thermal_photons_uniform_open(part)
                          * thermal_photons_uniform_open(part)) / index;
                if (xi < xi_min || xi <= 0.){
                    continue;
                }
            } else {
                // Envelope xi exp(-xi) L(xi_min) exp(xi_min) above xi_min:
                // xi = xi_min + t with t distributed as (xi_min + t) exp(-t)
                double t = -log(thermal_photons_uniform_open(part));
                if (RandomUniformAccurate_generate(part)*(1. + xi_min) >= xi_min){
                    t -= log(thermal_photons_uniform_open(part));
                }
                xi = xi_min + t;
                if (RandomUniformAccurate_generate(part)
                        * thermal_photons_log1mexp_times_exp(xi_min)
                    >= thermal_photons_log1mexp_times_exp(xi)){
                    continue;
                }
            }

            // Exact kinematic range of the laboratory photon energy,
            // xi <= kappa <= gamma2_ratio*xi: weight 1 - L(hi)/L(lo)
            if (RandomUniformAccurate_generate(part)*thermal_photons_log1mexp(xi)
                < thermal_photons_log1mexp(gamma2_ratio*xi)){
                continue;
            }

            // Thomson angle, energy-loss cut and Klein-Nishina weight
            cos_theta = thermal_photons_thomson_cos_theta(
                            RandomUniformAccurate_generate(part));
            double const eta = xi * eps_T * (1. - cos_theta);
            if (eta < a){
                continue;
            }
            double const ratio = 1./(1. + eta);
            double const weight = ratio*ratio
                                * (ratio + 1./ratio - 1. + cos_theta*cos_theta)
                                / (1. + cos_theta*cos_theta);
            if (RandomUniformAccurate_generate(part) >= weight){
                continue;
            }
            accepted = 1;
            break;
        }

        if (!accepted){
            record[slot + 6] = -1.;
        } else {
            // Laboratory photon energy kappa = k/kT, distributed as
            // 1/(exp(kappa) - 1) on [xi, gamma2_ratio*xi], by inversion:
            // L(kappa) = (1 - u) L(xi) + u L(gamma2_ratio*xi)
            double const u = RandomUniformAccurate_generate(part);
            double const kappa = thermal_photons_log1mexp(
                              (1. - u) * thermal_photons_log1mexp(xi)
                              + u * thermal_photons_log1mexp(gamma2_ratio*xi));
            double const phi_k = XC_THERMAL_PHOTONS_TWO_PI
                               * RandomUniformAccurate_generate(part);
            double const phi   = XC_THERMAL_PHOTONS_TWO_PI
                               * RandomUniformAccurate_generate(part);

            // Incident photon in the laboratory frame: q = 1 - beta*cos(psi)
            double const k = kappa * kT;
            double const q = (1. + beta) * xi / kappa;
            double cos_psi = (1. - q) / beta;
            if (cos_psi > 1.){ cos_psi = 1.; }
            if (cos_psi < -1.){ cos_psi = -1.; }
            double const sin_psi = sqrt(1. - cos_psi*cos_psi);

            // Incident photon in the rest frame: energy, aberrated direction
            double const k_rest = gamma * k * q;
            double const eps = k_rest / mass;
            double cos_zeta = (cos_psi - beta) / q;
            if (cos_zeta > 1.){ cos_zeta = 1.; }
            if (cos_zeta < -1.){ cos_zeta = -1.; }
            double const sin_zeta = sin_psi / (gamma * q);

            // Scattering in the rest frame. The scattered direction has the
            // component n_z along the boost axis, n_1 in the plane of the
            // incident photon and the boost axis, and n_2 normal to it.
            double const sin_theta = sqrt(1. - cos_theta*cos_theta);
            double const rho = 1./(1. + eps*(1. - cos_theta));
            double const k_rest_out = rho * k_rest;
            double const n_z = cos_theta*cos_zeta - sin_theta*cos(phi)*sin_zeta;
            double const n_1 = cos_theta*sin_zeta + sin_theta*cos(phi)*cos_zeta;
            double const n_2 = sin_theta*sin(phi);

            // Scattered photon in the laboratory frame
            double const k_out   = gamma * k_rest_out * (1. + beta*n_z);
            double const k_out_z = gamma * k_rest_out * (n_z + beta);
            double const k_out_1 = k_rest_out * n_1;
            double const k_out_2 = k_rest_out * n_2;

            // Momentum transferred to the particle, in the frame of its
            // momentum
            double const dp_1 = k*sin_psi - k_out_1;
            double const dp_2 = -k_out_2;
            double const dp_x = dp_1*cos(phi_k) - dp_2*sin(phi_k);
            double const dp_y = dp_1*sin(phi_k) + dp_2*cos(phi_k);
            double const dp_z = k*cos_psi - k_out_z;

            // Energy conservation, and the change of the momentum modulus
            // without cancellation
            double const energy_out = energy + k - k_out;
            double const pc_out = sqrt(energy_out*energy_out - mass*mass);
            double const dpc = (energy_out - energy) * (energy_out + energy)
                             / (pc_out + pc);

            // Right-handed frame (u, v, p) attached to the particle momentum:
            // u = z x p normalised, v = p x u
            double pz2 = (1. + delta)*(1. + delta) - px*px - py*py;
            if (pz2 < 0.){ pz2 = 0.; }
            double const p_norm = sqrt(px*px + py*py + pz2);
            double const p_x = px / p_norm;
            double const p_y = py / p_norm;
            double const p_z = sqrt(pz2) / p_norm;
            double u_x = -p_y;
            double u_y = p_x;
            double const u_norm = sqrt(u_x*u_x + u_y*u_y);
            if (u_norm < 1.e-12){
                u_x = 1.;
                u_y = 0.;
            } else {
                u_x /= u_norm;
                u_y /= u_norm;
            }
            double const v_x = -p_z*u_y;
            double const v_y = p_z*u_x;

            double const scale = 1./(p0c * mass_ratio);
            LocalParticle_set_px(part, px + (dp_x*u_x + dp_y*v_x + dp_z*p_x)*scale);
            LocalParticle_set_py(part, py + (dp_x*u_y + dp_y*v_y + dp_z*p_y)*scale);
            LocalParticle_update_delta(part, delta + dpc*scale);

            record[slot]     = 1. - rho;
            record[slot + 1] = k_out;
            record[slot + 2] = atan2(sqrt(k_out_1*k_out_1 + k_out_2*k_out_2),
                                     k_out_z);
            record[slot + 3] = k;
            record[slot + 4] = cos_psi;
            record[slot + 5] = cos_theta;
            record[slot + 6] = (double) n_trials;
        }
    //end_per_particle_block
}

#endif /* XCOLL_THERMAL_PHOTONS_H */
