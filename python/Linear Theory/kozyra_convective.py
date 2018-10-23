# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import warnings
import numpy             as np
import matplotlib.pyplot as plt
import pdb

def cyclotron_frequency(qi, mi):
    return qi * B0 / (mi * c)

def plasma_frequency(qi, ni, mi):
    return np.sqrt(4. * np.pi * (qi ** 2) * ni / mi)

def geomagnetic_magnitude(L_shell, lat=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given colatitude, in Tesla
    '''
    RE         = 6371000.
    B_surf     = 3.12e-5

    r_loc = L_shell*RE*(np.cos(lat*np.pi / 180.) ** 2)
    B_tot = B_surf * ((RE / r_loc) ** 3) * np.sqrt(1 + 3*(np.sin(lat*np.pi / 180.) ** 2))
    return B_tot

def nu(idx_start, temp):
    '''c: cold, w: warm, b:sum of both'''
    try:
        if temp == 'c':
            out = M[idx_start:] * (wpc[idx_start:] ** 2 / wpw[0] ** 2)
        elif temp == 'w':
            out = M[idx_start:] * (wpw[idx_start:] ** 2 / wpw[0] ** 2)
        elif temp == 'b':
            out = M[idx_start:] * (wpc[idx_start:] ** 2 / wpw[0] ** 2) + \
                  M[idx_start:] * (wpw[idx_start:] ** 2 / wpw[0] ** 2)
    except IndexError:
        out = 0.
    return out

def get_k(X_in):
     return np.sqrt(
                     (wpw[0] ** 2 / c ** 2) *
                         (
                             (((1. + delta) * X_in**2) / (1 - X_in)) +
                             (nu(1, 'b') * ((M[1:] * X_in**2) / (1 - M[1:]*X_in))).sum()
                         )
                   )

def convective_growth_rate(Xi):

    outside_term = ((nu(0, 'w') * np.sqrt(np.pi)) * (M ** 2 * thermal_v_par)) * ( (spec[:, 5] + 1.)*(1. - M*Xi) - 1.)

    exp_num      = ((- nu(0, 'w')) / M) * (((M * Xi - 1.) ** 2) / (beta * Xi ** 2))

    exp_denom    = ((1. + delta) / (1. - Xi)) + ((nu(1, 'b') * M[1:]) / (1. - M[1:]*Xi)).sum()

    last_term    = (2. * Xi ** 2) * ( ((1. + delta) / (1. - Xi)) + (nu(1, 'b') * M[1:] / (1. - M[1:]*Xi) ).sum())

    for ii in range(spec.shape[0]):
        if np.isnan(outside_term[ii]) == True or abs(outside_term[ii]) == np.inf:
            outside_term[ii] = 0.

        if np.isnan(exp_num[ii] ) == True or abs(exp_num[ii]) == np.inf:
            exp_num[ii] = 0.
    pdb.set_trace()
    return ((outside_term * np.exp(exp_num / exp_denom) ).sum()) / last_term

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Constants
    mp  = 1.67e-24                              # g
    q   = 1.602e-20                             # Fr
    c   = 3e10                                  # cm/s
    kB  = 1.381e-16                             # erg/K (Boltzmann's constant)
    TeV = 11603.                                # Conversion factoor: eV -> Kelvin
    N   = 1000                                  # Number of points to solve for

    # Plasma parameters
    B0  = geomagnetic_magnitude(4.0)  * 1e4     # Background magnetic field (in Gauss: 10,000 is conversion factor T > G)

                      # Mass (mp)   Charge (q) n_cold (cc) n_warm(cc)   E_perp (eV) Anisotropy
    spec = np.array([[ 1.,         1.,         10.,        0.,          5e4,         1.         ],    # Hydrogen
                     [ 4.,         1.,         0.,         5.,          5e4,         1.         ]])#,    # Helium (cold)
                     #[ 16.,        1.,         0.,         5.,          5e4,         1.          ]])   # Oxygen (cold)

    # Scale by physical constants
    spec[:, 0] *= mp
    spec[:, 1] *= q

    wpc = plasma_frequency(spec[:, 1], spec[:, 2], spec[:, 0])
    wpw = plasma_frequency(spec[:, 1], spec[:, 3], spec[:, 0])

    wc = cyclotron_frequency(spec[:, 1], spec[:, 0])

    E_par        = (1. / (spec[:, 5] + 1.)) * spec[:, 4]
    T_par        = TeV * E_par

    thermal_v_par    = np.sqrt(kB * T_par / (spec[:, 0]))
    beta             = 8 * np.pi * spec[:, 3] * kB * T_par / (B0 ** 2)

    delta       = wpc[0] ** 2 / wpw[0] ** 2
    M           = q * spec[:, 0] / (spec[:, 1] * mp)                            # Ratio of mass in units of mp to charge number
    X_marginal  = spec[:, 5]  / (M * (1 + spec[:, 5]))                          # Frequency of marginal stability (highest frequency for which there is positive growth rate)

    norm_freq = np.linspace(0, 0.5, N, endpoint=False)
    CGR       = np.zeros(N)
    k         = np.zeros(N)

    for ii in range(N):
        #k[ii]   = get_k(norm_freq[ii])
        CGR[ii] = convective_growth_rate(norm_freq[ii])

    plt.plot(norm_freq, CGR)
    plt.xlim(0, 0.5)

# =============================================================================
#     # Plot stop bands
#     for ii in range(N):
#         if (np.isnan(k[ii]) == True or k[ii] == np.inf) and ii != N - 1:
#             plt.axvspan(norm_freq[ii], norm_freq[ii + 1])
#
#     plt.show()
# =============================================================================


# =============================================================================
#     for cyc in wc:
#         plt.axvline(cyc / wc[0])
# =============================================================================