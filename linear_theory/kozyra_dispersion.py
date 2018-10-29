# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy                     as np
import scipy.optimize            as opt
from   scipy.special import wofz as Z

def cyclotron_frequency(qi, Bi, mi):
    return qi * Bi / (mi * c)

def plasma_frequency(qi, ni, mi):
    return np.sqrt(4 * np.pi * (qi ** 2) * ni / mi)

def dispersion_function(w, k):
    return (
              w ** 2 - (k*c) ** 2 \
           + (wp_w ** 2 * (anisotropy
           - (Z((w - wc_w) / (thermal_v_par*k)) / (thermal_v_par*k)) * ((anisotropy + 1)*(wc_w - w) - wc_w))).sum()
           - ((w * wp_c ** 2) / (w - wc_c)).sum()
           )

if __name__ == '__main__':
    # Constants
    mp  = 1.67e-24                              # g
    q   = 4.803e-10                             # Fr
    c   = 3e10                                  # cm/s
    kB  = 8.617e-5                              # ev/K (Boltzmann's constant)
    TeV = 11603                                 # Conversion factoor: eV -> Kelvin

    # Plasma parameters
    B0  = 4e-9  * 1e5                           # Background magnetic field (in Gauss: 1e5 is conversion factor T > G)

    # Cold populations
    mass_c       = np.array([1.])  * mp         # In g
    charge_c     = np.array([1.])  * q          # In Fr
    density_c    = np.array([10.])              # in cm^-3

    
    # Warm Populations
    mass_w       = np.array([1.]) * mp
    charge_w     = np.array([1.]) * q
    density_w    = np.array([5.])

    energy_w     = np.array([2e3, 2e3])         # Total energy in eV
    anisotropy   = np.array([1., 1.])           # Anisotropy (T_perpendicular / T_parallel)

    ne           = density_c.sum() + density_w.sum()              # Electron density in cm^-1
    mass_c       = np.append(mass_c, 1. / 1836 * mp)    
    charge_c     = np.append(charge_c, -1. * q)    
    density_c    = np.append(density_c, ne)    
    
    # Characteristic frequencies for populations
    wc_c         = np.array([cyclotron_frequency(qq, B0, mm) for qq, mm in zip(charge_c, mass_c)])
    wp_c         = np.array([plasma_frequency(qq, nn, mm) for qq, nn, mm in zip(charge_c, density_c, mass_c)])
    wc_w         = np.array([cyclotron_frequency(qq, B0, mm) for qq, mm in zip(charge_w, mass_w)])
    wp_w         = np.array([plasma_frequency(qq, nn, mm) for qq, nn, mm in zip(charge_w, density_w, mass_w)])

    E_par        = (1. / (anisotropy + 2.)) * energy_w
    E_perp       = energy_w - E_par
    
    T_par        = (2. / kB) * E_par
    T_perp       = (2. / kB) * E_perp
    

    thermal_v_par    = np.sqrt(kB * T_par / mass_w)        # Thermal velocity (parallel) as defined in Gary (1993) with Boltzmann's constant include as per Koyzra (1984)
    n_cold = mass_c.shape[0]
    n_warm = mass_w.shape[0]


    k_max  = 
    #dispersion_function(1, 1)