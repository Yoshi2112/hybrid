# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.special import wofz as Z
import scipy.optimize as opt

def cyclotron_frequency(qi, Bi, mi):
    return qi * Bi / (mi * c)

def plasma_frequency(qi, ni, mi):
    return np.sqrt(4 * np.pi * (qi ** 2) * ni / mi)

def dispersion_function(w, k):
    return (
              w ** 2 - (k*c) ** 2 \
           + (wp_w ** 2 * (anisotropy
           - (Z((w - wc_w) / (thermal_v*k)) / (thermal_v*k)) * ((anisotropy + 1)*(wc_w - w) - wc_w))).sum()
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
    ne  = 8.48                                  # Electron density in cm^-1

    # Cold populations
    mass_c       = np.array([1. / 1836, 1.,    1.]) * mp
    charge_c     = np.array([-1,        1.,    1.]) * q
    density_c    = np.array([1.,        0.985, 0.015])  * ne


    # Warm Populations
    mass_w       = np.array([1.,    1.]) * mp
    charge_w     = np.array([1.,    1.]) * q
    density_w    = np.array([0.985, 0.015])  * ne

    temp_w       = np.array([2e3, 2e3]) * TeV
    anisotropy   = np.array([1., 1.])


    # Characteristic frequencies for populations
    wc_c         = np.array([cyclotron_frequency(qq, B0, mm) for qq, mm in zip(charge_c, mass_c)])
    wp_c         = np.array([plasma_frequency(qq, nn, mm) for qq, nn, mm in zip(charge_c, density_c, mass_c)])
    wc_w         = np.array([cyclotron_frequency(qq, B0, mm) for qq, mm in zip(charge_w, mass_w)])
    wp_w         = np.array([plasma_frequency(qq, nn, mm) for qq, nn, mm in zip(charge_w, density_w, mass_w)])

    thermal_v    = np.sqrt(kB * temp_w / mass_w)
    n_cold = mass_c.shape[0]
    n_warm = mass_w.shape[0]

    #dispersion_function(1, 1)