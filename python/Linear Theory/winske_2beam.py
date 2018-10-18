# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#from scipy.special import wofz as faddeeva_function
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pdb

# =============================================================================
# def cyclotron_frequency(qi, Bi, mi, radians=True):
#     if radians == True:
#         return (qi * Bi / mi)
#     else:
#         return (qi * Bi / (2 * np.pi * mi))
#
# def plasma_frequency(qi, ni, mi, radians=True):
#     if radians == True:
#         return np.sqrt(ni * (qi ** 2) / (mi * e0))
#     else:
#         return np.sqrt(ni * (qi ** 2) / (2 * np.pi * mi * e0))
# =============================================================================

def wc(qi, Bi, mi):
    return qi * Bi / (mi * c)

def wp(qi, ni, mi):
    return np.sqrt(4 * np.pi * (qi ** 2) * ni / mi)

def dispersion_function(w, k):
    return (w ** 2 - (k*c) ** 2
        - pl[0] ** 2 * (w - k*velocity[0]) / (w - k*velocity[0] + cy[0])
        - pl[1] ** 2 * (w - k*velocity[1]) / (w - k*velocity[1] + cy[1])
        - pl[2] ** 2 * (w - k*velocity[2]) / (w - k*velocity[2] + cy[2]))

if __name__ == '__main__':
    mp  = 1.67e-24                              # g
    q   = 4.803e-10                             # Fr
    c   = 3e10                                  # cm/s

    B0  = (4e-9) * 10000                        # Gauss
    ne  = 8.48                                  # /cc
    va  = B0 / np.sqrt(4 * np.pi * ne * mp)

    mass     = np.array([1. / 1836, 1., 1.])       * mp
    charge   = np.array([-1,        1., 1.])       * q
    density  = np.array([1.,        0.985, 0.015]) * ne
    velocity = np.array([0,        -0.15, 10.])    * va

    n_species = len(mass)

    pl = np.zeros(n_species)
    cy = np.zeros(n_species)
    wi = np.sqrt(4 * np.pi * q ** 2 * ne / mp)

    for ii in range(n_species):
        pl[ii] = wp(charge[ii], density[ii], mass[ii])
        cy[ii] = wc(charge[ii], B0, mass[ii])

    kmin =  0.0
    kmax =  0.16
    kin  = np.linspace(kmin, kmax, 1000) * wi / c

    sol = np.zeros(kin.shape[0])

    for ii in range(1, kin.shape[0]):
        sol[ii] = opt.fsolve(dispersion_function, sol[ii - 1], args=(kin[ii],))

    plt.plot(kin * c / wi, sol / cy[1])

# =============================================================================
#     wmin = 0.00
#     wmax = 0.30
#     win  = np.linspace(wmin, wmax, 100) * cy[1]
#
#     sol  = np.zeros((kin.shape[0], win.shape[0]))
#
#     for ii in range(kin.shape[0]):
#         for jj in range(win.shape[0]):
#             sol[ii, jj] = dispersion_function(win[jj], kin[ii])
#
#     plt.pcolormesh(kin * c / wi, win / cy[1], sol.T, cmap='jet')
#     plt.xlim(-0.08, 0.18)
#     plt.ylim(0, 0.3)
# =============================================================================

