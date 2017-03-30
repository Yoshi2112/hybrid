# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:26:20 2017

@author: c3134027
"""
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import sys
import pdb

np.seterr(all='raise')

#----- CONSTANTS -----#
c     = 3.0e8           # Vacuum speed of light
mu0   = pi*4e-7         # Vacuum permeability
e0    = 8.854e-12       # Vacuum permittivity
e     = 1.602e-19       # Elementary charge
me    = 9.109e-31       # Electron mass
mi    = 3.344e-27       # Ion (D+) mass
qi    =  e              # Ion (D+) charge
qe    = -e              # Electron charge

def calc_index(f, theta, B0, n0):
    '''Calculates the refractive index of a plasma for given frequency and parameters'''
    
    #----- FREQUENCY CALCULATIONS -----#
    wpi2 = (n0 * (qi ** 2)) / (mi * e0)     # Ion plasma frequency SQUARED
    wci = (qi * B0) / mi                     # Ion cyclotron frequency
    
    wpe2 = (n0 * (qe ** 2)) / (me * e0)     # Electron plasma frequency SQUARED
    wce = (qe * B0) / me                     # Electron cyclotron frequency
    
    w   = 2 * pi * f                        # Propagating wave frequency (in radians/sec)
    
    #----- LETTER CALCULATIONS -----#
    R = 1 - (wpi2 / (w * (w + wci))) - (wpe2 / (w * (w + wce)))
    L = 1 - (wpi2 / (w * (w - wci))) - (wpe2 / (w * (w - wce)))
    S = 0.5*(R + L)
    D = 0.5*(R - L)
    P = 1 - (wpi2 / (w ** 2)) - (wpe2 / (w ** 2))
    
    #----- QUADRATIC COEFFICIENT CALCULATIONS -----#
    phi = theta*pi / 180
    
    A = S * (np.sin(phi) ** 2) + P*(np.cos(phi) ** 2)
    B = R*L*(np.sin(phi) ** 2) + P*S*(1 + np.cos(phi) ** 2)
    C = P*R*L
    
    F = np.sqrt((B ** 2) - 4*A*C)
    #F = (((R*L - P*S) ** 2) * (np.sin(phi) ** 4)) + (4 * (P ** 2) * (D ** 2) * (np.cos(phi) ** 2))
    
    soln1 = (B + F) / (2*A)
    soln2 = (B - F) / (2*A)
    
    try:
        n1 = np.sqrt(soln1)
    except FloatingPointError:
        n1 = np.nan
    
    try:
        n2 = np.sqrt(soln2)
    except FloatingPointError:
        n2 = np.nan
    
    n_out = np.array([n1, n2])

    return n_out

if __name__ == '__main__':
    
    #----- PLASMA PARAMETERS -----#
    f_in     = 80e6            # Input wave frequency (Hz)
    theta_in = 0               # Angle from the magnetic field
    B0_in     = 3.504           # Magnetic field magnitude
    n0_in    = 1e19            # Electron/Ion density (assuming quasineutrality)

    #n_plasma = calc_index(f_in, theta_in, B0_in, n0_in)

    N = 10001                 # Number of points in graph
    n0_in    = np.linspace(1e18, 1e20, N)
    n_plasma = np.zeros((2, N))

    for ii in range(N):
        n_plasma[:, ii] = calc_index(f_in, theta_in, B0_in, n0_in[ii])

    v_phase = c / n_plasma
   
    # Plot the things
    plt.ioff()
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.plot(n0_in, n_plasma[1, :])
    
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.draw()
    plt.show()

