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
    #n0 = 10**19
    #----- FREQUENCY CALCULATIONS -----#
    wpi2 = (n0 * (qi ** 2)) / (mi * e0)     # Ion plasma frequency SQUARED
    wci  = (qi * B0) / mi                   # Ion cyclotron frequency
    
    wpe2 = (n0 * (qe ** 2)) / (me * e0)     # Electron plasma frequency SQUARED
    wce  = qe * B0 / me                     # Electron cyclotron frequency
    
    w   = f                        # Propagating wave frequency (in radians/sec)
    
    #----- LETTER CALCULATIONS -----#
    R = 1 - (wpi2 / (w * (w + wci))) - (wpe2 / (w * (w + wce)))
    
    if w == wce:
        return()
    else:
        L = 1 - (wpi2 / (w * (w - wci))) - (wpe2 / (w * (w - wce)))
        
    S = 0.5*(R + L)
    D = 0.5*(R - L)
    P = 1 - (wpi2 / (w ** 2)) - (wpe2 / (w ** 2))
    
    #----- QUADRATIC COEFFICIENT CALCULATIONS -----#
    phi = theta*pi / 180
    
    A = S * (np.sin(phi) ** 2) + P*(np.cos(phi) ** 2)
    B = R*L*(np.sin(phi) ** 2) + P*S*(1 + np.cos(phi) ** 2)
    C = P*R*L

    F2 = (((R*L - P*S) ** 2) * ((np.sin(phi)) ** 4)) + (4 * (P ** 2) * (D ** 2) * (np.cos(phi) ** 2))   
    F  = np.sqrt(F2)
     
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
    #pdb.set_trace()
    return n_out

if __name__ == '__main__':
   
    #----- PLASMA PARAMETERS -----#
    f_in     = 80e6            # Input wave frequency (Hz)
    theta_in = 90              # Angle from the magnetic field
    B0_in    = 3.504           # Magnetic field magnitude
    n0_in    = 1e19            # Electron/Ion density (assuming quasineutrality)

    wce  = qe * B0_in / me     # Electron cyclotron frequency
    
    #n_plasma = calc_index(f_in, theta_in, B0_in, n0_in)

    N        = 10000           # Number of points in graph
    x_start  = -2              # Initial domain value, power of 10              
    x_stop   = 1               # Final domain value, power of 10               
    f_in     = (np.logspace(x_start, x_stop, N, base=10)) * wce
    n_plasma = np.zeros((2, N))

    for ii in range(N):
        n_plasma[:, ii] = calc_index(f_in[ii], theta_in, B0_in, n0_in)

    v_phase = c / n_plasma
   
# ----- FIRST PLOT: REFRACTIVE INDEX (SQUARED) ----- #
    plt.ioff()
    fig = plt.figure(1)
    ax  = fig.add_subplot(111)

    x_mode, = ax.plot(f_in, (n_plasma[0,:] ** 2), c='r', label='X mode')
    o_mode, = ax.plot(f_in, (n_plasma[1,:] ** 2), c='b', label='O mode')
    
    ax.set_xlabel(r'Number density, $n_0$ ($m^{-3}$)')
    ax.set_ylabel(r'Refractive index, $n^2$', rotation=90)
    ax.set_title(r'Dispersion Relation for $n_0$ = %e $m^{-3}$, B = %.3fT, $\theta = %d^{\circ}$' % (n0_in, B0_in, theta_in))

    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    #plt.legend()

    #ax.set_xlim(10**x_start, 10**x_stop)
    #ax.set_ylim(0, 5)

# ----- SECOND PLOT: PHASE VELOCITY ----- #
    fig2 = plt.figure(2)
    ax2  = fig2.add_subplot(111)

    x_mode2 = ax2.plot(f_in, v_phase[0,:], c='r', label='X mode')
    o_mode2 = ax2.plot(f_in, v_phase[1,:], c='b', label='O mode')

    ax2.set_xlabel(r'Number density, $n_0$ ($m^{-3}$)')
    ax2.set_ylabel(r'Phase velocity, $v_{ph}$ ($m/s$)') 
    ax2.set_title(r'Phase velocity plot for $n_0$ = %e $m^{-3}$, B = %.3fT, $\theta = %d^{\circ}$' % (n0_in, B0_in, theta_in))
    
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    #plt.legend()
    plt.draw()
    plt.show()
