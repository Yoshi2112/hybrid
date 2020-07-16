# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:18:55 2017

@author: c3134027
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

def set_constants():
    global q, c, mp, me, mu0, kB, e0    
    q   = 1.602e-19                             # Elementary charge (C)
    c   = 3e8                                   # Speed of light (m/s)
    mp  = 1.67e-27                              # Mass of proton (kg)
    me  = 9.11e-31                              # Mass of electron (kg)
    mu0 = (4e-7) * pi                           # Magnetic Permeability of Free Space (SI units)
    kB  = 1.38065e-23                           # Boltzmann's Constant (J/K)
    e0  = 8.854e-12                             # Epsilon naught - permittivity of free space
    return  

def calc_plasma(n):
    return np.sqrt(n * (q ** 2) / (e0 * me))

def calc_e_cyclotron(B):
    return (q * B / me)

def calc_upper_hybrid(plas, wce):
    return np.sqrt(plas ** 2 + wce ** 2)
    
def O_mode(w0, wp):
    return ((w0 ** 2 - wp ** 2) / (c ** 2))
    
def X_mode(w0, wp, wh):
    return (((w0 ** 2) / (c ** 2)) * (1 - ((wp ** 2) / (w0 ** 2)) * ((w0 ** 2 - wp ** 2) / (w0 ** 2 - wh ** 2))))
    
if __name__ == '__main__':
    set_constants()

    f0 = 8e7            # Pump frequency (Hz)
    w0 = 2 * pi * f0    # Pump frequency (rad/s)
    B0 = 3.504          # Background B
    n0 = 1e19           # Plasma density (electron density)
    
    wp = calc_plasma(n0)            # Plasma       frequency (electron)
    wc = calc_e_cyclotron(B0)       # Cyclotron    frequency (electron)
    wh = calc_upper_hybrid(wp, wc)  # Upper hybrid frequency
    
    ko = O_mode(w0, wp)
    kx = X_mode(w0, wp, wh)
    
    #print 'O mode k is: %e' % ko
    #print 'X mode k is: %e' % kx