# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:42:40 2019

@author: Yoshi

Emperical quantities derived from models, such as the Sheely density models
and the dipole magnetic field magnitude
"""
import numpy as np

def geomagnetic_magnitude(L_shell, lat=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given MLAT, in nanoTesla.
    
    INPUT:
        L_shell : McIlwain L-parameter defining distance of disired field line at equator, in RE
        lat     : Geomagnetic latitude (MLAT) in degrees. Default value 0.
        
    OUPUT:
        B_tot   : Magnetic field magnitude, in T
    '''
    B_surf     = 3.12e-5
    r_loc      = L_shell * np.cos(np.pi * lat / 180.) ** 2
    B_tot      = B_surf / (r_loc ** 3) * np.sqrt(1. + 3.*np.sin(np.pi * lat / 180.) ** 2)
    return B_tot


def sheely_plasmasphere(L, av=True):
    '''
    Returns density in /m3
    '''
    mean = 1390* (3 / L) ** 4.83
    var  = 440 * (3 / L) ** 3.60
    if av == True:
        return mean*1e6
    else:
        return np.array([mean - var, mean + var])*1e6


def sheeley_trough(L, LT=0, av=True):
    '''
    Returns the plasmaspheric trough density at a specific L shell and Local Time (LT).
    
    INPUT:
        L  -- McIlwain L-shell of interest
        LT -- Local Time in hours
        av -- Flag to return average value at location (True) or max/min bounds as list
    '''
    mean = 124*(3/L) ** 4 + 36*(3/L) ** 3.5 * np.cos((LT - (7.7 * (3/L) ** 2 + 12))*np.pi / 12)
    var  = 78 * (3 / L) ** 4.72 + 17 * (3 / L) ** 3.75 * np.cos((LT - 22)*np.pi / 12)
    
    if av == True:
        return mean
    else:
        return [mean - var, mean + var]
    
if __name__ == '__main__':
    L_value = 4.0
    
    print('Values at L = {}'.format(L_value))
    print('Density     = {:.2f} /cm3'.format(sheely_plasmasphere(L_value) / 1e6))
    print('Field       = {:.2f} nT'.format(geomagnetic_magnitude(L_value)*1e9))