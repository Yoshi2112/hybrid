# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 22:32:53 2020

@author: Yoshi
"""
import numba as nb
import numpy as np
import matplotlib.pyplot as plt

'''
Generate flux distribution for one-sided maxwellian (gaussian)
 -- Testing to see if Monte-carlo rejection method works
 
Does distro have to be normalized?
'''
@nb.njit()
def vfx(vx, vth):
    f_vx  = np.exp(- 0.5 * (vx / vth) ** 2)
    f_vx /= vth * np.sqrt(2 * np.pi)
    return vx * f_vx


@nb.njit()
def generate_vx(vth):
    while True:
        y_uni = np.random.uniform(0, 4*vth)
        Py    = vfx(y_uni, vth)
        x_uni = np.random.uniform(0, Px_max)
        if Py >= x_uni:
            return y_uni

    


thermal_vel     = 740692.86842899
x_arr           = np.linspace(0, 4*thermal_vel, 10000)
analytic_distro = vfx(x_arr, thermal_vel)

plt.figure()
plt.plot(x_arr, analytic_distro)
plt.xlim(0, 4*thermal_vel)
plt.ylim(0, None)

N_samples = 10000
Px_max    = 0.25 

# Basic (slow and inefficient)
if False:
    N_kept = []
    for ii in range(N_samples):
        y_uni = np.random.uniform(0, 4*thermal_vel)
        Py    = vfx(y_uni)
        x_uni = np.random.uniform(0, Px_max)
        if Py >= x_uni:
            N_kept.append(y_uni)
    N_kept = np.asarray(N_kept)        
    
    plt.figure()
    plt.hist(N_kept)
    
# Vectorised and generates exactly desired number of sample points
if True:
    sampled_points = np.zeros(N_samples)
    for ii in range(N_samples):
        sampled_points[ii] = generate_vx(thermal_vel)
        
    plt.figure()
    plt.hist(sampled_points, bins=20)
    plt.xlim(0, 4*thermal_vel)
    plt.ylim(0, None)
