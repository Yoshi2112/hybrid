# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:05:50 2020

@author: Yoshi
"""
import numpy as np

## Constants ##
c      = 2.998925e+08               # Speed of light (m/s)
mp     = 1.672622e-27               # Mass of proton (kg)
kB     = 1.380649e-23               # Boltzmann's Constant (J/K)
mu0    = (4e-7) * np.pi             # Magnetic Permeability of Free Space (SI units)

def get_B0x(x):
    return B_eq * (1 + a * x ** 2)


def move_particles(pos, vel):
    B0x = get_B0x(pos)
    if False: # Old version
        mu_eq       = 0.5 * mp * (vel[1] ** 2 + vel[2] ** 2) / B_eq
        v_mag2_eq   = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
        v_perp_new  = np.sqrt(2 * mu_eq * B0x / mp)
        
        theta       = np.arctan2(vel[2], vel[1])                 # Velocity gyrophase
        vel[0]      = np.sqrt(v_mag2_eq - v_perp_new ** 2)       # New vx,    preserving velocity/energy
        vel[1]      = v_perp_new * np.cos(theta)                 # New vy, vz preserving gyrophase, invariant
        vel[2]      = v_perp_new * np.sin(theta)
    elif True: # Newer version
        v_perp2     = vel[1] ** 2 + vel[2] ** 2
        v_mag2_eq   = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
        
        v_perp_new  = np.sqrt(v_perp2 * B0x / B_eq)
        v_para_new  = np.sqrt(v_mag2_eq - v_perp_new**2)
    else:
        v_perp_new  = np.sqrt(vel[1] ** 2 + vel[2] ** 2)
    return v_para_new, v_perp_new



if __name__ == '__main__':
    ni     = 200e6                      # Ion density
    
    xmin   = -1000.                     # Lower boundary
    xmax   =  1000.                     # Upper boundary
    B_eq   =  200e-9                    # Local    magnetic field
    B_xmax =  800e-9                    # Boundary magnetic field
    a      =  (B_xmax / B_eq - 1) / xmax ** 2
    
    va        = B_eq / np.sqrt(mu0 * ni * mp)            # Alfven speed (m/s)
    loss_cone = np.arcsin(np.sqrt(B_eq / B_xmax))*180 / np.pi
    
    
    
    
    