# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np
from   simulation_parameters_1D  import xmin, xmax, qm_ratios, B_eq, a, particle_boundary

@nb.njit()
def eval_B0x(x):
    return B_eq * (1. + a * x**2)

@nb.njit()
def eval_B0_particle(pos, Bp):
    rL     = np.sqrt(pos[1]**2 + pos[2]**2)
    
    B0_r   = - a * B_eq * pos[0] * rL
    Bp[0]  = eval_B0x(pos[0])   
    Bp[1] += B0_r * pos[1] / rL
    Bp[2] += B0_r * pos[2] / rL
    return

@nb.njit()
def velocity_update(pos, vel, idx, dt):
    for ii in nb.prange(vel.shape[1]):  
        if idx[ii] >= 0:
            qmi = 0.5 * dt * qm_ratios[idx[ii]]                                 # Charge-to-mass ration for ion of species idx[ii]

            Ep      = np.zeros(3)
            Bp      = np.zeros(3)
            v_minus = vel[:, ii] + qmi * Ep                                  # First E-field half-push
            
            eval_B0_particle(pos[:, ii], Bp)                                    # Add B0 at particle location
            
            T = qmi * Bp                                                        # Vector Boris variable
            S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable
            
            v_prime    = np.zeros(3)
            v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
            v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
            v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                    
            v_plus     = np.zeros(3)
            v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
            v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
            v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
            
            vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push
    return Bp


@nb.njit()
def position_update(pos, vel, idx, dt):
    for ii in nb.prange(pos.shape[1]):
        # Only update particles that haven't been absorbed (positive species index)
        if idx[ii] >= 0:
            pos[0, ii] += vel[0, ii] * dt
            pos[1, ii] += vel[1, ii] * dt
            pos[2, ii] += vel[2, ii] * dt
            
            # Particle boundary conditions (0: Absorb, 1: Reflect, 2: Periodic)
            if (pos[0, ii] < xmin or pos[0, ii] > xmax):
                
                if particle_boundary == 0:              # Absorb
                    vel[:, ii] *= 0          			# Zero particle velocity
                    idx[ii]     = -128 + idx[ii]        # Fold index to negative values (preserves species ID)
                elif particle_boundary == 1:            # Reflect
                    if pos[0, ii] > xmax:
                        pos[0, ii] = 2*xmax - pos[0, ii]
                    elif pos[0, ii] < xmin:
                        pos[0, ii] = 2*xmin - pos[0, ii]
                    vel[:, ii] *= -1.0                  # 'Reflect' velocities as well. 
                elif particle_boundary == 2:            # Mario (Periodic)
                    if pos[0, ii] > xmax:
                        pos[0, ii] = pos[0, ii] - xmax + xmin
                    elif pos[0, ii] < xmin:
                        pos[0, ii] = pos[0, ii] + xmax - xmin    
    return



                    

