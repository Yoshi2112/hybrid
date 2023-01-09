# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:47:00 2022

@author: Yoshi
"""

import numpy as np
import numba as nb
import main_1D as m1d

# Code to test out if its possible to update particle positions/velocity 
# using a cuda version of the algorithms
# It'd be even better if we could do the histogram on the GPU too
# Particle tasks
# -- Update position
# -- Update particle weight
# -- Update velocity
# -- Deposit particles on grid
#
# Biggest issue will be interfacing the field arrays for velocity calculation
# and the source arrays for particle deposition

@nb.njit(parallel=True)
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, B, E, 
                    dt, hot_only=False):    
    '''
    Note: Keeping the code in case it's useful later, but commenting it out
    for speed. Also removed requirement to call Ji, Ve, q_dens (rho) because
    it makes coding the equilibrium bit easier. Also resisitive_array.
    '''
    for ii in nb.prange(pos.shape[0]):
        # Calculate wave fields at particle position
        Ep = np.zeros(3, dtype=np.float64)  
        Bp = np.zeros(3, dtype=np.float64)

        for jj in nb.prange(3):
            for kk in nb.prange(3):
                Ep[kk] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]   
                Bp[kk] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]   

        # Start Boris Method
        qmi = 0.5 * dt * qm_ratios[idx[ii]]                             # q/m variable including dt
        
        # vel -> v_minus
        vel[0, ii] += qmi * Ep[0]
        vel[1, ii] += qmi * Ep[1]
        vel[2, ii] += qmi * Ep[2]
        
        # Calculate background field at particle position (using v_minus)
        # Could probably make this more efficient for a=0
        Bp[0]    += B_eq * (1.0 + a * pos[ii] * pos[ii])
        constant  = a * B_eq
        l_cyc     = qm_ratios[idx[ii]] * Bp[0]
        Bp[1]    += constant * pos[ii] * vel[2, ii] / l_cyc
        Bp[2]    -= constant * pos[ii] * vel[1, ii] / l_cyc
        
        T         = qmi * Bp 
        S         = 2.*T / (1. + T[0]*T[0] + T[1]*T[1] + T[2]*T[2])
            
        # Calculate v_prime
        v_prime    = np.zeros(3, dtype=np.float64)
        v_prime[0] = vel[0, ii] + vel[1, ii] * T[2] - vel[2, ii] * T[1]
        v_prime[1] = vel[1, ii] + vel[2, ii] * T[0] - vel[0, ii] * T[2]
        v_prime[2] = vel[2, ii] + vel[0, ii] * T[1] - vel[1, ii] * T[0]
        
        # vel_minus -> vel_plus
        vel[0, ii] += v_prime[1] * S[2] - v_prime[2] * S[1]
        vel[1, ii] += v_prime[2] * S[0] - v_prime[0] * S[2]
        vel[2, ii] += v_prime[0] * S[1] - v_prime[1] * S[0]
        
        # vel_plus -> vel (updated)
        vel[0, ii] += qmi * Ep[0]
        vel[1, ii] += qmi * Ep[1]
        vel[2, ii] += qmi * Ep[2]
    return


@nb.njit(parallel=True)
def position_update(pos, vel, idx, Ie, W_elec, Ib, W_mag, mp_flux, dt):
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt
        
        # Check if particle has left simulation and apply boundary conditions
        if (pos[ii] < xmin or pos[ii] > xmax):

            if particle_periodic == 1:  
                idx[ii] += Nj                            
            elif particle_open == 1:                
                pos[ii]     = 0.0
                vel[0, ii]  = 0.0
                vel[1, ii]  = 0.0
                vel[2, ii]  = 0.0
                idx[ii]     = Nj
            elif particle_reinit == 1:
                vel[0, ii]  = 0.0
                vel[1, ii]  = 0.0
                vel[2, ii]  = 0.0
                idx[ii]    += Nj                            
            else:
                idx[ii] += Nj 
    return


if __name__ == '__main__':
    particle_periodic = 1
    particle_open     = 0
    particle_reinit   = 0
    
    qi = 1.602e-19
    mi = 1.673e-27
    a = 0.0
    B_eq = 243e-9
    
    Nj = 1
    qm_ratios = np.array([qi/mi])
    xmax = 1e6
    xmin = -1e6
    
    N  = int(1e6)

    m1d.initialize_particles()


