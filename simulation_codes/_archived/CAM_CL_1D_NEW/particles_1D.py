# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:23:44 2017

@author: iarey
"""
import numba as nb
import numpy as np

from simulation_parameters_1D  import dx, xmax, xmin, qm_ratios

@nb.njit()
def assign_weighting_TSC(pos, I, W, E_nodes=True):
    '''Triangular-Shaped Cloud (TSC) weighting scheme used to distribute particle
    densities to nodes and interpolate field values to particle positions.
        
    NOTE: The addition of `epsilon' in left_node prevents banker's rounding in
    left_node due to precision limits.
    '''
    Np      = pos.shape[0]
    epsilon = 1e-15
    
    if E_nodes == True:
        grid_offset   = 0.5
    else:
        grid_offset   = 1.0
    
    for ii in np.arange(Np):
        I[ii]  = int(round(pos[ii] / dx + grid_offset + epsilon) - 1.0)
        delta_left     = I[ii] - (pos[ii] + epsilon) / dx - grid_offset
    
        W[0, ii] = 0.5  * np.square(1.5 - abs(delta_left))
        W[1, ii] = 0.75 - np.square(delta_left + 1.)
        W[2, ii] = 1.0  - W[0, ii] - W[1, ii]
    return


@nb.njit()
def velocity_update(pos, vel, Ie, W_elec, Ib, W_mag, idx, Ep, Bp, B, E, v_prime, S, T, qmi, dt):
    '''
    Interpolates the fields to the particle positions using TSC weighting, then
    updates velocities using a Boris particle pusher.
    Based on Birdsall & Langdon (1985), pp. 59-63.
    '''
    assign_weighting_TSC(pos, Ib, W_mag, E_nodes=False)
    
    Ep *= 0.0;  Bp *= 0.0
    for ii in range(vel.shape[1]):
        if idx[ii] >= 0:
            qmi[ii] = 0.5 * dt * qm_ratios[idx[ii]]                           # q/m for ion of species idx[ii]
            for jj in range(3):                                               # Nodes
                for kk in range(3):                                           # Components
                    Ep[kk, ii] += E[Ie[ii] + jj, kk] * W_elec[jj, ii]         # Vector E-field  at particle location
                    Bp[kk, ii] += B[Ib[ii] + jj, kk] * W_mag[ jj, ii]         # Vector b1-field at particle location

    vel[:, :] += qmi[:] * Ep[:, :]                                            # First E-field half-push IS NOW V_MINUS

    T[:, :] = qmi[:] * Bp[:, :]                                               # Vector Boris variable
    S[:, :] = 2.*T[:, :] / (1. + T[0, :] ** 2 + T[1, :] ** 2 + T[2, :] ** 2)  # Vector Boris variable
    
    v_prime[0, :] = vel[0, :] + vel[1, :] * T[2, :] - vel[2, :] * T[1, :]     # Magnetic field rotation
    v_prime[1, :] = vel[1, :] + vel[2, :] * T[0, :] - vel[0, :] * T[2, :]
    v_prime[2, :] = vel[2, :] + vel[0, :] * T[1, :] - vel[1, :] * T[0, :]
            
    vel[0, :] += v_prime[1, :] * S[2, :] - v_prime[2, :] * S[1, :]
    vel[1, :] += v_prime[2, :] * S[0, :] - v_prime[0, :] * S[2, :]
    vel[2, :] += v_prime[0, :] * S[1, :] - v_prime[1, :] * S[0, :]
    
    vel[:, :] += qmi[:] * Ep[:, :]                                            # Second E-field half-push
    return


@nb.njit()
def position_update(pos, vel, Ie, W_elec, dt):
    '''Updates the position of the particles using x = x0 + vt. 
    Also updates particle nearest node and weighting.
    '''
    pos += vel[0] * dt
    for ii in np.arange(pos.shape[0]):
        if pos[ii] < xmin:
            pos[ii] += xmax
        elif pos[ii] > xmax:
            pos[ii] -= xmax
            
    assign_weighting_TSC(pos, Ie, W_elec)
    return


# =============================================================================
#         Jp = J[Ie    , 0:3] * W_elec[0]                 \
#            + J[Ie + 1, 0:3] * W_elec[1]                 \
#            + J[Ie + 2, 0:3] * W_elec[2]                 # Current at particle location
#            
#         Ep -= (charge[idx] / q) * e_resis * Jp          # "Effective" E-field accounting for electron resistance
# =============================================================================