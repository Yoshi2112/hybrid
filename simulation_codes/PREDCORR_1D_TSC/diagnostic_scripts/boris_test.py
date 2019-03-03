# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:01:21 2019

@author: iarey
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

def position_update(pos, vel, DT):
    pos += vel * DT
    return pos


def update_velocity(vel, E, B, DT, charge, mass):
    pdb.set_trace()
    T       = charge/mass*B*0.5*DT
    S       = 2. * T / (1. + T.dot(T))

    v_minus = vel + charge / mass * E * 0.5 * DT
    v_prime = v_minus + np.cross(v_minus, T)
    v_plus  = v_minus + np.cross(v_prime, S)

    vel     = v_plus + charge / mass * E * 0.5 * DT
    return vel


def boris_alg(v0, Bp, Ep, dt, qi, mi):
    v_minus = np.zeros(3)                                               # First velocity
    v_prime = np.zeros(3)                                               # Rotation velocity
    v_plus  = np.zeros(3)                                               # Second velocity

    T = (qi * Bp / mi) * dt / 2.                                        # Boris variable
    S = 2.*T / (1. + T.dot(T))                                          # Boris variable

    # Actual Boris Method
    v_minus    = v0 + qi * Ep * dt / (2. * mi)

    v_prime[0] = v_minus[0] + (v_minus[1] * T[2] - v_minus[2] * T[1])   # Removed multiplicative from second term: (charge[idx] * dt / (2 * mass[idx]))
    v_prime[1] = v_minus[1] - (v_minus[0] * T[2] - v_minus[2] * T[0])
    v_prime[2] = v_minus[2] + (v_minus[0] * T[1] - v_minus[1] * T[0])

    v_plus[0]  = v_minus[0] + (v_prime[1] * S[2] - v_prime[2] * S[1])
    v_plus[1]  = v_minus[1] - (v_prime[0] * S[2] - v_prime[2] * S[0])
    v_plus[2]  = v_minus[2] + (v_prime[0] * S[1] - v_prime[1] * S[0])
 
    v_out = v_plus + qi * Ep * dt / (2. * mi)
    return v_out


def test_particle_orbit():
    v0  = 1e5                          # Particle velocity magnitude 
    B0  = 0.01                         # B-field magnitude
    
    Bc  = np.array([0., 0., B0])       # Magnetic field vector
    Ec  = np.array([0., 0., 0.])       # Electric field vector
    vp  = np.array([0., v0, 0.])       # Initial particle velocity vector
    
# =============================================================================
#     qi = 1.602e-19                     # Proton charge
#     mi = 1.673e-27                     # Proton mass
#     
#     gyperiod   = (2 * np.pi * mi) / (qi * B0) 
#     resolution = 30                       # Steps per gyroperiod (orbit resolution)
#     num_rev    = 50                          # Number of revolutions (periods) to compute
#     maxtime    = resolution*num_rev         # Number of iterations
#     dt         = gyperiod / resolution      # Time increment
# =============================================================================

    mi = 9.109e-31
    qi = -1.602e-19
    dt = 3e-11
    rL = (mi * vp[1]) / (abs(qi) * Bc[2])
    
    xp  = np.array([rL, 0.,  0.])
    #vp  = update_velocity(vp, Ec, Bc, -0.5*dt, qi, mi)
    vp  = boris_alg(vp, Bc, Ec, -0.5*dt, qi, mi)
    
# =============================================================================
#     circle = plt.Circle((0, 0), v0, color='k', fill=False)
#     fig    = plt.figure()
#     ax     = fig.gca()
# =============================================================================
    
    for ii in range(10):
        #plt.scatter(vp[1], vp[2], c='b', s=3)
        #vp  = update_velocity(vp, Ec, Bc, dt, qi, mi)
        vp  = boris_alg(vp, Bc, Ec, dt, qi, mi)
        xp  = position_update(xp, vp, dt)
        
        if ii%2 == 0:
            print ii, xp, vp
        
# =============================================================================
#     plt.scatter(vp[1], vp[2], c='b', s=1)   
#     ax.add_artist(circle)
#     plt.axis('equal')
# =============================================================================
    
    #v_final  = np.sqrt(vp[0] ** 2 + vp[1] ** 2 + vp[2] ** 2)
    
    #print '\nInitial Velocity: {}m/s'.format(round(v0, 2))
    #print 'Final Velocity:   {}m/s'.format(round(v_final, 2))    
    return

if __name__ == '__main__':
    test_particle_orbit()