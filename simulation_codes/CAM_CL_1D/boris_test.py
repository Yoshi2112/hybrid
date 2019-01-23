# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:01:21 2019

@author: iarey
"""

import numpy as np
import matplotlib.pyplot as plt

def boris_alg(v0, Bp, Ep, dt, qi, mi):
    '''Updates the velocity of a particle using the Boris method as detailed
    in Birdsall & Langdon (1985),  pp. 59-63.

    INPUT:
        v0   -- Original particle velocity
        B    -- Magnetic field value at particle  position
        E    -- Electric field value at particle position
        dt   -- Simulation time cadence
        
    OUTPUT:
        v0   -- Updated particle velocity (overwrites input value on return)
    '''
    
    v_minus = np.zeros(3)                                               # First velocity
    v_prime = np.zeros(3)                                               # Rotation velocity
    v_plus  = np.zeros(3)                                               # Second velocity

    T = (qi * Bp / mi) * dt / 2.                        # Boris variable
    S = 2.*T / (1. + np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2] ** 2))        # Boris variable

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
    v0  = 100.                         # Particle velocity magnitude 
    B0  = 4e-9                         # B-field magnitude
    
    Bc  = np.array([B0, 0., 0.])       # Magnetic field vector
    Ec  = np.array([0., 0., 0.])       # Electric field vector
    vp  = np.array([0., v0, 0.])       # Initial particle velocity vector
    
    qi = 1.602e-19                     # Proton charge
    mi = 1.673e-27                     # Proton mass
    gyperiod   = (2 * np.pi * mi) / (qi * B0) 
    
    resolution = 30                       # Steps per gyroperiod (orbit resolution)
    num_rev    = 50                          # Number of revolutions (periods) to compute
    maxtime    = resolution*num_rev         # Number of iterations
    dt         = gyperiod / resolution      # Time increment

    print 'Total number of points: {}'.format(maxtime)    
    print '\nGyroperiod: {}s'.format(round(gyperiod, 2))
    print 'Timestep: {}s'.format(round(dt, 3))
    
    circle = plt.Circle((0, 0), v0, color='k', fill=False)
    fig    = plt.figure()
    ax     = fig.gca()
    
    for ii in range(maxtime+1):
        plt.scatter(vp[1], vp[2], c='b', s=3)
        vp = boris_alg(vp, Bc, Ec, dt, qi, mi)
        
    plt.scatter(vp[1], vp[2], c='b', s=1)   
    ax.add_artist(circle)
    plt.axis('equal')
    
    v_final  = np.sqrt(vp[0] ** 2 + vp[1] ** 2 + vp[2] ** 2)
    
    print '\nInitial Velocity: {}m/s'.format(round(v0, 2))
    print 'Final Velocity:   {}m/s'.format(round(v_final, 2))    
    return

if __name__ == '__main__':
    test_particle_orbit()