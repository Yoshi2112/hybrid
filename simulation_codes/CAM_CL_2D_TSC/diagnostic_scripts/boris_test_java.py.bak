# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:08:08 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt

def push_particle(pos, vel, DT):
    pos += vel * DT
    return pos

def update_velocity(vel, E, B, DT):
    T       = charge/mass*B*0.5*DT
    S       = 2. * T / (1. + T.dot(T))

    v_minus = vel + charge / mass * E * 0.5 * DT
    v_prime = v_minus + np.cross(v_minus, T)
    v_plus  = v_minus + np.cross(v_prime, S)

    vel     = v_plus + charge / mass * E * 0.5 * DT
    return vel


if __name__ == '__main__':
    ''' Verified against the java program sourced from https://www.particleincell.com/2011/vxb-rotation/
    and natively run in java. Gives exact same values.
    '''    
    
    mass   = 9.109e-31
    charge = -1.602e-19
    dt     = 3e-11
    
    E   = np.array([0., 0., 0.])
    B   = np.array([0., 0., 0.01])
    vp  = np.array([0., 1e5, 0.])
    rL  = (mass * vp[1]) / (abs(charge) * B[2])
    GP  = (2 * np.pi * mass) / (abs(charge) * B[2])
    
    
    xp  = np.array([rL, 0.,  0.])
    vp  = update_velocity(vp, E, B, -0.5*dt)

    print 'Gyroperiod is: {}s'.format(GP)
    print 'Timestep is: {}s'.format(dt)
    print 'Number of iterations per orbit: {}'.format(round(GP/dt, 2))

    #plt.figure()
    for ii in range(10):
        vp = update_velocity(vp, E, B, dt)
        xp = push_particle(xp, vp, dt)

        if ii%2 == 0:
            print ii, xp, vp

# =============================================================================
#         plt.scatter(xp[0], xp[1], s=1, c='b')
#         
#     plt.axis('equal')
#     plt.xlim(-1.2*rL, 1.2*rL)
#     plt.ylim(-1.2*rL, 1.2*rL)
# =============================================================================
