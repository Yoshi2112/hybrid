# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:29:37 2019

@author: Yoshi
"""
import numpy as np

def save_n_load(pos, vel, Ie, W_elec, B, Ji, Ve, Te, old_particles, old_fields):
    old_particles[0  , :] = pos
    old_particles[1:4, :] = vel
    old_particles[4  , :] = Ie
    old_particles[5:8, :] = W_elec
    
    old_fields[:, 0:3]    = B
    old_fields[:, 3:6]    = Ji
    old_fields[:, 6:9]    = Ve
    old_fields[:,   9]    = Te
    
    # Zero them all
    pos *= 0
    vel *= 0
    Ie *= 0
    W_elec *= 0
    
    B *= 0
    Ji *= 0
    Ve *= 0
    Te *= 0
    
    pos[:]    = old_particles[0  , :]
    vel[:]    = old_particles[1:4, :]
    Ie[:]     = old_particles[4  , :]
    W_elec[:] = old_particles[5:8, :]
    B[:]      = old_fields[:, 0:3]
    Ji[:]     = old_fields[:, 3:6]
    Ve[:]     = old_fields[:, 6:9]
    Te[:]     = old_fields[:,   9]
    return


if __name__ == '__main__':
    N  = 5
    NX = 7
    
    old_particles_ = np.zeros((8, N))
    old_fields_    = np.zeros((NX + 3, 10))
    
    pos_ = np.ones(N)
    vel_ = np.ones((3, N))
    Ie_  = np.ones(N)
    W_elec_ = np.ones((3, N))
    
    B_ = np.ones((NX + 3, 3))
    Ji_ = np.ones((NX + 3, 3))
    Ve_ = np.ones((NX + 3, 3))
    Te_ = np.ones(NX + 3)

    save_n_load(pos_, vel_, Ie_, W_elec_, B_, Ji_, Ve_, Te_, old_particles_, old_fields_)