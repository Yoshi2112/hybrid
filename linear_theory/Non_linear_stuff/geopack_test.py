# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:17:17 2021

@author: Yoshi
"""
import pdb
import matplotlib.pyplot as plt
import numpy as np
import geopack as gp

# Want to use geopack to give me a trace of the magnetic field alone a field
# line, and somehow convert (x, y, z) in GSM (?) to an s-value

# Would it be easier to trace out the field line first and output (x,yz) at
# standard intervals along the line, then retrieve the fields at those points
# using whatever? That seems like the easy part (if each point has MLAT, r)

def read_QIN_file(filepath):
    '''
    Hard-coded because I'm too lazy to learn JSON
    '''
    # Could potentially be hardcoded with n_lines = 1440 but this is safer
    print('Reading', filepath)
    n_lines = 0
    with open(filepath) as f:
        for line in f:
            if line[0] != '#':
                n_lines += 1
    
    # Initialize arrays
    year, month, day, hour, minute, second, ByIMF, BzIMF, Vsw, den_P, Pdyn,  \
    ByIMF_status, BzIMF_status, Vsw_status, den_P_status, Pdyn_status, \
    Kp, akp3, Dst, = [np.zeros(n_lines) for _ in range(19)]
    
    epoch    = np.zeros((n_lines), dtype=str)
    G        = np.zeros((n_lines, 3))
    G_status = np.zeros((n_lines, 3))
    Bz       = np.zeros((n_lines, 6))
    W        = np.zeros((n_lines, 6))
    W_status = np.zeros((n_lines, 6))
    
    # Pack in dict? Do later
    with open(filepath) as f:
        
        ii = 0
        for line in f:
            if line[0] != '#':
                A = line.split()

                epoch[ii]        = A[0]
                year[ii]         = int(A[1])
                month[ii]        = int(A[2])
                day[ii]          = int(A[3])
                hour[ii]         = int(A[4])
                minute[ii]       = int(A[5])
                second[ii]       = int(A[6])
                ByIMF[ii]        = float(A[7])
                BzIMF[ii]        = float(A[8])
                Vsw[ii]          = float(A[9])
                den_P[ii]        = float(A[10])
                Pdyn[ii]         = float(A[11])
                G[ii, 0] = float(A[12])
                G[ii, 1] = float(A[13])
                G[ii, 2] = float(A[14])
                ByIMF_status[ii] = float(A[15])
                BzIMF_status[ii] = float(A[16])
                Vsw_status[ii]   = float(A[17])
                den_P_status[ii] = float(A[18])
                Pdyn_status[ii]  = float(A[19])
                G_status[ii, 0]     = float(A[20])
                G_status[ii, 1]     = float(A[21])
                G_status[ii, 2]     = float(A[22])
                Kp[ii]           = float(A[23])
                akp3[ii]         = float(A[24])
                Dst[ii]          = float(A[25])
                Bz[ii, 0] = float(A[26]); Bz[ii, 1] = float(A[27]); Bz[ii, 2] = float(A[28])
                Bz[ii, 3] = float(A[29]); Bz[ii, 4] = float(A[30]); Bz[ii, 5] = float(A[31])
                W[ii, 0]  = float(A[32]); W[ii, 1]  = float(A[33]); W[ii, 2]  = float(A[34])
                W[ii, 3]  = float(A[35]); W[ii, 4]  = float(A[36]); W[ii, 5]  = float(A[37])
                W_status[ii, 0] = float(A[38]); W_status[ii, 1] = float(A[39])
                W_status[ii, 2] = float(A[40]); W_status[ii, 3] = float(A[41])
                W_status[ii, 4] = float(A[42]); W_status[ii, 5] = float(A[43])
                
                ii += 1        
    return


if __name__ == '__main__':
    FPATH = 'G://DATA//QIN_DENTON//2020//QinDenton_20200101_1min.txt'
    read_QIN_file(FPATH)

# =============================================================================
# L_value = 6          # GSM: (L, 0, 0) would be the equatorial point
# 
# xf, yf, zf, xn, yn, zn=gp.geopack.trace(L_value, 0.0, 0.0, -1)
# xf, yf, zf, xs, ys, zs=gp.geopack.trace(L_value, 0.0, 0.0, 1)
# 
# # Check radius:
# r = np.sqrt(xf ** 2 + yf ** 2 + zf ** 2)
# 
# earth = plt.Circle((0, 0), 1.0, color='k', fill=False)
# # Plot field
# fig, ax = plt.subplots()
# 
# ax.scatter(xn, zn)
# ax.scatter(xs, zs)
# ax.add_patch(earth)
# ax.axis('equal')
# =============================================================================
