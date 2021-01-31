# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:17:17 2021

@author: Yoshi
"""
#import matplotlib.pyplot as plt
#import numpy as np
import PyGeopack as gp

# Want to use geopack to give me a trace of the magnetic field alone a field
# line, and somehow convert (x, y, z) in GSM (?) to an s-value

# Would it be easier to trace out the field line first and output (x,yz) at
# standard intervals along the line, then retrieve the fields at those points
# using whatever? That seems like the easy part (if each point has MLAT, r)

gp.UpdateParameters(SkipWParameters=True)

L_value = 6          # GSM: (L, 0, 0) would be the equatorial point

# =============================================================================
# T = gp.TraceField(L_value, 0.0, 0.0, 20210131, 21.0, Model='T96',CoordIn='GSM',CoordOut='GSM',
# 		alt=100.0,MaxLen=1000,DSMax=1.0,FlattenSingleTraces=True,Verbose=True)
# =============================================================================

# =============================================================================
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
