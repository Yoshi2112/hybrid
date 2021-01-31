# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:42:40 2019

@author: Yoshi

Emperical quantities derived from models, such as the Sheely density models
and the dipole magnetic field magnitude
"""
import numpy as np
import matplotlib.pyplot as plt

def geomagnetic_magnitude(L_shell, MLAT=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given MLAT, in nanoTesla.
    
    INPUT:
        L_shell : McIlwain L-parameter defining distance of disired field line at equator, in RE
        MLAT    : Magnetic latitude (MLAT) in degrees. Default value 0.
        
    OUPUT:
        B_tot   : Magnetic field magnitude, in T
    '''
    B_surf     = 3.12e-5
    r_loc      = L_shell * np.cos(np.pi * MLAT / 180.) ** 2
    B_tot      = B_surf / (r_loc ** 3) * np.sqrt(1. + 3.*np.sin(np.pi * MLAT / 180.) ** 2)
    return B_tot


def CLW_geomagnetic_magnitude(L_shell, MLAT=0.):
    '''Returns the magnetic field magnitude (intensity) on the specified L shell at the given MLAT, in nanoTesla.
    
    INPUT:
        L_shell : McIlwain L-parameter defining distance of disired field line at equator, in RE
        MLAT    : Magnetic latitude (MLAT) in degrees. Default value 0.
        
    OUPUT:
        B_tot   : Magnetic field magnitude, in T (originally Gauss?)
    '''
    RE      = 6371e3
    M       = 7.8e22
    colat   = (90. - MLAT)*np.pi/180.
    #pdb.set_trace()
    r_loc   = L_shell * RE * (np.sin(colat) ** 2)
    B_tot_G = - M / (r_loc ** 3) * np.sqrt(3*np.cos(colat) + 1)
    B_tot_T = B_tot_G*1e-4
    return B_tot_T



def sheely_plasmasphere(L, av=True):
    '''
    Returns density in /m3
    '''
    mean = 1390* (3 / L) ** 4.83
    var  = 440 * (3 / L) ** 3.60
    if av == True:
        return mean*1e6
    else:
        return np.array([mean - var, mean + var])*1e6


def sheeley_trough(L, LT=0, av=True):
    '''
    Returns the plasmaspheric trough density at a specific L shell and Local Time (LT).
    
    INPUT:
        L  -- McIlwain L-shell of interest
        LT -- Local Time in hours
        av -- Flag to return average value at location (True) or max/min bounds as list
    '''
    mean = 124*(3/L) ** 4 + 36*(3/L) ** 3.5 * np.cos((LT - (7.7 * (3/L) ** 2 + 12))*np.pi / 12)
    var  = 78 * (3 / L) ** 4.72 + 17 * (3 / L) ** 3.75 * np.cos((LT - 22)*np.pi / 12)
    
    if av == True:
        return mean
    else:
        return [mean - var, mean + var]
    

def plot_dipole_field_line(L, MLAT, length=False):
    '''
    Plots field lines with basic L = r*sin^2(theta) relation. Can plot
    multiple for all in Ls. Can also calculate arclengths from lat_st/lat_min
    and print as the title (can be changed if you want)
    '''
    import matplotlib.pyplot as plt

    colat  = np.pi/2 - MLAT*np.pi/180.
    dtheta = 0.001
    
    lat0   = np.pi/2 - np.arccos(np.sqrt(1.0/L))                              # Latitude for this L value (at ionosphere height)
    theta  = np.arange(lat0, np.pi + dtheta - lat0, dtheta)
    
    plt.ioff()
    plt.figure()
    plt.gcf().gca().add_artist(plt.Circle((0,0), 1.0, color='k', alpha=0.2))
    
    r     = L * np.sin(theta) ** 2
    x     = r * np.cos(theta)
    y     = r * np.sin(theta) 

    xp    = L * np.sin(colat) ** 2 * np.cos(colat)
    yp    = L * np.sin(colat) ** 2 * np.sin(colat) 

    plt.scatter(y,  x,  c='b', s=1,  marker='o')
    plt.scatter(yp, xp, c='k', s=20, marker='o')
        
    plt.axis('equal')
    plt.axhline(0, ls=':', alpha=0.2, color='k')
    plt.axvline(0, ls=':', alpha=0.2, color='k')
    plt.show()
    return


def plot_field_slice(max_L):
    '''
    Need to convert L, MLAT, MLON into a coordinate system that meshes with geopack:
       -- GSM (in RE)
       -- GSW
       
    Nah for now just plot in GSM around xy plane (z = 0) up to max_L in R_E
    '''    
    from geopack import geopack
    ds    = 0.1
    x_gsm = np.arange(-max_L, max_L, ds)
    y_gsm = np.arange(-max_L, max_L, ds)
    z_gsm = 0.0
    r_lim = 3.0
    
    B_out = np.zeros((x_gsm.shape[0], y_gsm.shape[0], 3), dtype=float)
    for ii in range(x_gsm.shape[0]):
        for jj in range(y_gsm.shape[0]):
            r_gsm = np.sqrt(x_gsm[ii] ** 2 + y_gsm[jj] ** 2)
            if r_gsm < r_lim:
                B_out[ii, jj] = np.ones(3)*np.nan
            else:
                B_out[ii, jj] = geopack.dip(x_gsm[ii], y_gsm[jj], z_gsm)
    B0 = np.sqrt(B_out[:, :, 0] ** 2 + B_out[:, :, 1] ** 2 + B_out[:, :, 2] ** 2)
    
    # Normalize
    B_out[:, :, 0] = B_out[:, :, 0] / np.sqrt(B_out[:, :, 0]**2 + B_out[:, :, 1]**2);
    B_out[:, :, 1] = B_out[:, :, 1] / np.sqrt(B_out[:, :, 0]**2 + B_out[:, :, 1]**2);        
    
    plt.ioff()
    fig, ax = plt.subplots()
    im1 = ax.quiver(x_gsm, y_gsm, B_out[:, :, 0].T, B_out[:, :, 1].T, B0)
    ax.set_xlabel('X (RE, GSM)')
    ax.set_ylabel('Y (RE, GSM)')
    fig.colorbar(im1).set_label('|B| (nT)')
    plt.show()
    return x_gsm, y_gsm, B0

    
if __name__ == '__main__':
    B_eq = geomagnetic_magnitude(4.27, MLAT=0.)
    print(B_eq*1e9)
# =============================================================================
#     _L = 5.0; _MLAT = 40.0; _MLON = 0.0
#     
#     _X, _Y, _B = plot_field_slice(_L)
# 
#     #plot_dipole_field_line(_L, _MLAT)
#     
#     if False:
#         _B0  = geomagnetic_magnitude(    _L, MLAT=_MLAT)
#         _B02 = CLW_geomagnetic_magnitude(_L, MLAT=_MLAT)
#         
#         print('Values at L = {}'.format(_L))
#         print('Field       = {:.2f} nT'.format(_B0*1e9))
#         print('Field CLW   = {:.2f} nT'.format(_B02))
# =============================================================================
