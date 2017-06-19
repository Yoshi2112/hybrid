from timeit import default_timer as timer
import numpy as np
from numpy import pi
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import sys
import pdb

def set_constants():
    global q, c, mp, mu0, kB, e0, RE    
    q   = 1.602e-19                             # Elementary charge (C)
    c   = 3e8                                   # Speed of light (m/s)
    mp  = 1.67e-27                              # Mass of proton (kg)
    mu0 = (4e-7) * pi                           # Magnetic Permeability of Free Space (SI units)
    kB  = 1.38065e-23                           # Boltzmann's Constant (J/K)
    e0  = 8.854e-12                             # Epsilon naught - permittivity of free space
    RE  = 6371000.                              # Earth radius in metres
    return    
    
def set_parameters():
    global seed, t_res, NX, NY, max_sec, cellpart, ie, B0, size, N, k, ne, xmax, ymax, Te0
    t_res    = 0                               # Time resolution of data in seconds; how often data is dumped to file. Every frame captured if '0'.
    NX       = 20                              # Number of cells in x dimension
    NY       = 20                              # Number of cells in y dimension
    max_sec  = 10000                           # Number of (real) seconds to run program for   
    square   = 10                              # Number of species particles per cell (assuming even representation)
    cellpart = 2*(square ** 2)                 # No. particles per cell (# species times # particles/species)
    ie       = 0                               # Adiabatic electrons. 0: off (constant), 1: on.    
    B0       = 4e-9                            # Unform initial B-field magnitude (in T)
    k        = 5                               # Sinusoidal Density Parameter - number of wavelengths in spatial domain (k - 2 waves)
    ne       = 10.0e6                          # Electron density (used to assign portions of ion)
    Te0      = 0                               # Initial isotropic electron temperature in eV. '0': Isothermal with ions

    size     = NX + 2                          # Size of grid arrays
    N        = cellpart*NX*NY                  # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
    np.set_printoptions(threshold='nan')
    seed     = 21
    return    

def initialize_particles():
    np.random.seed(seed)                          # Random seed 
    global Nj, Te0, dx, dy, xmax, ymax, partin, idx_start, idx_end, xmax, cell_N, n_contr, ne, scramble_position, xmin

    f = 0.015       # Relative beam density
    V = 0.0e5       # Relative streaming velocity 
    # Species Characteristics - use column number as species identifier
    #                        H+ (cold)             H+ (hot)               
    partin = np.array([[           1.0  ,             1.0 ],        #(0) Mass   (proton units)
                       [           1.0  ,             1.0 ],        #(1) Charge (charge units)
                       [             0. ,              V  ],        #(2) Bulk Velocity (m/s)
                       [           1-f  ,              f  ],        #(3) Real density as a portion of ne
                       [           0.5  ,            0.5  ],        #(4) Simulated (superparticle) Density (as a portion of 1)
                       [             0  ,              0  ],        #(5) Distribution type         0: Uniform, 1: Sinusoidal (or beam)
                       [             1.0,              1.0],        #(6) Parallel      Temperature (eV) (x)
                       [             1.0,              1.0],        #(7) Perpendicular Temperature (eV) (y, z)
                       [             1  ,              0  ]])       #(8) Hot (0) or Cold (1) species
    
    part_type     = ['$H^{+}$ (cold)',
                     '$H^{+}$ (hot)'] 
   
    # Reconfigure space for Winske & Quest (1986) testing
    wpi           = np.sqrt((ne * (q ** 2)) / (mp * e0))
    dx            = (c/wpi)                                             # Spacial step (in metres)
    dy            = (c/wpi)
    xmax          = NX * dx
    ymax          = NY * dy

    Nj            = int(np.shape(partin)[1])                                # Number of species (number of columns above)    
    N_species     = np.round(N * partin[4, :]).astype(int)                  # Number of sim particles for each species, total    
    n_contr       = (partin[3, :] * ne * xmax * ymax) / N_species           # Real particles per macroparticle        
    
    Te0  = 11603.                                                           # (Initial) Electron temperature (K)
    Tpar = partin[6, :] * 11603                                             # Parallel ion temperature
    Tper = partin[7, :] * 11603                                             # Perpendicular ion temperature
    
    part     = np.zeros((8, N), dtype=float)                                # Create array of zeroes N x 9 for 2d, idx, 3v,2I and 2W
    old_part = np.zeros((8, N), dtype=float)                                # Place to store last particle states while using Predictor-Corrector method
    
    idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]    # Start index values for each species in order
    idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]    # End   index values for each species in order
    
    # Place particles in configuration and velocity space
    for jj in range(Nj):
        part[2, idx_start[jj]: idx_end[jj]] = jj           # Give index identifier to each particle  
        m       = partin[0, jj] * mp                       # Species mass
        vpar    = np.sqrt(    kB * Tpar[jj] / m)           # Species parallel thermal velocity (x)
        vper    = np.sqrt(2 * kB * Tper[jj] / m)           # Species perpendicular thermal velocity (y, z)
        cell_N  = cellpart * partin[4, :]
        
        if partin[5, jj] == 0:
            acc         = 0
            n_particles = int(cell_N[jj])
            sq          = int(np.sqrt(cellpart/Nj))
            
            for ii in range(NX):
                for kk in range(NY):
                    for mm in range(sq):
                        for nn in range(sq):
                            p_idx = mm*sq + nn
                            part[0, idx_start[jj] + acc + p_idx] = ((float(mm) / sq + ii) * dx)
                            part[1, idx_start[jj] + acc + p_idx] = ((float(nn) / sq + kk) * dy)
                    
                    part[3, (idx_start[jj] + acc): (idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB * Tpar[jj]) / (partin[0, jj] * mp)), n_particles) + partin[2, jj]
                    part[4, (idx_start[jj] + acc): (idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB * Tper[jj]) / (partin[0, jj] * mp)), n_particles)
                    part[5, (idx_start[jj] + acc): (idx_start[jj] + acc + n_particles)] = np.random.normal(0, np.sqrt((kB * Tper[jj]) / (partin[0, jj] * mp)), n_particles)
                    
                    acc += n_particles

    part[6, :] = part[0, :] / dx + 0.5 ; part[6, :] = part[6, :].astype(int)    # Initial leftmost node, Ix
    part[7, :] = part[1, :] / dy + 0.5 ; part[7, :] = part[7, :].astype(int)    # Bottom-most node, Iy
    return part, part_type, old_part

  
def set_timestep(part):
    gyfreq    = q*B0/mp                          # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
    gyperiod  = 2*pi / gyfreq                    # Gyroperiod in seconds
    ion_ts    = 0.05 * gyperiod                  # Timestep to resolve gyromotion
    vel_ts    = dx / (2 * np.max(part[3:5, :]))  # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    
    DT        = min(ion_ts, vel_ts)              # Smallest of the two
    framegrab = int(t_res / DT)                  # Number of iterations between dumps
    maxtime   = int(max_sec / DT) + 1            # Total number of iterations to achieve desired final time
    
    if framegrab == 0:
        framegrab = 1

    print 'Proton gyroperiod = %.2fs' % gyperiod
    print 'Timestep: %.4fs, %d iterations total' % (DT, maxtime)
    return DT, maxtime, framegrab

def update_timestep(part, dt):
    flag = 0
    if dx/(2*np.max(part[3:6, :])) < dt:
        dt  /= 2.
        flag = 1
        print 'Timestep halved: DT = %.5fs' % dt
    return (dt, flag)

def initialize_fields():  
    global Bc, theta

    theta   = 0                                       # Angle of B0 to x axis (in xy plane in units of degrees)
    Bc      = np.zeros(3)                             # Constant components of magnetic field based on theta and B0
    Bc[0]   = B0 * np.cos(theta * pi / 180.)          # x-component of background magnetic field (theta in degrees)
    Bc[1]   = B0 * np.sin(theta * pi / 180.)          # y-component of background magnetic field (theta in degrees)  
    Bc[2]   = 0                                       # Assume Bzc = 0, orthogonal to field line direction
    
    B = np.zeros((size, size, 6), dtype=float)    # Magnetic Field Array of shape (size), each element holds 3-vector
        # Where:
        #       B[mm, 0-2] represent the current field and
        #       B[mm, 3-5] store the last state of the magnetic field previous to the Predictor-Corrector scheme
    B[:, :, 0] = Bc[0]      # Set Bx initial
    B[:, :, 1] = Bc[1]      # Set By initial
    B[:, :, 2] = Bc[2]      # Set Bz initial
    
    E = np.zeros((size, size, 9), dtype=float)    # Electric Field Array
        # Where:
        #       E[mm, 0-2] represent the current field and
        #       E[mm, 3-5] store the last state of the electric field previous to the Predictor-Corrector scheme E (N + 0.5)
        #       E[mm, 6-8] store two steps ago: E-field at E^N

    Vi      = np.zeros((size, size, Nj, 3), dtype=float)          # Ion Flow (3 dimensions)
    dns     = np.zeros((size, size, Nj),    dtype=float)          # Species number density in each cell (in /m3)
    dns_old = np.zeros((size, size, Nj),    dtype=float)          # For PC method
    W       = np.zeros((4, N),              dtype=float)          # Particle Weighting (linear: generalize later)
    Wb      = np.zeros((4, N),              dtype=float)
    return B, E, Vi, dns, dns_old, W, Wb    
    

def velocity_update(part, B, E, dt, WE_in, WB_in):  # Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
    
    for n in range(N):
        vn = part[3:6, n]                # Existing particle velocity
        E_p = 0; B_p = 0                 # Initialize field values
        
        # E & B fields at particle location: Node values /w weighting factors in x, y 
        Ix   = int(part[6, n])             # Nearest (leftmost) node
        Iy   = int(part[7, n])             # Nearest (bottom)   node
        Ibx  = int(part[0, n] / dx)        # Nearest (leftmost) magnetic node
        Iby  = int(part[1, n] / dy)        # Nearest (bottom)   magnetic node
        
        WEx  = WE_in[0:2, n]                # E-field weighting (x)
        WEy  = WE_in[2:4, n]                # E-field weighting (y)
        WBx  = WE_in[0:2, n]                # B-field weighting (x)
        WBy  = WB_in[2:4, n]                # B-field weighting (y)
        
        idx  = int(part[2, n])             # Particle species index
       
        # Interpolate fields to particle position
        for ii in range(2):
            for jj in range(2):
                E_p += E[Ix  + ii, Iy  + jj, 0:3] * WEx[ii] * WEy[jj]
                B_p += B[Ibx + ii, Iby + jj, 0:3] * WBx[ii] * WBy[jj]

        # Intermediate calculations
        h  = (partin[1 , idx] * q * dt) / (partin[0 , idx] * mp)
        f  = 1 - (h**2) / 2 * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2 )
        g  = h / 2 * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
        v0 = vn + (h/2)*E_p
    
        # Velocity push
        part[3, n] = f * vn[0] + h * ( E_p[0] + g * B_p[0] + (v0[1]*B_p[2] - v0[2]*B_p[1]) )
        part[4, n] = f * vn[1] + h * ( E_p[1] + g * B_p[1] - (v0[0]*B_p[2] - v0[2]*B_p[0]) )
        part[5, n] = f * vn[2] + h * ( E_p[2] + g * B_p[2] + (v0[0]*B_p[1] - v0[1]*B_p[0]) )
    return part        
    
def position_update(part):  # Basic Push (x, v) vectors and I, W update
    
    part[0:2, :] += part[3:5, :] * DT                   # Add velocity
    
    for ii in range(N):                                 # Boundary conditions (periodic)
        if part[0, ii] < 0:
            part[0, ii] = xmax + part[0,ii]
            
        if part[0, ii] > xmax:
            part[0, ii] = part[0,ii] - xmax

        if part[1, ii] < 0:
            part[1, ii] = ymax + part[0, ii]

        if part[1, ii] > ymax:
            part[1, ii] = part[1, ii] - xmax
    
    part[6, :] = part[0, :] / dx + 0.5 ; part[6, :] = part[6, :].astype(int)            # Ix update
    part[7, :] = part[1, :] / dy + 0.5 ; part[7, :] = part[7, :].astype(int)            # Iy update
    We_out      = assign_weighting(part[0:2, :], part[6:8, :], 1)
    Wb_out      = assign_weighting(part[0:2, :], part[6:8, :], 0)                   # Magnetic field weighting (due to E/B grid displacement) 
    return part, We_out, Wb_out
    
    
def assign_weighting(pos, I, BE):                               # Magnetic/Electric Field, BE == 0/1
    '''Assigns weighting for the nodes adjacent to each
       particle. Outputs the left, right, bottom, and top 
       weighting factors, respectively'''
    W_x = ((pos[0, :])/(dx)) - I[0, :] + (BE / 2.)              # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
    W_y = ((pos[1, :])/(dy)) - I[1, :] + (BE / 2.)              # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
    
    W_out = np.stack(((1 - W_x), W_x, (1 - W_y), W_y), axis=0)  # Left, Right, 'Left' (Bottom), 'Right' (Top)
    return W_out
    
def push_B(B, E, dt):   # Basically Faraday's Law. (B, E) vectors
        
    Bp = np.zeros((size, size, 3), dtype=float)
    
    # Consider B1 only (perturbation)    
    Bp[:, :, 0] = B[:, :, 0] - Bc[0]    
    Bp[:, :, 1] = B[:, :, 1] - Bc[1]
    Bp[:, :, 2] = B[:, :, 2] - Bc[2]   
    
    # Curl of E - 2D only (derivatives of z are zero)
    for mm in range (1, size - 1):
        for nn in range(1, size - 1):
            Bp[mm, nn, 0] = Bp[mm, nn, 0] - (dt / 2.) * (  (E[mm, nn + 1, 2] - E[mm, nn - 1, 2]) / (2 * dy))
            Bp[mm, nn, 1] = Bp[mm, nn, 1] + (dt / 2.) * (  (E[mm + 1, nn, 2] - E[mm - 1, nn, 2]) / (2 * dx))     # Flipped sign due to j
            Bp[mm, nn, 2] = Bp[mm, nn, 2] - (dt / 2.) * ( ((E[mm + 1, nn, 1] - E[mm - 1, nn, 1]) / (2 * dx)) - ((E[mm, nn + 1, 0] - E[mm, nn - 1, 0]) / (2 * dy)) )
    
    Bp = manage_ghost_cells(Bp, 0)

    # Combine total B-field: B0 + B1    
    B[:, :, 0] = Bp[:, :, 0] + Bc[0]    
    B[:, :, 1] = Bp[:, :, 1] + Bc[1]
    B[:, :, 2] = Bp[:, :, 2] + Bc[2]
    return B
    
    
def push_E(B, J_species, n_i, dt):

    E_out = np.zeros((size, size, 3))               # Output array - new electric field
    Ji    = np.zeros((size, size, 3))               # Total ion current
    JixB  = np.zeros((size, size, 3))               # V cross B holder
    BdB   = np.zeros((size, size, 3))               # B cross del cross B holder     
    del_p = np.zeros((size, size, 3))               # Electron pressure tensor gradient array
    qn    = np.zeros((size, size    ), dtype=float) # Ion charge density
    
    Te = np.ones((size, size)) * Te0                             # Isothermal (const) approximation until adiabatic thing happens 
    
    # Calculate average/summations over species
    for jj in range(Nj):
        qn += partin[1, jj] * n_i[:, :, jj] * q                  # Total charge density, sum(qj * nj)
        
        for kk in range(3):
            Ji[:, :, kk]  += J_species[:, :, jj, kk]             # Total ion current vector: J_k = qj * nj * Vjk
    
    # J cross B
    JixB[:, :, 0] +=    Ji[:, :, 1] * B[:, :, 2] - Ji[:, :, 2] * B[:, :, 1]   
    JixB[:, :, 1] += - (Ji[:, :, 0] * B[:, :, 2] - Ji[:, :, 2] * B[:, :, 0])   
    JixB[:, :, 2] +=    Ji[:, :, 0] * B[:, :, 1] - Ji[:, :, 1] * B[:, :, 0]
    
    for mm in range(1, size - 1):
        for nn in range(1, size - 1):
            # B cross curl B
            BdB[mm, nn, 0] =   B[mm, nn, 1] * ((B[mm + 1, nn, 1] - B[mm - 1, nn, 1]) / (2 * dx)) \
                             - B[mm, nn, 1] * ((B[mm, nn + 1, 0] - B[mm, nn - 1, 0]) / (2 * dy)) \
                             + B[mm, nn, 2] * ((B[mm + 1, nn, 2] - B[mm - 1, nn, 2]) / (2 * dx))  

            BdB[mm, nn, 1] = - B[mm, nn, 0] * ((B[mm + 1, nn, 1] - B[mm - 1, nn, 1]) / (2 * dx)) \
                             + B[mm, nn, 0] * ((B[mm, nn + 1, 0] - B[mm, nn - 1, 0]) / (2 * dy)) \
                             + B[mm, nn, 2] * ((B[mm, nn + 1, 2] - B[mm, nn - 1, 2]) / (2 * dy))

            BdB[mm, nn, 2] = - B[mm, nn, 0] * ((B[mm + 1, nn, 2] - B[mm - 1, nn, 2]) / (2 * dx)) \
                             - B[mm, nn, 1] * ((B[mm, nn + 1, 2] - B[mm, nn - 1, 2]) / (2 * dy))
    
            # del P
            del_p[mm, nn, 0] = (kB / (2*dx*q)) * ( Te[mm, nn] * (qn[mm + 1, nn] - qn[mm - 1, nn]) +
                                                   qn[mm, nn] * (Te[mm + 1, nn] - Te[mm - 1, nn]) )
            del_p[mm, nn, 1] = (kB / (2*dy*q)) * ( Te[mm, nn] * (qn[mm, nn + 1] - qn[mm, nn - 1]) + 
                                                   qn[mm, nn] * (Te[mm, nn + 1] - Te[mm, nn - 1]) )
            del_p[mm, nn, 2] = 0
    
    for xx in range(3):
        JixB[ :, :, xx] /= qn[:, :]
        BdB[  :, :, xx] /= qn[:, :]*mu0
        del_p[:, :, xx] /= qn[:, :]
        E_out[:, :, xx]  = - JixB[:, :, xx] - del_p[:, :, xx] - BdB[:, :, xx] 

    E_out = manage_ghost_cells(E_out, 0)

    #X, Y = np.meshgrid(range(size), range(size))
    #fig = plt.figure()

    #ax_Jx = plt.subplot2grid((2, 2), (0, 0), projection='3d')
    #ax_Jx.plot_wireframe(X, Y, Ji[:, :, 0])
    #ax_Jx.view_init(elev=30., azim=300.)
    #ax_Jx.set_title('Jx')

    #ax_Jy = plt.subplot2grid((2, 2), (0, 1), projection='3d')
    #ax_Jy.plot_wireframe(X, Y, Ji[:, :, 1])
    #ax_Jy.view_init(elev=30., azim=300.)
    #ax_Jy.set_title('Jy')
    
    #ax_Jz = plt.subplot2grid((2, 2), (1, 0), projection='3d')
    #ax_Jz.plot_wireframe(X, Y, Ji[:, :, 2])
    #ax_Jz.view_init(elev=30., azim=300.) 
    #ax_Jz.set_title('Jz')
    
    #ax_J = plt.subplot2grid((2, 2), (1, 1), projection='3d')
    #ax_J.plot_wireframe(X, Y, E_out[:, :, 2])
    #ax_J.view_init(elev=30., azim=300.)
    #ax_J.set_title('E_out z')
    #plt.show()

    return E_out


def collect_density(part, W): 
    '''Function to collect charge density in each cell in each cell
    at each timestep. These values are weighted by their distance
    from cell nodes on each side. Can send whole array or individual particles?
    How do I sum up the densities one at a time?'''

    n_i = np.zeros((size, size, Nj), float)
    
    # Collect number density of all particles
    for ii in range(N):
        idx  = int(part[2, ii])     # Species index
        Ix   = int(part[6, ii])     # Left
        Iy   = int(part[7, ii])     # Bottom nodes
        Wx   = W[0:2, ii]           # Left,   right
        Wy   = W[2:4, ii]           # Bottom, top   node weighting factors
        
        for jj in range(2):
            for kk in range(2):
                n_i[Ix + jj, Iy + kk, idx] += Wx[jj] * Wy[kk] * n_contr[idx]
    
    n_i = manage_ghost_cells(n_i, 1) / (dx*dy)         # Divide by cell size for density per unit volume

    for jj in range(Nj):
        n_i[:, :, jj] = smooth(n_i[:, :, jj])
    return n_i


def collect_current(part, ni, W):
    
    J_i = np.zeros((size, size, Nj, 3), float)    
    
    # Loop through all particles: sum velocities for each species: Limited testing due to almost-identical algorithm as used for n_i?
    for ii in range(N):
        idx = int(part[2, ii])
        Ix  = int(part[6, ii])
        Iy  = int(part[7, ii])
        Wx  = W[0:2, ii]
        Wy  = W[2:4, ii]
       
        for jj in range(2):
            for kk in range(2):
                J_i[Ix + jj, Iy + kk, idx, :] += Wx[jj] * Wy[kk] * n_contr[idx] * part[3:6, ii]     # Does all 3 dimensions at once (much more efficient/parallel)
    
    for jj in range(Nj):    # Turn those velocities into currents (per species)
        J_i[:, :, jj, :] *= partin[1, jj] * q

    J_i = manage_ghost_cells(J_i, 1) / (dx*dy)     # Divide by spatial cell size for current per unit

    for jj in range(Nj):
        for kk in range(3):
            J_i[:, :, jj, kk] = smooth(J_i[:, :, jj, kk])
    return J_i
    
def manage_ghost_cells(arr, src):
    '''Deals with ghost cells: Moves their contributions and mirrors their counterparts.
       Works like a charm if spatial dimensions always come first in an array. Condition
       variable passed with array because ghost cell field values do not need to be moved:
       But they do need correct (mirrored) ghost cell values'''
    
    if src == 1:   # Move source term contributions to appropriate edge cells
        arr[1, 1]                   += arr[size - 1, size - 1]    # TR -> BL : Move corner cell contributions
        arr[1, size - 2]            += arr[size - 1, 0]           # BR -> TL
        arr[size - 2, 1]            += arr[0, size - 1]           # TL -> BR
        arr[size - 2, size - 2]     += arr[0, 0]                  # BL -> TR
        
        arr[size - 2, 1: size - 1]  += arr[0, 1: size - 1]        # Move contribution: Bottom to top
        arr[1, 1:size - 1]          += arr[size - 1, 1: size - 1] # Move contribution: Top to bottom
        arr[1: size - 1, size - 2]  += arr[1: size - 1, 0]        # Move contribution: Left to Right
        arr[1: size - 1, 1]         += arr[1: size - 1, size - 1] # Move contribution: Right to Left
   
    arr[0, 0]                   = arr[size - 2, size - 2]         # Fill corner cell: BL
    arr[0, size - 1]            = arr[size - 2, 1]                # Fill corner cell: TL 
    arr[size - 1, 0]            = arr[1, size - 2]                # Fill corner cell: BR 
    arr[size - 1, size - 1]     = arr[1, 1]                       # Fill corner cell: TR

    arr[size - 1, 1: size - 1]  = arr[1, 1: size - 1]             # Fill ghost cell: Top
    arr[0, 1: size - 1]         = arr[size - 2, 1: size - 1]      # Fill ghost cell: Bottom
    arr[1: size - 1, 0]         = arr[1: size - 1, size - 2]      # Fill ghost cell: Left
    arr[1: size - 1, size - 1]  = arr[1: size - 1, 1]             # Fill ghost cell: Right
    return arr

def smooth(fn):
    '''Performs a Gaussian smoothing function to a 2D array.'''
    
    new_function = np.zeros((size, size), dtype=float)
    
    for ii in range(1, size - 1):
        for jj in range(1, size - 1):
            new_function[ii, jj] = (4. / 16.) * (fn[ii, jj])                                                                        \
                                 + (2. / 16.) * (fn[ii + 1, jj]     + fn[ii - 1, jj]     + fn[ii, jj + 1]     + fn[ii, jj - 1])     \
                                 + (1. / 16.) * (fn[ii + 1, jj + 1] + fn[ii - 1, jj + 1] + fn[ii + 1, jj - 1] + fn[ii - 1, jj - 1])
   
    new_function = manage_ghost_cells(new_function, 1)        
    return new_function

def check_cell_dist_2d(part, node, species):
    xlocs = (np.arange(0, size) - 0.5) * dx     # Spatial locations (x, y) of E-field nodes
    ylocs = (np.arange(0, size) - 0.5) * dy
    X, Y  = np.meshgrid(xlocs, ylocs)
    
    f     = np.zeros((1, 6), dtype=float)       # Somewhere to put the particle data
    count = 0                                   # Number of particles in area

    # Collect particle infomation if within (0.5dx, 0.5dy) of node
    for ii in range(N):
        if ((abs(part[0, ii] - xlocs[node[0]]) <= dx) and
            (abs(part[1, ii] - ylocs[node[1]]) <= dy) and
            (part[2, ii] == species)):

            f = np.append(f, [part[0:6, ii]], axis=0)
            count += 1
    print 'Node (%d, %d) recieving contributions from %d particles.' % (node[0], node[1], count)

    plt.rc('grid', linestyle='dashed', color='black', alpha=0.3)
    
    # Draw figure and spatial boundaries
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_patch(patches.Rectangle((0, 0), xmax, ymax, fill=False, edgecolor='green'))
   
    # Shade in cell containing E-field node. Not useful due to particle shapes?
    ax.add_patch(patches.Rectangle(((node[0] - 1.5)*dx, (node[1] - 1.5)*dy),    # Bottom left position
                                   2*dx, 2*dy,                                  # Rectangle dimensions
                                   facecolor='grey',                        # Rectangle colour
                                   edgecolor='none',                        # Rectangle edge colour (no edges)
                                   alpha=0.5))                              # Rectangle opacity
    
    # Draw cell grid
    ax.set_xticks(np.arange(0, xmax+dx, dx))
    ax.set_yticks(np.arange(0, ymax+dy, dy))
    plt.grid(True)

    # Plot data and set limits
    ax.scatter(X, Y, s=10, c='red', marker='^')                      # Draw nodes
    ax.scatter(part[0, :], part[1, :], s=1, c='blue')   # Draw particles
    ax.set_xlim(-dx, xmax+dx)
    ax.set_ylim(-dy, ymax+dy)
    ax.set_title(r'$N_{cell} = %d$' % (np.sqrt(cellpart/Nj)))
    
    fig2 = plt.figure(2, figsize=(12, 10))
    fig2.patch.set_facecolor('w')
    num_bins = 50
    vmag = np.sqrt(part[3, :] ** 2 + part[4, :] ** 2 + part[5, :] ** 2)

    xax = plt.subplot2grid((2, 2), (0, 0))  
    yax = plt.subplot2grid((2, 2), (0, 1))  
    zax = plt.subplot2grid((2, 2), (1, 0))  
    tax = plt.subplot2grid((2, 2), (1, 1))

    xs, BinEdgesx = np.histogram((f[:, 3] - partin[2, species]), bins=num_bins)
    bx = 0.5*(BinEdgesx[1:] + BinEdgesx[:-1])
    xax.plot(bx, xs, '-', c='c', drawstyle='steps')
    xax.set_xlabel(r'$v_x$')

    ys, BinEdgesy = np.histogram((f[:, 4]), bins=num_bins)
    by = 0.5*(BinEdgesy[1:] + BinEdgesy[:-1])
    yax.plot(by, ys, '-', c='c', drawstyle='steps')
    yax.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram((f[:, 5]), bins=num_bins)
    bz = 0.5*(BinEdgesz[1:] + BinEdgesz[:-1])
    zax.plot(bz, zs, '-', c='c', drawstyle='steps')
    zax.set_xlabel(r'$v_z$')

    ts, BinEdgest = np.histogram(vmag, bins=num_bins)
    bt = 0.5*(BinEdgest[1:] + BinEdgest[:-1])
    tax.plot(bt, ts, '-', c='c', drawstyle='steps')
    tax.set_xlabel(r'$|v|$')

    plt.show()
    return

if __name__ == '__main__':                         # Main program start
    start_time     = timer()                       # Start Timer
    drive          = '/media/yoshi/VERBATIM HD/'   # Drive letter for portable HDD (changes between computers. Use /home/USER/ for linux.)
    save_path      = 'runs/two_d_test/'            # Save path on 'drive' HDD - each run then saved in numerically sequential subfolder with images and associated data
    generate_data  = 0                             # Save data? Yes (1), No (0)
    generate_plots = 1  ;   plt.ioff()             # Save plots, but don't draw them
    run_desc = '''Full 2D test. 1eV, two proton species with isothermal electrons. Smoothing included. Just to see if anything explodes. Should be in equilibrium hopefully.'''
    
    print 'Initializing parameters...'
    set_constants()
    set_parameters()
    part, part_type, old_part     = initialize_particles()
    B, E, Vi, dns, dns_old, W, Wb = initialize_fields()
    DT, maxtime, framegrab        = set_timestep(part)
    ts_history                    = []

    for qq in range(10):
        if qq == 0:
            print 'Simulation starting...'
            W            = assign_weighting(part[0:2, :], part[6:8, :], 1)                  # Assign initial (E) weighting to particles
            Wb           = assign_weighting(part[0:2, :], part[6:8, :], 0)                  # Magnetic field weighting (due to E/B grid displacement)
            dns          = collect_density(part, W)                                         # Collect initial density   
            Vi           = collect_current(part, dns, W)                                    # Collect initial current
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], 0)                            # Initialize magnetic field (should be second?)
            E[:, :, 0:3] = push_E(B[:, :, 0:3], Vi, dns, 0)                                 # Initialize electric field
            
            initial_cell_density = dns 
            part = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], -0.5*DT, W, Wb)  # Retard velocity to N - 1/2 to prevent numerical instability
            
            #check_cell_dist_2d(part, (2, 2), 0) 

            #X, Y = np.meshgrid(np.arange(size), np.arange(size))
            #dplt = plt.subplot2grid((1, 1), (0, 0), projection='3d')
            #dplt.plot_wireframe(X, Y, dns[:, :, 1])
            #plt.show()
        else:
            # N + 1/2
            print 'Timestep %d' % qq
            
            DT, ts_flag = update_timestep(part, DT)
            if ts_flag == 1:
                ts_history.append(qq)
                if len(ts_history) >= 7:
                    sys.exit('Timestep less than 1%% of initial. Consider parameter change.')

            part          = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], DT, W, Wb)    # Advance Velocity to N + 1/2
            part, W, Wb   = position_update(part)                                           # Advance Position to N + 1
            B[:, :, 0:3]  = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                          # Advance Magnetic Field to N + 1/2
            
            dns           = 0.5 * (dns + collect_density(part, W))                          # Collect ion density at N + 1/2 : Collect N + 1 and average with N                                             
            Vi            = collect_current(part, dns, W)                                   # Collect ion flow at N + 1/2
            E[:, :, 6:9]  = E[:, :, 0:3]                                                    # Store Electric Field at N because PC, yo
            E[:, :, 0:3]  = push_E(B[:, :, 0:3], Vi, dns, DT)                               # Advance Electric Field to N + 1/2   ii = even numbers
                      
            # -------- Predictor-Corrector Method -------- #

            # Predict values of fields at N + 1 
            B[:, :, 3:6] = B[:, :, 0:3]                                                     # Store last "real" magnetic field (N + 1/2)
            E[:, :, 3:6] = E[:, :, 0:3]                                                     # Store last "real" electric field (N + 1/2)
            E[:, :, 0:3] = -E[:, :, 6:9] + 2 * E[:, :, 0:3]                                 # Predict Electric Field at N + 1
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                           # Predict Magnetic Field at N + 1 (Faraday, based on E(N + 1))
            
            # Extrapolate Source terms and fields at N + 3/2
            old_part = part                                                                 # Back up particle attributes at N + 1  
            dns_old  = dns                                                                  # Store last "real" densities (in an E-field position, I know....)
           
            part         = velocity_update(part, B[:, :, 0:3], E[:, :, 0:3], DT, W, Wb)     # Advance particle velocities to N + 3/2
            part, W, Wb  = position_update(part)                                            # Push particles to positions at N + 2
            dns          = 0.5 * (dns + collect_density(part, W))                           # Collect ion density as average of N + 1, N + 2
            Vi           = collect_current(part, dns, W)                                    # Collect ion flow at N + 3/2
            B[:, :, 0:3] = push_B(B[:, :, 0:3], E[:, :, 0:3], DT)                           # Push Magnetic Field again to N + 3/2 (Use same E(N + 1)
            E[:, :, 0:3] = push_E(B[:, :, 0:3], Vi, dns, DT)                                # Push Electric Field to N + 3/2   ii = odd numbers
            
            # Correct Fields
            E[:, :, 0:3] = 0.5 * (E[:, :, 3:6] + E[:, :, 0:3])                              # Electric Field interpolation
            B[:, :, 0:3] = push_B(B[:, :, 3:6], E[:, :, 0:3], DT)                           # Push B using new E and old B
            
            # Reset Particle Array to last real value
            part = old_part                                                                 # The stored densities at N + 1/2 before the PC method took place (previously held PC at N + 3/2)
            dns  = dns_old

        ##############################
        # -------- PLOTTING -------- #
        ##############################
        if generate_plots == 1:
            # Initialize Figure Space
            fig_size = 4, 9
            fig = plt.figure(figsize=(20,10))   
            fig.patch.set_facecolor('w')    
            
            # Set font things
            rcParams.update({'text.color'   : 'k',
                        'axes.labelcolor'   : 'k',
                        'axes.edgecolor'    : 'k',
                        'axes.facecolor'    : 'w',
                        'mathtext.default'  : 'regular',
                        'xtick.color'       : 'k',
                        'ytick.color'       : 'k',
                        'axes.labelsize'    : 16,
                        })
            
            species_colour = ['cyan', 'red']
            # Slice some things for simplicity
            sim_time    = qq * DT                   # Corresponding "real time"
            pos         = part[0:2, :] / RE         # Particle x-positions in Earth-Radii 
            x_cell_num  = np.arange(size)           # Numerical cell numbering: x-axis
            y_cell_num  = np.arange(size)
            
            alfie       = np.sum([partin[0, jj] * partin[3, jj] * ne for jj in range(Nj)])
            vel         = part[3:6, :] / alfie      # Velocities as multiples of the alfven speed 
        
            # PLOT: Spatial values of Bz
            ax_main = plt.subplot2grid(fig_size, (0, 0), projection='3d', rowspan=4, colspan=4)
            ax_main.set_title(r'$B_z$ (nT)')
            X, Y = np.meshgrid(x_cell_num, y_cell_num)

            ax_main.plot_wireframe(X, Y, (B[:, :, 2]*1e9))
            ax_main.set_xlim(0, size)
            ax_main.set_ylim(0, size)
            ax_main.set_zlim(-B0*1e9, B0*1e9)
            ax_main.view_init(elev=21., azim=300.)

            # PLOT: Spatial values of Ez
            ax_main2 = plt.subplot2grid(fig_size, (0, 4), projection='3d', rowspan=4, colspan=4)
            ax_main2.set_title(r'$E_z$ (mV)')
            X, Y = np.meshgrid(x_cell_num, y_cell_num)

            ax_main2.plot_wireframe(X, Y, (E[:, :, 2]*1e6))
            ax_main2.set_xlim(0, size)
            ax_main2.set_ylim(0, size)
            ax_main2.set_zlim(-150, 150)
            ax_main2.view_init(elev=25., azim=300.)

            ax_main.set_xlabel('x')
            ax_main.set_ylabel('y')
            ax_main.set_zlabel(r'$B_z$ (nT)')

            ax_main2.set_xlabel('x (m)')
            ax_main2.set_ylabel('y (m)')
            ax_main2.set_zlabel(r'$E_z (\mu V)$')

            plt.figtext(0.85, 0.90, 'N  = %d' % N, fontsize=24)
            plt.figtext(0.85, 0.85, r'$T_{b\parallel}$ = %.2feV' % partin[6, 1], fontsize=24)
            plt.figtext(0.85, 0.80, r'$T_{b\perp}$ = %.2feV' % partin[7, 1], fontsize=24)
            
            plt.figtext(0.85, 0.70, r'$NX$ = %d' % NX, fontsize=24)
            plt.figtext(0.85, 0.65, r'$NY$ = %d' % NY, fontsize=24)
                    
            

        ################################
        # ---------- SAVING ---------- #
        ################################
        if qq%framegrab == 0:       # Dump data at specified interval   
            r = qq / framegrab          # Capture number
       
            # Initialize run directory
            if ((generate_plots == 1 or generate_data == 1) and (qq == 0)) == True:

                if os.path.exists('%s/%s' % (drive, save_path)) == False:
                    os.makedirs('%s/%s' % (drive, save_path))              # Create master test series directory
                    print 'Master directory created'
                    
                num = len(os.listdir('%s%s' % (drive, save_path)))        # Count number of existing runs. Set to run number manually for static save
                path = ('%s/%s/run_%d' % (drive, save_path, num))          # Set root run path (for images)
                
                if os.path.exists(path) == False:
                    os.makedirs(path)
                    print 'Run directory created'            
            
            # Save Plots
            if generate_plots == 1:
                filename = 'anim%05d.png' % r
                fullpath = os.path.join(path, filename)
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
                print 'Plot %d produced' % r
                plt.close('all')            
            
            # Save Data
            if generate_data == 1:
                
                d_path = ('%s/%s/run_%d/data' % (drive, save_path, num))   # Set path for data                
                
                if os.path.exists(d_path) == False:                         # Create data directory
                    os.makedirs(d_path)
                
                if qq ==0:
                    # Save Header File: Important variables for Data Analysis
                    params = dict([('Nj', Nj),
                                   ('DT', DT),
                                   ('NX', NX),
                                   ('NY', NY),
                                   ('dx', dx),
                                   ('dy', dy),
                                   ('xmax', xmax),
                                   ('ymax', ymax),
                                   ('k' , k ),
                                   ('ne', ne),
                                   ('size', size),
                                   ('cellpart', cellpart),
                                   ('B0', B0),
                                   ('Te0', Te0),
                                   ('ie', ie),
                                   ('seed', seed),
                                   ('theta', theta),
                                   ('framegrab', framegrab),
                                   ('ts_history', ts_history),
                                   ('run_desc', run_desc)])
                                   
                    h_name = os.path.join(d_path, 'header.pckl')                                # Data file containing variables used in run
                    
                    with open(h_name, 'wb') as f:
                        pickle.dump(params, f)
                        f.close() 
                        print 'Header file saved'
                    
                    p_file = os.path.join(d_path, 'p_data')
                    np.savez(p_file, partin=partin, part_type=part_type)       # Data file containing particle information
                    print 'Particle data saved'

                d_filename = 'data%05d' % r
                d_fullpath = os.path.join(d_path, d_filename)
                np.savez(d_fullpath, part=part, Vi=Vi, dns=dns, E = E[:, 0:3], B = B[:, 0:3])   # Data file for each iteration
                print 'Data saved'
    
    #%%        ----- PRINT RUNTIME -----
    # Print Time Elapsed
    elapsed = timer() - start_time
    print "Time to execute program: {0:.2f} seconds".format(round(elapsed,2))

