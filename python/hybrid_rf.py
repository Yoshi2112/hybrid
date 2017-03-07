from timeit import default_timer as timer
import numpy as np
from numpy import pi
import os
import pickle
from BLcodes import basek_scramble                      # Input an integer (index) and a base and output a number between 0 and 1 (radix-k scrambling routine)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pdb

def check_cell_distribution(part, node_number, j): #        
    
    # Collect information about particles within +- 0.5dx of node_number (E-field nodes are in the cell centers)
    x_node = (node_number - 0.5) * dx   # Position of node in question
    f = np.zeros((1, 6))                
    count = 0           

    for ii in range(N):
        if (abs(part[0, ii] - x_node) <= 0.5*dx) and (part[8, ii] == j):       
            f = np.append(f, [part[0:6, ii]], axis=0)
            count += 1

    #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })
        
    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w') 
    num_bins = 20
    
    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))
    
    xs, BinEdgesx = np.histogram((f[:, 3] - partout[2, j]), bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x$')
    
    ys, BinEdgesy = np.histogram(f[:, 4], bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram(f[:, 5], bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z$')
    
    plt.show()    
    
    return

def check_position_distribution(part, j):
    
        #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })
        
    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w') 
    num_bins = 128
    
    ax_x = plt.subplot()    
    
    xs, BinEdgesx = np.histogram(part[0, idx_start[j]: idx_end[j]] / float(dx), bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$x_p$')
    ax_x.set_xlim(0, NX)
    
    plt.show() 
    
    return

def check_velocity_distribution(part, j):

    #Plot it
    rcParams.update({'text.color'   : 'k',
            'axes.labelcolor'   : 'k',
            'axes.edgecolor'    : 'k',
            'axes.facecolor'    : 'w',
            'mathtext.default'  : 'regular',
            'xtick.color'       : 'k',
            'ytick.color'       : 'k',
            'axes.labelsize'    : 24,
            })
        
    fig = plt.figure(figsize=(12,10))
    fig.patch.set_facecolor('w') 
    num_bins = 100
    
    ax_x = plt.subplot2grid((2, 3), (0,0), colspan=2, rowspan=2)
    ax_y = plt.subplot2grid((2, 3), (0,2))
    ax_z = plt.subplot2grid((2, 3), (1,2))
    
    xs, BinEdgesx = np.histogram(part[3, idx_start[j]: idx_end[j]] / alfie, bins=num_bins)
    bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
    ax_x.plot(bx, xs, '-', c='c', drawstyle='steps')
    ax_x.set_xlabel(r'$v_x$')
    #ax_x.set_xlim(6, 14)
    
    ys, BinEdgesy = np.histogram(part[4, idx_start[j]: idx_end[j]] / alfie, bins=num_bins)
    by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
    ax_y.plot(by, ys, '-', c='c', drawstyle='steps')
    ax_y.set_xlabel(r'$v_y$')
    
    zs, BinEdgesz = np.histogram(part[5, idx_start[j]: idx_end[j]] / alfie, bins=num_bins)
    bz = 0.5 * (BinEdgesz[1:] + BinEdgesz[:-1])
    ax_z.plot(bz, zs, '-', c='c', drawstyle='steps')
    ax_z.set_xlabel(r'$v_z$')

    plt.show()
    
    return

def set_constants():
    global q, c, mp, mu0, kB, e0    
    q   = 1.602e-19                             # Elementary charge (C)
    c   = 3e8                                   # Speed of light (m/s)
    mp  = 1.67e-27                              # Mass of proton (kg)
    mu0 = (4e-7) * pi                           # Magnetic Permeability of Free Space (SI units)
    kB  = 1.38065e-23                           # Boltzmann's Constant (J/K)
    e0  = 8.854e-12                             # Epsilon naught - permittivity of free space
    return    
    
    
def set_parameters():
    global dxm, t_res, NX, max_sec, cellpart, ie, B0, size, N, k, ne, xmax, f0, E0
    
    f0       = 8e7                              # Frequency of source wave
    E0       = 2e4                              # Amplitude of source wave
    
    xmax     = 0.004                            # Max domain size in meters
    NX       = 128                              # Number of cells - dimension of array (not including ghost cells)
    max_sec  = 150 * 1 / f0                     # Number of (real) seconds to run program for - 150 periods of f0   
    t_res    = max_sec * 1e-3                   # Time resolution of output data
    cellpart = 500                              # Number of Particles per cell (make it an even number for 50/50 hot/cold)
    ie       = 1                                # Adiabatic electrons. 0: off (constant), 1: on.    
    B0       = 3.504                            # Unform initial magnetic field value (in T) (must be parallel to an axis)
    ne       = 1e19                             # Electron number density (/m3)

    # Derived Values ##
    size     = NX + 2
    N        = cellpart*NX                      # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells

    

    return    

def initialize_particles():
    np.random.seed(21)                          # Random seed 
    global Nj, Te0, partout, n_rel, n0, alfie, dx, partin, idx_start, idx_end, xmax, wpi
        
    # Species Characteristics - use column number as species identifier
    #                           D+            
    partin = np.array([[  1.00000000e+00],      # (0) Mass   (proton units)
                       [  1.00000000e+00],      # (1) Charge (elementary units)
                       [               0],      # (2) Bulk velocity (multiples of vA)
                       [              ne],      # (3) Real density (/m3)
                       [               1],      # (4) Simulated density (portion of 1)
                       [               0],      # (5) Distribution Type (not used here)
                       [            5.00],      # (6) Parallel temperature (eV)
                       [            5.00],      # (7) Perpendicular temperature (eV)
                       [  0.00000000e+00]])     # (8) Hot/Cold (0/1) species flag
    
    part_type     = ['$H^{+}$']
    
    n0            = np.sum(partin[3, :])                                    # Total ion density - initial density per cell (in m-3 : First number representative of density in cm-3 )
    Nj            = int(np.shape(partin)[1])                                # Number of species (number of columns above)    
    wpi           = np.sqrt((n0 * (q**2)) / (mp * e0 ))                     # Plasma Frequency (rad/s)
    dx            = xmax / float(NX)
    
    N_real        = (dx * 1. * 1.) * (n0) * NX                              # Total number of real, mobile particles (rect prism with sides dx x 1 x 1 metres)
    n_rel         = np.asarray([partin[3, nn] / n0 for nn in range(Nj)])    # Relative proportion (out of 1) of each species
    N_species     = np.round(N * partin[4, :]).astype(int)                  # Number of sim particles for each species, total    
        
    rho      = np.sum([partin[0, nn] * mp * partin[3, nn] for nn in range(Nj)])    # Total mass density
    alfie    = B0/np.sqrt(mu0 * rho)                                               # Alfven Velocity (m/s): Constant (but only used for initialization)
    
    # Output Particle Values
    partout = np.array([partin[0, :] * mp,                       # (0) Actual Species Mass    (in kg)
                        partin[1, :] * q,                        # (1) Actual Species Charge  (in coloumbs)
                        partin[2, :] * alfie,                    # (2) Actual Species streaming velocity
                        (N_real * n_rel) / N_species])           # (3) Density contribution of each particle of species (real particles per sim particle)

    # Particle Array: Initalization and Loading
    part     = np.zeros((9, N), dtype=float)         # Create array of zeroes N x 13 for pos, vel and F 3-vectors
    old_part = np.zeros((9, N), dtype=float)         # Place to store last particle states while using Predictor-Corrector method
    
    Tpar = partin[6, :] * 11603
    Tper = partin[7, :] * 11603
    Te0  = Tpar[0]

    idx_start = [np.sum(N_species[0:ii]    )     for ii in range(0, Nj)]                     # Start index values for each species in order
    idx_end   = [np.sum(N_species[0:ii + 1])     for ii in range(0, Nj)]                     # End   index values for each species in order
    idx       = 0    
    
    for ii in N_species:                             
    
        part[8, idx_start[idx]: idx_end[idx]] = idx      # Give index identifier to each particle  
        m    = partout[0, idx]                           # Species mass
        vpar = 0#p.sqrt(    kB * Tpar[idx] / m)          # Species parallel thermal velocity (x)
        vper = 0#np.sqrt(2 * kB * Tper[idx] / m)         # Species perpendicular thermal velocity (y, z)

        for jj in range(ii):
            part[0, idx_start[idx] + jj] = xmax * basek_scramble(jj, 2)     # Assign position
            
            thetaR = basek_scramble(jj, 3)                                  # Scrambling set for theta (y, z)
            thetaX = basek_scramble(jj, 7)                                  # Scrambling set for theta (x) - throws away second component
            vr_per = vper * np.sqrt(-2 * np.log( (jj + 0.5) / ii))          # Maxwellian for perpendicular velocity component
            vr_par = vpar * np.sqrt(-2 * np.log( (jj + 0.5) / ii))          # Maxwellian for parallel (streaming) velocity component
            
            part[3, idx_start[idx] + jj] = vr_par * np.cos(2 * pi * thetaX) + partout[2, idx]   # Parallel particle velocity (distribution + streaming)
            part[4, idx_start[idx] + jj] = vr_per * np.sin(2 * pi * thetaR)                     # Perpendicular (y) particle velocity
            part[5, idx_start[idx] + jj] = vr_per * np.cos(2 * pi * thetaR)                     # Perpendicular (z) particle velocity
        
        idx += 1                                    # Move into next species
           
    part[6, :] = part[0, :] / dx + 0.5 ; part[6, :] = part[6, :].astype(int)                        # Initial leftmost node, I

    return part, part_type, old_part


def set_timestep(part):
    gyfreq   = q*B0/mp                          # Proton Gyrofrequency (rad/s) (since this will be the highest of all species)
    gyperiod = 2*pi / gyfreq                    # Gyroperiod in seconds
    ion_ts   = 0.05 * gyperiod                  # Timestep to resolve gyromotion
    vel_ts   = dx / (2 * np.max(part[3:6, :]))  # Timestep to satisfy CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    
    DT        = 0.6 * min(ion_ts, vel_ts)       # Smallest of the two
    framegrab = int(t_res / DT)                 # Number of iterations between dumps
    maxtime   = int(max_sec / DT) + 1           # Total number of iterations to achieve desired final time
    
    if framegrab == 0:
        framegrab = 1
    
    print '\nProton gyroperiod = %.2fs' % gyperiod
    print 'Timestep: %es, %d iterations total\n' % (DT, maxtime)
    return DT, maxtime, framegrab

    
def initialize_fields():  
    global Bc, theta

    theta = 0                                   # Angle of B0 to x axis (in xy plane in units of degrees)
    Bc    = np.zeros(3)                         # Constant components of magnetic field based on theta and B0
    
    ## Magnetic Field Stuff ##
    Bc[0]   = B0 * np.sin((90 - theta) * pi / 180 )   # Constant x-component of magnetic field (theta in degrees)
    Bc[1]   = B0 * np.cos((90 - theta) * pi / 180 )   # Constant Y-component of magnetic field (theta in degrees)  
    Bc[2]   = 0                                       # Assume Bzc = 0, orthogonal to field line direction
    
    # Initialize Field Arrays: (size, size, 3) for 2D, (size, size, size, 3) for 3D
    B = np.zeros((size, 6), dtype=float)    # Magnetic Field Array of shape (size), each element holds 3-vector
        # Where:
        #       B[mm, 0-2] represent the current field and
        #       B[mm, 3-5] store the last state of the magnetic field previous to the Predictor-Corrector scheme
    B[:, 0] = Bc[0]      # Set Bx initial
    B[:, 1] = Bc[1]      # Set By initial
    B[:, 2] = Bc[2]      # Set Bz initial
    
    E = np.zeros((size, 9), dtype=float)    # Electric Field Array
        # Where:
        #       E[mm, 0-2] represent the current field and
        #       E[mm, 3-5] store the last state of the electric field previous to the Predictor-Corrector scheme E (N + 0.5)
        #       E[mm, 6-8] store two steps ago: E-field at E^N
    E[:, 0] = 0         # Set Ex initial
    E[:, 1] = 0         # Set Ey initial
    E[:, 2] = 0         # Set Ez initial
    
    Vi      = np.zeros((size, Nj, 3), dtype=float)          # Ion Flow (3 dimensions)
    dns     = np.zeros((size, Nj),    dtype=float)          # Species number density in each cell (in /m3)
    dns_old = np.zeros((size, Nj),    dtype=float)          # For PC method
    W       = np.zeros(N,             dtype=float)          # Particle Weighting (IDW)
    
    return B, E, Vi, dns, dns_old, W    
    

def velocity_update(part, B, E, dt, W_in):  # Based on Appendix A of Ch5 : Hybrid Codes by Winske & Omidi.
  
    Wb = assign_weighting(part[0, :], part[6, :], 0)        # Magnetic field weighting
    
    for n in range(N):
        vn = part[3:6, n]                # Existing particle velocity
        
        # Weighted E & B fields at particle location (node) - Weighted average of two nodes on either side of particle
        I   = int(part[6,n])             # Nearest (leftmost) node, I
        Ib  = int(part[0, n] / dx)       # Nearest (leftmost) magnetic node
        We  = W_in[n]                    # E-field weighting
        idx = int(part[8, n])
        
        E_p = E[I,  0:3] * (1 - We   ) + E[I  + 1, 0:3] * We
        B_p = B[Ib, 0:3] * (1 - Wb[n]) + B[Ib + 1, 0:3] * Wb[n]    
        
        # Intermediate calculations
        h = (partout[1 , idx] * dt) / partout[0 , idx] 
        f = 1 - (h**2) / 2 * (B_p[0]**2 + B_p[1]**2 + B_p[2]**2 )
        g = h / 2 * (B_p[0]*vn[0] + B_p[1]*vn[1] + B_p[2]*vn[2])
        v0 = vn + (h/2)*E_p
    
        # Velocity push
        part[3,n] = f * vn[0] + h * ( E_p[0] + g * B_p[0] + (v0[1]*B_p[2] - v0[2]*B_p[1]) )
        part[4,n] = f * vn[1] + h * ( E_p[1] + g * B_p[1] - (v0[0]*B_p[2] - v0[2]*B_p[0]) )
        part[5,n] = f * vn[2] + h * ( E_p[2] + g * B_p[2] + (v0[0]*B_p[1] - v0[1]*B_p[0]) )
        
    return part        
    
    
def position_update(part):  # Basic Push (x, v) vectors and I, W update
    
    # Update position vectors - turn off y,z for 1D sim
    part[0, :] += part[3, :] * DT
    xmax = NX * dx
    
    # Periodic Boundary Condition: xmax = NX * dx
    for ii in range(N):
        if part[0, ii] < 0:
            part[0, ii] = xmax + part[0,ii]
            
        if part[0, ii] > xmax:
            part[0, ii] = part[0,ii] - xmax
    
    part[6, :] = part[0, :] / dx + 0.5             # Leftmost node, I
    part[6, :] = part[6, :].astype(int)            # Integer-ize
    W_out      = assign_weighting(part[0, :], part[6, :], 1)
    return part, W_out
    
    
def assign_weighting(xpos, I, BE):                      # Magnetic/Electric Field, BE == 0/1
    W_o = ((xpos)/(dx)) - I + (BE / 2.)       # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
    return W_o
    
    
def push_B(B, E, dt):   # Basically Faraday's Law. (B, E) vectors
        
    Bp = np.zeros((size, 3), dtype=float)

    # Consider B1 only (perturbation)    
    Bp[:, 0] = B[:, 0] - Bc[0]    
    Bp[:, 1] = B[:, 1] - Bc[1]
    Bp[:, 2] = B[:, 2] - Bc[2]   
    
    # Fill ghost cells
    Bp[0, :] = Bp[size - 2, :]
    E[ 0, :] = E[ size - 2, :]
    Bp[size - 1] = Bp[1]
    E[ size - 1] = E[ 1]

    for mm in range (1, size - 1):
        Bp[mm, 0] = Bp[mm, 0] - (dt / 2) * 0                                                # Since derivatives for y and z go to zero
        Bp[mm, 1] = Bp[mm, 1] - (dt / 2) * ( (E[mm - 1, 2] - E[mm + 1, 2]) / (2 * dx) )
        Bp[mm, 2] = Bp[mm, 2] - (dt / 2) * ( (E[mm + 1, 1] - E[mm - 1, 1]) / (2 * dx) )
        
    # Update ghost cells
    Bp[0, :] = Bp[size - 2, :]
    Bp[size - 1, :] = Bp[1, :]
    
    # Combine total B-field: B0 + B1    
    B[:, 0] = Bp[:, 0] + Bc[0]    
    B[:, 1] = Bp[:, 1] + Bc[1]
    B[:, 2] = Bp[:, 2] + Bc[2] 
    
    return B
    
def push_E(B, V_i, n_i, dt, qq): # Based off big F(B, n, V) eqn on pg. 140 (eqn. 10)

    global J
        
    E_out = np.zeros((size, 3))     # Output array - new electric field
    JxB   = np.zeros((size, 3))     # V cross B holder
    BdB   = np.zeros((size, 3))     # B cross del cross B holder     
    del_p = np.zeros((size, 3))     # Electron pressure tensor gradient array
    J     = np.zeros((size, 3))     # Ion current
    qn    = np.zeros( size,    dtype=float)     # Ion charge density

    E_source = E0 * np.sin(2 * pi * f0 * (qq - 0.5) * dt)           # No initial subtraction from E needed since E is calculated fresh per timestep from source terms.

    # Adiabatic Electron Temperature Calculation   
    if ie == 1:    
        gamma = 5./3.
        ni = np.asarray([np.sum(n_i[xx, :]) for xx in range(size)])
        Te = Te0 * ((ni / (n0)) ** (gamma - 1))  
    else:
        Te = [Te0 for ii in range(size)]
        
    # Calculate average/summations over species
    for jj in range(Nj):
        qn += partout[1, jj] * n_i[:, jj]                  # Total charge density, sum(qj * nj)
        
        for kk in range(3):
            J[:, kk]  += partout[1, jj] * n_i[:, jj] * V_i[:, jj, kk]   # Total ion current vector: J_k = qj * nj * Vj_k
            
    # J cross B
    JxB[:, 0] +=    J[:, 1] * B[:, 2] - J[:, 2] * B[:, 1]  
    JxB[:, 1] += - (J[:, 0] * B[:, 2] - J[:, 2] * B[:, 0]) 
    JxB[:, 2] +=    J[:, 0] * B[:, 1] - J[:, 1] * B[:, 0]   
    
    for mm in range(1, size - 1):
        
        # B cross curl B
        BdB[mm, 0] =    B[mm, 1]  * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx)) + B[mm, 2] * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
        BdB[mm, 1] = (- B[mm, 0]) * ((B[mm + 1, 1] - B[mm - 1, 1]) / (2 * dx))
        BdB[mm, 2] = (- B[mm, 0]) * ((B[mm + 1, 2] - B[mm - 1, 2]) / (2 * dx))
    
        # del P
        del_p[mm, 0] = ((qn[mm + 1] - qn[mm - 1]) / (2*dx*q)) * kB * Te[mm]
        del_p[mm, 1] = 0
        del_p[mm, 2] = 0
    
    # Final Calculation
    E_out[:, 0] = (- JxB[:, 0] - (del_p[:, 0] ) - (BdB[:, 0] / (mu0))) / (qn[:])
    E_out[:, 1] = (- JxB[:, 1] - (del_p[:, 1] ) - (BdB[:, 1] / (mu0))) / (qn[:])
    E_out[:, 2] = (- JxB[:, 2] - (del_p[:, 2] ) - (BdB[:, 2] / (mu0))) / (qn[:])
    
    # Update ghost cells
    E_out[0, :]        = E_out[size - 2, :]
    E_out[size - 1, :] = E_out[1, :]
    
    # Add source term
    source_location = NX / 2                
    E_out[source_location, 0] += E_source
    
    return E_out


def collect_density(I_in, W_in, ptype): 
    '''Function to collect charge density in each cell in each cell
    at each timestep. These values are weighted by their distance
    from cell nodes on each side. Can send whole array or individual particles?
    How do I sum up the densities one at a time?'''

    n_i = np.zeros((size, Nj), float)
   
    # Collect number density of all particles
    for ii in range(N):
        
        I   = int(I_in[ii])
        W   = W_in[ii]
        idx = int(ptype[ii])
        
        n_i[I,     idx] += (1 - W) * partout[3, idx]
        n_i[I + 1, idx] +=      W  * partout[3, idx]
    
    # Move ghost cell contributions - Ghost cells at 0 and size - 2
    n_i[size - 2, :] += n_i[0, :]
    n_i[0, :]         = n_i[size - 2, :]              # Fill ghost cell  
    
    n_i[1, :]       += n_i[size - 1, :]
    n_i[size - 1, :] = n_i[1, :]                      # Fill ghost cell
    
    n_i /= float(dx)        # Divide by cell dimensions to give densities per cubic metre
 
    # Smooth density using Gaussian smoother (1/4, 1/2, 1/4)
    for jj in range(Nj):
        smoothed   = smooth(n_i[:, jj])
        n_i[:, jj] = smoothed
     
    return n_i


def collect_flow(part, ni, W_in): 
    
    # Empty 3-vector for flow velocities at each node
    V_i = np.zeros((size, Nj, 3), float)    
    
    # Loop through all particles: sum velocities for each species. Alter for parallelization?
    for ii in range(N):
        I   = int(part[6, ii])
        idx = int(part[8, ii])
        W   =     W_in[ii]
    
        V_i[I, idx, 0] += (1 - W) * partout[3, idx] * part[3, ii]
        V_i[I, idx, 1] += (1 - W) * partout[3, idx] * part[4, ii]
        V_i[I, idx, 2] += (1 - W) * partout[3, idx] * part[5, ii]
        
        V_i[I + 1, idx, 0] +=  W  * partout[3, idx] * part[3, ii]
        V_i[I + 1, idx, 1] +=  W  * partout[3, idx] * part[4, ii]
        V_i[I + 1, idx, 2] +=  W  * partout[3, idx] * part[5, ii]
        
    # Move ghost cell contributions and mirror end values
    V_i[size - 2, :, :] += V_i[0, :, :]
    V_i[0, :, :] = V_i[size - 2, :, :]                # Fill ghost cell
    
    V_i[1, :, :]  += V_i[size - 1, :, :]
    V_i[size - 1, :, :] = V_i[1, :, :]                # Fill ghost cell
    
    for ii in range(3):                               # Divide each dimension by density for averaging (ion flow velocity)
        V_i[:, :, ii] /= (ni * dx)                    # ni is in m3 - multiply by dx to get entire cell's density (for averaging purposes) 
        
    # Smooth ion velocity as with density for each species/component
    for jj in range(Nj):
        for kk in range(3):
            smoothed       = smooth(V_i[:, jj, kk])
            V_i[:, jj, kk] = smoothed
    
    #V = [np.mean(V_i[xx, 0, :]) for xx in range(size)]
    #pdb.set_trace()
    
    return V_i
    

def smooth(function): 
    
    new_function = np.zeros(size)
    
    # Smooth: Assumes nothing in ghost cells
    for ii in range(1, size - 1):
        new_function[ii - 1] = 0.25*function[ii] + new_function[ii - 1]
        new_function[ii]     = 0.5*function[ii]  + new_function[ii]
        new_function[ii + 1] = 0.25*function[ii] + new_function[ii + 1]
        
    # Move Ghost Cell Contributions: Periodic Boundary Condition
    new_function[1]        += new_function[size - 1]
    new_function[size - 2] += new_function[0]
    
    # Set ghost cell values to mirror corresponding real cell
    new_function[0]        = new_function[size - 2]
    new_function[size - 1] = new_function[1]
    return new_function
    

#%% Main Program Script
if __name__ == '__main__':
    
    # Metadata
    start_time     = timer()                       # Start Timer
    drive          = 'C:/'                         # Drive letter for portable HDD (changes between computers) - INCLUDE COLON. For UNIX, put home path here
    save_path      = 'Runs/Jenkins'                # Save path on 'drive' HDD - each run then saved in numerically sequential subfolder with images and associated data
    generate_data  = 0            ;  plt.ioff()    # Save data? Yes (1), No (0)
    generate_plots = 1  
    run_desc = '''1D hybrid code reproduction of PIC code parameters from Jenkins et al. (2013).'''
    
    # Initialize Things
    print 'Initializing parameters...'
    set_constants()
    set_parameters()
    part, part_type, old_part = initialize_particles()
    B, E, Vi, dns, dns_old, W = initialize_fields()

    DT, maxtime, framegrab    = set_timestep(part)
    
    # Numerical checks
    which_species = 0
    which_cell    = 50
    
    #check_cell_distribution(part, which_cell, which_species)
    #check_velocity_distribution(part, which_species)
    #check_position_distribution(part, which_species)
    
    for qq in range(maxtime):
        if qq == 0:

            print 'Simulation starting...'

            W           = assign_weighting(part[0, :], part[6, :], 1)                       # Assign initial (E) weighting to particles
            dns         = collect_density(part[6, :], W, part[8, :])                        # Collect initial density   
            Vi          = collect_flow(part, dns, W)                                        # Collect initial current
            
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], 0)                                     # Initialize magnetic field (should be second?)
            E[:, 0:3] = push_E(B[:, 0:3], Vi, dns, 0, qq)                                   # Initialize electric field
            
            part = velocity_update(part, B[:, 0:3], E[:, 0:3], -0.5*DT, W)                  # Retard velocity to N - 1/2 to prevent numerical instability
            
        else:
            # N + 1/2
            part      = velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)                  # Advance Velocity to N + 1/2
            part, W   = position_update(part)                                               # Advance Position to N + 1
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], DT)                                    # Advance Magnetic Field to N + 1/2
            
            dns       = 0.5 * (dns + collect_density(part[6, :], W, part[8, :]))            # Collect ion density at N + 1/2 : Collect N + 1 and average with N                                             
            Vi        = collect_flow(part, dns, W)                                          # Collect ion flow at N + 1/2
            E[:, 6:9] = E[:, 0:3]                                                           # Store Electric Field at N because PC, yo
            E[:, 0:3] = push_E(B[:, 0:3], Vi, dns, DT, qq)                                  # Advance Electric Field to N + 1/2
            
            
            # ----- Predictor-Corrector Method ----- #

            # Predict values of fields at N + 1 
            B[:, 3:6] = B[:, 0:3]                                                           # Store last "real" magnetic field (N + 1/2)
            E[:, 3:6] = E[:, 0:3]                                                           # Store last "real" electric field (N + 1/2)
            E[:, 0:3] = -E[:, 6:9] + 2*E[:, 0:3]                                            # Predict Electric Field at N + 1
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], DT)                                    # Predict Magnetic Field at N + 1 (Faraday, based on E(N + 1))
            
            # Extrapolate Source terms and fields at N + 3/2
            old_part = part                                                                 # Back up particle attributes at N + 1  
            dns_old = dns                                                                   # Store last "real" densities (in an E-field position, I know....)
            
            part = velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)                       # Advance particle velocities to N + 3/2
            part, W = position_update(part)                                                 # Push particles to positions at N + 2
            dns  = 0.5 * (dns + collect_density(part[6, :], W, part[8, :]))                 # Collect ion density as average of N + 1, N + 2
            Vi   = collect_flow(part, dns, W)                                               # Collect ion flow at N + 3/2
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], DT)                                    # Push Magnetic Field again to N + 3/2 (Use same E(N + 1)
            E[:, 0:3] = push_E(B[:, 0:3], Vi, dns, DT, qq + 1)                              # Push Electric Field to N + 3/2   ii = odd numbers
            
            # Correct Fields
            E[:, 0:3] = 0.5 * (E[:, 3:6] + E[:, 0:3])                                       # Electric Field interpolation
            B[:, 0:3] = push_B(B[:, 3:6], E[:, 0:3], DT)                                    # Push B using new E and old B
            
            # Reset Particle Array to last real value
            part = old_part                                                                 # The stored densities at N + 1/2 before the PC method took place (previously held PC at N + 3/2)
            dns  = dns_old     
        
        if qq%5 == 0:
            print 'Iteration %d of %d complete (%ds)' % (qq, maxtime, int(timer() - start_time))
            
# ----- Plot commands ----- #
        if generate_plots == 1:
            # Initialize Figure Space
            fig_size = 4, 6
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
            
            # Set some things    
            sim_time = qq * DT
                
            x_pos = part[0, 0:N] * 1000                 # Particle x-positions (km) (For looking at particle characteristics)  
            x_cell_num = np.arange(size - 2)            # Numerical cell numbering: x-axis
      
        #####       
            # Plot: Normalized vy - Assumes species 0 is hot hydrogen population
            vy_pos_hot =  0, 0
            
            ax_vy_hot = plt.subplot2grid(fig_size, vy_pos_hot, rowspan=4, colspan=3)    
            
            # Normalized to Alfven velocity: For y-axis plot
            norm_yvel   = part[4, 0:N] / 1e5        # vy (vA / ms-1)
            
            # Plot Scatter
            ax_vy_hot.scatter(x_pos[idx_start[0]: idx_end[0]], norm_yvel[idx_start[0]: idx_end[0]], s=1, c='r', lw=0)        # Hot population

            # Make it look pretty
            ax_vy_hot.set_title(r'Normalized velocity $v_y$ vs. Position (x)')    
            
            ax_vy_hot.set_xlim(0, xmax*1000)
            ax_vy_hot.set_ylim(-3, 3)
            ax_vy_hot.set_xlabel('Position (mm)', labelpad=10)
            ax_vy_hot.set_ylabel(r'$v_y \; (10^{-5} ms^{-1})$', fontsize=24, rotation=90, labelpad=8) 
                       
            plt.text(0, -2.99, 'N = %d' % N, fontsize='16')
        
        #####
            # Plot Density
            den_pos = 0, 3
            ax_den = plt.subplot2grid((fig_size), (den_pos), colspan=3)
            
            # Plot the Things
            dns_norm = np.zeros((size - 2, Nj), dtype=float)            
            
            for ii in range(Nj):
                if partin[5, ii] == 1:
                    dns_norm[:, ii] = dns[1: size-1, ii] / (partin[3, ii])
                else:
                    dns_norm[:, ii] = dns[1: size-1, ii] / partin[3, ii]
                    
            species_colors = ['red', 'cyan']
            
            for ii in range(Nj):
                ax_den.plot(x_cell_num, dns_norm[:, ii], color=species_colors[ii])
            
            # Make things pretty
            ax_den.set_title('Normalized Ion Densities and Magnetic Fields (y, mag) vs. Cell')
            ax_den.set_xlim(0, NX)
            ax_den.set_ylim(0, 2)
            ax_den.set_ylabel('Normalized Density', fontsize=14, rotation=90, labelpad=5)
            
        #####
            # Plot: Electric Field Ez
            ax_Ez = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
        
            Ez = E[1:size-1, 0]
            
            ax_Ez.plot(x_cell_num, Ez, color='magenta')
            
            ax_Ez.set_xlim(0, NX)
            
            #ax_Ez.set_ylim(-200e-6, 200e-6)
            #ax_Ez.set_yticks(np.arange(-200e-6, 201e-6, 50e-6))
            #ax_Ez.set_yticklabels(np.arange(-150, 201, 50))   
            #ax_Ez.set_ylabel(r'$E_z$ ($\mu$V)', labelpad=25, rotation=0, fontsize=14)
            
            
        #####
            # Plot: Magnetic Field - y component + magnitude
            ax_By = plt.subplot2grid((fig_size), (2, 3), colspan=3, sharex=ax_den)
            ax_B  = plt.subplot2grid((fig_size), (3, 3), colspan=3, sharex=ax_den)
                
            # Normalize field to initial B0
            mag_B = (np.sqrt(B[1:size-1, 0] ** 2 + B[1:size-1, 1] ** 2 + B[1:size-1, 2] ** 2)) / B0
            B_y = B[1:size-1, 1] / B0
            
            ax_B.plot(x_cell_num, mag_B, color='g')
            ax_By.plot(x_cell_num, B_y, color='g')
                
            ax_B.set_xlim(0,  NX)
            ax_By.set_xlim(0, NX)
                
            ax_B.set_ylim(0, 4)
            ax_By.set_ylim(-2, 2)
                
            ax_B.set_ylabel( r'$|B|$', rotation=0, labelpad=20)
            ax_By.set_ylabel(r'$B_y$', rotation=0, labelpad=10)
            ax_B.set_xlabel('Cell Number')
                    
            for ax in [ax_den, ax_Ez, ax_By]:
                plt.setp(ax.get_xticklabels(), visible=False)
                # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
                ax.set_yticks(ax.get_yticks()[1:])  
                
            #for ax in [ax_den, ax_Ez, ax_By, ax_B]:
                #ax.yaxis.tick_right()
                #ax.set_label_position("right")
                
            # Last Minute plot adjustments
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0) 
            

        if qq%framegrab == 0:       # Dump data at specified interval   
        
            r = qq / framegrab          # Capture number
        
            # Initialize run directory
            if ((generate_plots == 1 or generate_data == 1) and (qq == 0)) == True:

                if os.path.exists('%s%s' % (drive, save_path)) == False:
                    os.makedirs('%s%s' % (drive, save_path))              # Create master test series directory
                    print 'Master directory created'
                    
                num  = 0    #len(os.listdir('%s%s' % (drive, save_path)))        # Count number of existing runs. Set to run number manually for static save

                path = ('%s%s/Run %d' % (drive, save_path, num))          # Set root run path (for images)
                
                
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
                
                d_path = ('%s%s/Run %d/Data' % (drive, save_path, num))   # Set path for data                
                
                if os.path.exists(d_path) == False:                         # Create data directory
                    os.makedirs(d_path)
                
                if qq ==0:
                    # Save Header File: Important variables for Data Analysis
                    params = dict([('Nj', Nj),
                                   ('DT', DT),
                                   ('NX', NX),
                                   ('dxm', dxm),
                                   ('dx', dx),
                                   ('size', size),
                                   ('cellpart', cellpart),
                                   ('n0', n0),
                                   ('B0', B0),
                                   ('Te0', Te0),
                                   ('ie', ie),
                                   ('theta', theta),
                                   ('framegrab', framegrab),
                                   ('run_desc', run_desc)])
                                   
                    h_name = os.path.join(d_path, 'Header.pckl')                                # Data file containing variables used in run
                    
                    with open(h_name, 'wb') as f:
                        pickle.dump(params, f)
                        f.close() 
                        print 'Header file saved'
                    
                    p_file = os.path.join(d_path, 'p_data')
                    np.savez(p_file, partin=partin, partout=partout, part_type=part_type)       # Data file containing particle information
                    print 'Particle data saved'

                d_filename = 'data%05d' % r
                d_fullpath = os.path.join(d_path, d_filename)
                np.savez(d_fullpath, part=part, Vi=Vi, dns=dns, E = E[:, 0:3], B = B[:, 0:3])   # Data file for each iteration
                print 'Data saved'
    
    #%%        ----- PRINT RUNTIME -----
    # Print Time Elapsed
    elapsed = timer() - start_time
    print "Time to execute program: {0:.2f} seconds".format(round(elapsed,2))
