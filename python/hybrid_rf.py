from timeit import default_timer as timer
import numpy as np
from numpy import pi
import os
import pickle
from BLcodes import basek_scramble                      # Input an integer (index) and a base and output a number between 0 and 1 (radix-k scrambling routine)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pdb
import sys

def check_CFL():
    global S
    S = c * (DT / dx)

    if S > 1:
        print 'Courant condition violated. Change space/time step.\nS = %d' % int(S)
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
    global dxm, t_res, NX, max_sec, cellpart, ie, B0, size, N, k, ne, xmax, f0, E0, src_loc, src_dim, dt_OVR
    xmax     = 0.004                            # Max domain size in meters
    NX       = 256                              # Number of cells - dimension of array (not including ghost cells)
    t_res    = 0                                # Time resolution of output data. Set to 0 to output every timestep.
    cellpart = 100                              # Number of Particles per cell (make it an even number for 50/50 hot/cold)
    ie       = 0                                # Adiabatic electrons. 0: off (constant), 1: on.    
    B0       = 3.504                            # Unform initial magnetic field value (in T) (must be parallel to an axis)
    ne       = 1e19                             # Electron number density (/m3)

    # Derived Values ##
    size     = NX + 2
    N        = cellpart*NX                      # Number of Particles to simulate: # cells x # particles per cell, excluding ghost cells
 
    f0       = 8e7                              # Frequency of source wave
    E0       = 2e4                              # Amplitude of source wave
    src_loc  = size / 2                         # Location of E0(t)
    src_dim  = 1                                # Which dimension the E0 field is oscillating in
    dt_OVR   = 0                                # Forced value for dt (used for diagnostics). 0 means no override.
    
    max_sec  = 10 / f0                          # Number of (real) seconds to run program for - 150 periods of f0   
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
                       [            0.00],      # (6) Parallel temperature (eV)
                       [            0.00],      # (7) Perpendicular temperature (eV)
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
        vpar = np.sqrt(    kB * Tpar[idx] / m)          # Species parallel thermal velocity (x)
        vper = np.sqrt(2 * kB * Tper[idx] / m)         # Species perpendicular thermal velocity (y, z)

        for jj in range(ii):
            part[0, idx_start[idx] + jj] = (float(jj) / float(ii)) * xmax   #xmax * basek_scramble(jj, 2)     # Assign position
            
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
    vel_ts   = dx / (2 * np.max(part[3:6, :]))  # Timestep to satisfy particle CFL condition: Fastest particle doesn't traverse more than half a cell in one time step
    cfl_ts   = 0.001 * dx / c                   # Timestep to satisfy wave CFL condition
    f0_ts    = 1 / (20 * f0)                    # Timestep to resolve varying E0

    if dt_OVR == 0:
        DT = min(ion_ts, vel_ts, f0_ts, cfl_ts) # Smallest of the two
    else:
        DT = dt_OVR
    
    framegrab = int(t_res / DT)                 # Number of iterations between dumps
    maxtime   = int(max_sec / DT) + 1           # Total number of iterations to achieve desired final time

    if framegrab == 0:
        framegrab = 1
    
    print '\nProton gyroperiod = %es' % gyperiod
    print 'Timestep: %es, %d iterations total\n' % (DT, maxtime)

    return DT, maxtime, framegrab

def update_dt(maxtime, DT, framegrab, Vx):
    V_max = np.max(Vx)              # Particle maximum velocity
    
    qm    = np.max(partout[1, :] / partout[0, :])             
    dVmax = DT * qm * 2*pi*f0*E0    # Maximum possible change in E(t)
    
    while ((V_max+dVmax) > (0.5 * dx/DT)):  # If max particle velocity covers more than half a cell in DT
        DT        /= 2             # Halve the timestep
        maxtime   *= 2             # Double the number of future iterations
        framegrab *= 2             # Double the length between data dumps
        print 'Timestep halved. %d iterations remain' % maxtime     # Display that it has been halved and print how many iterations remain now.
    return maxtime, DT, framegrab

    
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
    
    
def assign_weighting(xpos, I, BE):                 # Magnetic/Electric Field, BE == 0/1
    W_o = (xpos / dx) - I + (BE / 2.)              # Last term displaces weighting factor by half a cell by adding 0.5 for a retarded electric field grid (i.e. all weightings are 0.5 larger due to further distance from left node, and this weighting applies to the I + 1 node.)
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
        Bp[mm, 0] = Bp[mm, 0] - (dt / 2) * 0        # Since no y,z spatial derivatives in 1D
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
    
def E_ext(qq, dt):
    '''Returns the value for the driving external electric field at t = qq*dt. Minus 1/2 because field is
        analytically evaluated only at the half timestep (0.5, 1.5). Full timesteps use the predictor/corrector
        scheme, which should carry the value for the field'''
    E_0                     = np.zeros((size, 3))
    E_0[src_loc, src_dim]   = E0 * np.sin(2 * pi * f0 * (qq - 0.5) * dt)
    return E_0

def E_static():
    k = 1
    x = np.arange(0, xmax, dx)

    E_0                    = np.zeros((size, 3))
    E_0[1:size-1, src_dim] = E0 * np.sin(2 * pi * k * x / xmax)
    
    E_0[0]      = E_0[size - 2]
    E_0[size-1] = E_0[1]
    return E_0

def push_E(B, V_i, n_i, dt, qq): # Based off big F(B, n, V) eqn on pg. 140 (eqn. 10)
    
    global J
    
    E_out = np.zeros((size, 3))     # Output array - new electric field
    JxB   = np.zeros((size, 3))     # V cross B holder
    BdB   = np.zeros((size, 3))     # B cross del cross B holder     
    del_p = np.zeros((size, 3))     # Electron pressure tensor gradient array
    J     = np.zeros((size, 3))     # Ion current
    qn    = np.zeros(size, dtype=float)     # Ion charge density

    Te = [Te0 for ii in range(size)]
        
    # Calculate change/current summations over each species
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
        del_p[mm, 0] = ((qn[mm + 1] - qn[mm - 1]) / (2*dx*q)) * kB * Te[mm]     # This assumes constant Te. Recode for non-constant
        del_p[mm, 1] = 0
        del_p[mm, 2] = 0
    
    # Final Calculation
    E_out[:, 0] = (- JxB[:, 0] - (del_p[:, 0] ) - (BdB[:, 0] / (mu0))) / (qn[:])
    E_out[:, 1] = (- JxB[:, 1] - (del_p[:, 1] ) - (BdB[:, 1] / (mu0))) / (qn[:])
    E_out[:, 2] = (- JxB[:, 2] - (del_p[:, 2] ) - (BdB[:, 2] / (mu0))) / (qn[:])
    
    # Update ghost cells
    E_out[0, :]        = E_out[size - 2, :]
    E_out[size - 1, :] = E_out[1, :]
    return E_out


def collect_density(I_in, W_in, ptype): 
    '''Function to collect number density in each cell in each cell
    at each timestep. These values are weighted by their distance
    from cell nodes on each side.'''
    n_i = np.zeros((size, Nj), float)
   
    # Collect number density of all particles
    for ii in range(N):
        I   = int(I_in[ii])
        W   = W_in[ii]
        idx = int(ptype[ii])
        
        n_i[I,     idx] += (1 - W) * partout[3, idx]
        n_i[I + 1, idx] +=      W  * partout[3, idx]

    # Move ghost cell contributions - Ghost cells at 0 and size - 2
    n_i[size - 2, :] += n_i[0, :]                 # Move contributions
    n_i[1, :]        += n_i[size - 1, :]          # Move contributions
    n_i[0, :]         = n_i[size - 2, :]          # Fill ghost cell  
    n_i[size - 1, :]  = n_i[1, :]                 # Fill ghost cell
  
    n_i /= float(dx)        # Divide by cell dimensions to give densities per cubic metre
    
    # Smooth density using Gaussian smoother (1/4, 1/2, 1/4)
    #for jj in range(Nj):
    #    smoothed   = smooth(n_i[:, jj])
    #    n_i[:, jj] = smoothed
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
    

if __name__ == '__main__':
    # Metadata
    start_time     = timer()                       # Start Timer
    drive          = '/media/yoshi/VERBATIM HD/'   # Drive letter for portable HDD (changes between computers) - INCLUDE COLON. For UNIX, put home path here
    save_path      = 'runs/jenkins/'               # Save path on 'drive' HDD - each run then saved in numerically sequential subfolder with images and associated data
    generate_data  = 0            ;  plt.ioff()    # Save data? Yes (1), No (0)
    generate_plots = 1  
    run_desc = '''1D hybrid code reproduction of PIC code parameters from Jenkins et al. (2013).'''
    
    print 'Initializing parameters...'
    set_constants()
    set_parameters()
    part, part_type, old_part = initialize_particles()
    B, E, Vi, dns, dns_old, W = initialize_fields()
    DT, maxtime, framegrab    = set_timestep(part)      
    check_CFL() 

    for qq in range(1000):
        if qq == 0:

            print 'Simulation starting...'

            W         = assign_weighting(part[0, :], part[6, :], 1)                       # Assign initial (E) weighting to particles
            dns       = collect_density(part[6, :], W, part[8, :])                        # Collect initial density   
            Vi        = collect_flow(part, dns, W)                                        # Collect initial current
            
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], 0)                                  # Initialize magnetic field (should be second?)
            E[:, 0:3] = push_E(B[:, 0:3], Vi, dns, 0, qq)                               # Initialize electric field
            part      = velocity_update(part, B[:, 0:3], E[:, 0:3], -0.5*DT, W)         # Retard velocity to N - 1/2 to prevent numerical instability
            real_time  = 0                                                               # Initialize plasma time
        else:
            # N + 1/2
#==============================================================================
#             # ----- Update timestep ----- #
#             maxtime, DT, framegrab, time_pwr = update_dt(maxtime, DT, framegrab, part[3, :])    # Reset timestep to keep Courant condition
#             time_history.append(time_pwr)                                                       # Record when the timestep decreases (records zero unless halving takes place)
#             
#             if sum(time_history) >= 12:
#                 sys.exit('Simulation terminated: Timestep halved too much')
#             
#             # --------------------------- #
#==============================================================================
            
            part      = velocity_update(part, B[:, 0:3], E[:, 0:3], DT, W)                 # Advance Velocity to N + 1/2
            part, W   = position_update(part)                                               # Advance Position to N + 1
            B[:, 0:3] = push_B(B[:, 0:3], E[:, 0:3], DT)                                    # Advance Magnetic Field to N + 1/2
            
            dns       = 0.5 * (dns + collect_density(part[6, :], W, part[8, :]))            # Collect ion density at N + 1/2 : Collect N + 1 and average with N                                             
            Vi        = collect_flow(part, dns, W)                                          # Collect ion flow at N + 1/2
            E[:, 0:3] = E_static()                                                          # Store Electric Field at N because PC, yo
            E[:, 6:9] = E[:, 0:3]                                                           # Store Electric Field at N because PC, yo
            E[:, 0:3] = push_E(B[:, 0:3], Vi, dns, DT, qq) + E_ext(qq, DT)                  # Advance Electric Field to N + 1/2
            
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
            
# ----- Plot commands ----- #
        if generate_plots == 1:
            # Initialize Figure Space
            fig_size = 3, 4
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
            real_time  = qq * DT                        # Time within simulation (plasma time)
            x_pos      = part[0, 0:N] * 1000            # Particle x-positions (km) (For looking at particle characteristics)  
            x_cell_num = np.arange(size - 2)            # Numerical cell numbering: x-axis
      
        #----- Plot: Magnetic Field Components
            axbx = plt.subplot2grid(fig_size, (0, 0), colspan=2)
            axby = plt.subplot2grid(fig_size, (1, 0), colspan=2, sharex=axbx)
            axbz = plt.subplot2grid(fig_size, (2, 0), colspan=2, sharex=axbx)

            for ax, jj in zip([axbx, axby, axbz], range(3)):
                Bj = B[1:size-1, jj] / B0
                ax.plot(x_cell_num, Bj, color='g')
                ax.set_xlim(0, NX)
                ax.set_ylim(-1.5, 1.5)
                ax.set_ylabel(r'$B_{%d}$' % jj, labelpad=25, rotation=0, fontsize=14)

        #----- Plot: Electric Field Components
            axex = plt.subplot2grid(fig_size, (0, 2), colspan=2)
            axey = plt.subplot2grid(fig_size, (1, 2), colspan=2, sharex=axex)
            axez = plt.subplot2grid(fig_size, (2, 2), colspan=2, sharex=axex)
            
            for ax, jj in zip([axex, axey, axez], range(3)):
                Ej = E[1:size-1, jj]
                ax.plot(x_cell_num, Ej, color='magenta')
                ax.set_xlim(0, NX)
                ax.set_ylim(-1, 1)
                ax.set_ylabel(r'$E_%d$' % jj, labelpad=25, rotation=0, fontsize=14)

                    
            for ax in [axex, axey, axbx, axby]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_yticks(ax.get_yticks()[1:])               # Hide the bottom ticks of top plots

            #for ax in [ax_den, ax_Ez, ax_By, ax_B]:
                #ax.yaxis.tick_right()
                #ax.set_label_position("right")
                
            # Last Minute plot adjustments
            plt.suptitle('f = %dHz, dt = %es, E0 = %d' % (f0, DT, E0))
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0) 
            

        if qq%framegrab == 0:       # Dump data at specified interval   
        
            r = qq / framegrab          # Capture number
        
            # Initialize run directory
            if ((generate_plots == 1 or generate_data == 1) and (qq == 0)) == True:

                if os.path.exists('%s%s' % (drive, save_path)) == False:
                    os.makedirs('%s%s' % (drive, save_path))              # Create master test series directory
                    print 'Master directory created'
                    
                num  = len(os.listdir('%s%s' % (drive, save_path)))        # Count number of existing runs. Set to run number manually for static save

                path = ('%s%s/run_%d' % (drive, save_path, num))          # Set root run path (for images)
                
                
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
                
                d_path = ('%s%s/run_%d/Data' % (drive, save_path, num))   # Set path for data                
                
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
