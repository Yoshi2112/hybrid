# -*- coding: utf-8 -*-
'''
Diagnostic code that contains a stripped down version of the functions in the
hybrid code responsible for pushing the particles

velocity_update() :: Updates the velocity of a single particle using the Boris-
                    Buneman algorithm
position_update() :: Updates the position of a single particle using a simple
                    x_new = x_old + v*dt
eval_B0_particle():: Returns the magnetic field of a particle at its location,
                    analytically solved.
run_instance()   :: Helper function to initialise variable arrays, set initial 
                   conditions, record output values, and contains main loop
call_and_plot()  :: Calls run_instance() to perform a particle simulation and
                   retrieve output values. Mostly just plotting scripts.
                    
Simulation variables imported from the simulation_parameters_1D script
'''
import pdb
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

######################################
### CONSTANTS AND GLOBAL VARIABLES ###
######################################
ne         = 200e6                             # Electron density in /m3
B_eq       = 200e-9                            # Initial magnetic field at equator (x=0), in T
B_xmax     = 423.5e-9                          # Magnetic field intensity at boundary, in T

c          = 2.998925e+08                      # Speed of light (m/s)
q          = 1.602177e-19                      # Elementary charge (C)
mp         = 1.672622e-27                      # Mass of proton (kg)
e0         = 8.854188e-12                      # Epsilon naught - permittivity of free space
mu0        = (4e-7) * np.pi                    # Magnetic Permeability of Free Space (SI units)
wpi        = np.sqrt(ne * q ** 2 / (mp * e0))  # Proton   Plasma Frequency, wpi (rad/s)
va         = B_eq / np.sqrt(mu0*ne*mp)         # Alfven speed at equator: Assuming pure proton plasma

NX         = 512                               # Number of "cells". Sets simulation length as multiples of dx
dx         = 1.0 * c / wpi                     # Spatial cadence, based on ion inertial length
xmax       = NX // 2 * dx                      # Maximum simulation length, +/-ve on each side
xmin       =-NX // 2 * dx

a          = (B_xmax / B_eq - 1) / xmax ** 2   # Parabolic scale factor: Fitted to B_eq, B_xmax
loss_cone  = np.arcsin(np.sqrt(B_eq / B_xmax))*180 / np.pi


@nb.njit()
def eval_B0_particle(x, v):
    '''
    Calculates the B0 magnetic field at the position of a particle. Neglects B0_r
    in the calculation of local cyclotron frequency. B0_r is evaluated at the particle
    gyroradius (Larmor radius), which is calculated via the perp velocity and 
    local cyclotron frequency.
    
    Remember to add whistler field back to it
    
    The function this returns is smoother and less variable in rL, but the magnetic
    field returned is less accurate and leads to a periodic lessening of mu and 
    eventual flattening of the pitch angle towards zero.
    '''
    B0_xp    = np.zeros(3)
    B0_xp[0] = B_eq * (1 + a * x[0]**2)    

    l_cyc    = q * B0_xp[0] / mp                    # Neglects B0r component, since << B0x
    v_perp   = np.sqrt(v[1] ** 2 + v[2] ** 2)       # Perpendicular velocity (not required)
    rL       = v_perp / l_cyc                       # Larmor radius (not required, just returned)
    fac      = a * B_eq * x[0] / l_cyc              # Part of the multiplication for B0r
    
    B0_xp[1] =  v[2] * fac                          # Y-component of B0r, varying with vz
    B0_xp[2] = -v[1] * fac                          # Z-component of B0r, varying with vy (reverse direction)
    return B0_xp, rL


@nb.njit()
def eval_B0_particle_exact(x):
    '''
    Calculates the B0 magnetic field at the position of a particle. Neglects B0_r
    in the calculation of local cyclotron frequency. B0_r is evaluated at the particle
    gyroradius (Larmor radius), which is calculated via the perp velocity and 
    local cyclotron frequency.
    '''
    B0_xp    = np.zeros(3)
    B0_xp[0] = B_eq * (1 + a * x[0]**2)    

    gyroangle= np.arctan2(x[2], x[1])
    rL       = np.sqrt(x[1]**2 + x[2]**2)
    B0_r     = - a * B_eq * x[0] * rL
    
    B0_xp[1] = B0_r * np.cos(gyroangle)
    B0_xp[2] = B0_r * np.sin(gyroangle)
    return B0_xp, rL


#@nb.njit()
def velocity_update(pos, vel, dt):
    '''
    Updates N particle velocity via the Boris-Buneman algorithm.
    Based on Birdsall & Langdon (1985), pp. 59-63 (Sections 4.3, 4.4)

    INPUT:
        pos   -- Position array to be updated in place, in meters.           Shape (N,)  ndarray
        vel   -- Velocity array in m/s. Only x-component (index 0) accessed. Shape (3,N) ndarray
        dt    -- Temporal cadence of simulation in seconds.                  float64
        
    Notes: Still not sure how to parallelise this: There are a lot of array operations
    Probably need to do it more algebraically? Find some way to not create those temp arrays,
    since array creation (memory allocation) is mega resource intensive.
    '''
    B_particles = np.zeros((3, vel.shape[1]))
    r_lamor     = np.zeros((vel.shape[1]))
    for ii in nb.prange(vel.shape[1]):  
        qmi = 0.5 * dt * q / mp                                 
        
        Ep         = np.zeros(3)
        v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push (Eq. 4.3-7)
        
        # B0 and Larmor radius at particle location
        if ii == 0:
            B_particles[:, ii], r_lamor[ii] = eval_B0_particle(pos[:, ii], v_minus)
        else:
            B_particles[:, ii], r_lamor[ii] = eval_B0_particle_exact(pos[:, ii])

        T = qmi * B_particles[:, ii]                                        # Vector Boris variable (Eq. 4.4-11)
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable (Eq. 4.4-13)
        
        v_prime    = np.zeros(3, dtype=np.float64)
        v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation (Eq. 4.4-10)
        v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
        v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                
        v_plus     = np.zeros(3, dtype=np.float64)
        v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]     # (Eq. 4.4-12)
        v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
        v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
        
        vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push (Eq. 4.3-8)
    return B_particles, r_lamor


#@nb.njit()
def position_update(pos, vel, dt):
    '''
    Updates the position of the N particles using x_new = x_old + vx*t. 

    INPUT:
        pos   -- Position array to be updated in place, in meters.           Shape (N,)  ndarray
        vel   -- Velocity array in m/s. Only x-component (index 0) accessed. Shape (3,N) ndarray
        dt    -- Temporal cadence of simulation in seconds.                  float64

    Reflective boundaries to simulate the "open ends" that would have flux coming in from the ionosphere side.
    Could potentially merge this function with the velocity update function, but would have to include an
    if call so that the position isn't retarded when the velocity is (t = 0 -> t = -1/2)
    '''
    for ii in nb.prange(pos.shape[1]):
        pos[0, ii] += vel[0, ii] * dt               # Update position
        pos[1, ii] += vel[1, ii] * dt
        pos[2, ii] += vel[2, ii] * dt

        if (pos[0, ii] <= xmin or pos[0, ii] >= xmax):  # If simulation boundary reached
            vel[0, ii] *= -1.                           # Reflect velocity
            pos[0, ii] += vel[0, ii] * dt               # Get particle back in simulation space
    return


#@nb.njit()
def run_instance(max_rev=1000, v_mag=1.0, pitch=45.0, DT_mult=1.0, gyrofraction=0.1):
    '''
    Contains full particle pusher to simulate the motion of a single particle
    in the analytic magnetic bottle. No wave magnetic/electric
    fields present.
    '''
    # Set initial position and velocity based on pitch angle and particle energy
    pos       = np.zeros((3, 2), dtype=np.float64)
    vel       = np.zeros((3, 2), dtype=np.float64)

    vel[0, :] = v_mag * va * np.cos(pitch * np.pi / 180.)
    vel[1, :] = v_mag * va * np.sin(pitch * np.pi / 180.)

    # Assumes particle starts at equator (x = 0) with v_perp = vy
    rL        = mp * vel[1, 0] / (q * B_eq)
    pos[2, :] = rL
    
    # Initial quantities (save for export)
    init_pos = pos.copy() 
    init_vel = vel.copy()

    # Set timestep: Gyromotion resolver or particle vx limit
    gyroperiod = (2 * np.pi * mp) / (q * B_xmax)
    ion_ts     = gyrofraction * gyroperiod
    
    if vel[0, :].max() != 0.:
        vel_ts   = 0.6 * dx / vel[0, :].max()
    else:
        vel_ts = ion_ts
    
    if ion_ts < vel_ts:
        DT = ion_ts
    else:
        DT = vel_ts
    
    DT *= DT_mult       # Adjusts timestep manually
    
    # Set output arrays and max time/timesteps
    max_t    = max_rev * gyroperiod
    max_inc  = int(max_t / DT) + 1
    
    time        = np.zeros((max_inc),       dtype=np.float64)
    rad_history = np.zeros((max_inc, 2),    dtype=np.float64)
    pos_history = np.zeros((max_inc, 3, 2), dtype=np.float64)
    vel_history = np.zeros((max_inc, 3, 2), dtype=np.float64)
    mag_history = np.zeros((max_inc, 3, 2), dtype=np.float64)

    # Retard velocity by half a timestep
    Bp, rad = velocity_update(pos, vel, -0.5*DT)

    # Record initial values
    time[       0]         = 0.                       # t = 0
    rad_history[0, :]      = rad[:]                   # t = -0.5
    pos_history[0, :, :]   = pos[:, :]                # t = 0
    vel_history[0, :, :]   = vel[:, :]                # t = -0.5
    mag_history[0, :, :]   = Bp[ :, :]                # t = -0.5 (since it is evaluated only during velocity push)

    tt = 0; t_total = 0
    while tt < max_inc - 1:
        # Increment so first loop is at t = 1*DT
        tt      += 1
        t_total += DT
        
        # Update values: Velocity first, then position
        Bp, rad = velocity_update(pos, vel, DT)
        position_update(pos, vel, DT)
        
        time[       tt]       = t_total
        rad_history[tt, :]    = rad[:]                   # t = -0.5
        pos_history[tt, :, :] = pos[:, :]
        vel_history[tt, :, :] = vel[:, :]
        mag_history[tt, :, :] = Bp[ :, :]

    return init_pos, init_vel, time, pos_history, rad_history, vel_history, mag_history, DT, max_t


def call_and_plot():
    '''
    Diagnostic code to call the particle pushing part of the hybrid and check
    that its solving ok. Runs with zero background E field and B field defined
    by the constant background field specified in the parameter script.
    
    Note that "gyroperiods" are defined using the boundary magnetic field and not
    the equatorial one, since we want the shortest gyroperiod (i.e. highest field)
    to be resolved. Hence gymotion will be slower at the equator.
    
    Thoughts:
        - Why does 3D method have unstable rL but better mu conservation?
        - Periodicity of 3D rL - does it match bounce period?
        - Is it exactly conserved in uniform 3D field?
        - Analytic solution :: None simple
        - Adiabatic parameter? 0.05
    
    Ideas:
        - Plot rL from B0 with
    '''
    max_rev = 100          # Total simulation length in gyroperiods (bounce period ~761: 117.81s for v_mag = 1, pitch=45)
    v_mag   = 20.0         # Particle velocity magnitude
    pitch   = 44.0         # Particle pitch angle
    
    # Generate figure axes for the multi-DT plots
    dt_mults = [1.0, 0.5, 0.25, 0.1, 0.01]
    dt_color = ['k', 'b', 'r', 'g', 'm']
    Nt       = len(dt_mults)
    fig, ax  = plt.subplots(1, sharex=True)

    for dt_mult, tt in zip(dt_mults, np.arange(Nt)):
        # Call main function (split into two lines because its a long boi)
        init_pos, init_vel, time, pos_history, rad_history,  \
        vel_history, mag_history, DT, max_t      = run_instance(max_rev=max_rev, v_mag=v_mag, pitch=pitch, DT_mult=dt_mult)
        
        # Calculate a bunch of parameters that'll probably be useful in multiple plots
        init_vperp = np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
        init_vpara = init_vel[0]
        init_KE    = 0.5 * mp * init_vel ** 2
        init_pitch = np.arctan(init_vperp / init_vpara) * 180. / np.pi
        init_mu    = 0.5 * mp * init_vperp ** 2 / B_eq
        
        B_av                = 0.5 * (B_eq + B_xmax)
        adiabatic_parameter = mp * init_vperp / (q * B_av * xmax)
        
        vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
        vel_para      = vel_history[:, 0]
        vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
        vel_magnitude = np.sqrt(vel_history[:, 0] ** 2 + vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
        
        B_para      = mag_history[:, 0]
        B_perp      = np.sqrt(mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
        B_magnitude = np.sqrt(mag_history[:, 0] ** 2 + mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
        
        KE_perp = 0.5 * mp * (vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
        KE_para = 0.5 * mp *  vel_history[:, 0] ** 2
        KE_tot  = KE_para + KE_perp
        
        # Calculate first adiabatic invariant (magnetic moment, mu)
        mu         = KE_perp / B_magnitude
        mu_percent = (mu.max() - mu.min()) / init_mu * 100.
    
        # Radial quantities: Could come pos_radial to rad_history (directly evaluated Larmor)
        pos_radial = np.sqrt(pos_history[:, 1] ** 2 + pos_history[:, 2] ** 2)
        mag_radial = np.sqrt(mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
    
        #############################################################################
        ## Each one of these sections produces a plot. Turn them on/off by setting ##
        ## as either True/False                                                    ##
        #############################################################################
        if False:
            # Directly compares quantities for two separate runs (exact vs. approx B0r)
            # Mu, velocity, position, magnetic field
            fig, axes = plt.subplots(4, sharex=True)
            
            for ii, lbl, ls in zip([0, 1], ['1D', '3D'], ['-', ':']):
                axes[0].plot(time, mu[:, ii]*1e10,        label='$\mu$ %s'         % lbl, c='k', ls=ls)
                axes[1].plot(time, vel_perp[:, ii]*1e-3,  label='$v_\perp$ %s'     % lbl, c='b', ls=ls)
                axes[1].plot(time, vel_para[:, ii]*1e-3,  label='$v_\parallel$ %s' % lbl, c='r', ls=ls)
                
                for jj, comp, clr in zip([0, 1, 2], ['x', 'y', 'z'], ['k', 'b', 'r']):
                    axes[2].plot(time, pos_history[:, jj, ii]*1e-3, c=clr, label='%s %s'       % (comp, lbl), ls=ls)
                    axes[3].plot(time, mag_history[:, jj, ii]*1e9,  c=clr, label='$B_{0%s} %s$'% (comp, lbl), ls=ls)
    
            for ax in axes:
                ax.legend(loc='lower right')
    
            axes[2].axhline(0, c='k', ls=':')
            axes[2].axhline(xmin*1e-3, c='k', ls=':')
            axes[2].axhline(xmax*1e-3, c='k', ls=':')
    
            axes[0].set_title(r'$\mu$ and parameters for trapped particle :: Exact (3D) vs. Approx (1D) soln :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
            axes[0].set_ylabel('$\mu \times 10^{10}$', rotation=0, labelpad=30)
            axes[1].set_ylabel('Velocity\n(km/s)', rotation=0, labelpad=20)
            axes[2].set_ylabel('Position\n(km)', rotation=0, labelpad=20)
            axes[3].set_ylabel('$B$\n(nT)', rotation=0, labelpad=20)
            axes[3].set_xlabel('Time (s)')
            axes[3].set_xlim(0, time[-1])
            
            fig.subplots_adjust(hspace=0)
            
            
        if False:
            ## Same as above, but differences between each ##
            fig, axes = plt.subplots(4, sharex=True)
            
            axes[0].plot(time, (mu[:, 1]       - mu[:, 0])*1e10,        c='k', label='$\mu$')
            axes[1].plot(time, (vel_perp[:, 1] - vel_perp[:, 0])*1e-3,  c='b', label='$v_\perp$'    )
            axes[1].plot(time, (vel_para[:, 1] - vel_para[:, 0])*1e-3,  c='r', label='$v_\parallel$')
            
            for jj, comp, clr in zip([0, 1, 2], ['x', 'y', 'z'], ['k', 'b', 'r']):
                axes[2].plot(time, (pos_history[:, jj, 1] - pos_history[:, jj, 0])*1e-3, c=clr, label='%s' % comp)
                axes[3].plot(time, (mag_history[:, jj, 1] - mag_history[:, jj, 0])*1e9,  c=clr, label='$B_{0%s}$' % comp)
            
            for ax in axes:
                ax.legend(loc='lower right')
    
            axes[2].axhline(0, c='k', ls=':')
            axes[2].axhline(xmin*1e-3, c='k', ls=':')
            axes[2].axhline(xmax*1e-3, c='k', ls=':')
    
            axes[0].set_title(r'Exact (3D) vs. Approx (1D) parameters - Differences :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
            axes[0].set_ylabel('$\mu (\\times 10^{10})$', rotation=0, labelpad=30)
            axes[1].set_ylabel('Velocity\n(km/s)', rotation=0, labelpad=20)
            axes[2].set_ylabel('Position\n(km)', rotation=0, labelpad=20)
            axes[3].set_ylabel('$B$\n(nT)', rotation=0, labelpad=20)
            axes[3].set_xlabel('Time (s)')
            axes[3].set_xlim(0, time[-1])
            
            fig.subplots_adjust(hspace=0)
            
            
        if False:
            # Direct comparison, but with radial components instead of y,z
            # Mu, velocity, position, magnetic field
            lw = 0.5
            fig, axes = plt.subplots(4, sharex=True)
            
            for ii, lbl, ls in zip([0, 1], ['1D', '3D'], ['-', ':']):
                axes[0].plot(time, mu[:, ii]*1e10,        label='$\mu$ %s'         % lbl, c='k', ls=ls, lw=lw)
                axes[1].plot(time, vel_perp[:, ii]*1e-3,  label='$v_\perp$ %s'     % lbl, c='k', ls=ls, lw=lw)
                #axes[1].plot(time, vel_para[:, ii]*1e-3,  label='$v_\parallel$ %s' % lbl, c='r', ls=ls)
    
                axes[2].plot(time, pos_radial[:, ii]*1e-3, c='k', label='$r$ %s'     % lbl, ls=ls, lw=lw)
                axes[3].plot(time, mag_radial[:, ii]*1e9,  c='k', label='$B_{0r}$ %s'% lbl, ls=ls, lw=lw)
    
            for ax in axes:
                ax.legend(loc='lower right')
    
            axes[0].set_title(r'$\mu$ and parameters for trapped particle :: Exact (3D) vs. Approx (1D) soln :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
            axes[0].set_ylabel('$\mu \\times 10^{10}$', rotation=0, labelpad=30)
            axes[1].set_ylabel('Velocity\n(km/s)', rotation=0, labelpad=20)
            axes[2].set_ylabel('Position\n(km)', rotation=0, labelpad=20)
            axes[3].set_ylabel('$B$\n(nT)', rotation=0, labelpad=20)
            axes[3].set_xlabel('Time (s)')
            axes[3].set_xlim(0, time[-1])
            
            fig.subplots_adjust(hspace=0)
            
    
        if False:
            # Differences, but with radial components instead of y,z
            # Mu, velocity, position, magnetic field
            lw = 0.5
            fig, axes = plt.subplots(4, sharex=True)
            
            axes[0].plot(time, (mu[:, 1]         - mu[:, 0])        *1e10, label='$\mu$'    , c='k', lw=lw)
            axes[1].plot(time, (vel_perp[:, 1]   - vel_perp[:, 0])  *1e-3, label='$v_\perp$', c='k', lw=lw)
            axes[2].plot(time, (pos_radial[:, 1] - pos_radial[:, 0])*1e-3, label='$r$'      , c='k', lw=lw)
            axes[3].plot(time, (mag_radial[:, 1] - mag_radial[:, 0])*1e9,  label='$B_{0r}$' , c='k', lw=lw)
    
            for ax in axes:
                ax.legend(loc='lower right')
            
            axes[0].set_title(r'$\mu$ and parameters for trapped particle :: Exact (3D) vs. Approx (1D) soln :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
            axes[0].set_ylabel('$\mu \\times 10^{10}$', rotation=0, labelpad=30)
            axes[1].set_ylabel('Velocity\n(km/s)', rotation=0, labelpad=20)
            axes[2].set_ylabel('Position\n(km)', rotation=0, labelpad=20)
            axes[3].set_ylabel('$B$\n(nT)', rotation=0, labelpad=20)
            axes[3].set_xlabel('Time (s)')
            axes[3].set_xlim(0, time[-1])
            
            fig.subplots_adjust(hspace=0)
        
        
        if False:
            # Evaluation of Lamor radii
            #
            # Question: How accurately is the Larmor radius resolved for each one?
            # Second order accuracy via position method. Seems almost completely accurate
            # with rL approximation due to exact solution of v (Boris Method)
            # Four plots: Compare exact vs. approx solutions directly (1) and then difference (2)
            # Then compare the difference between each method and the particle position (3)
            # and the difference
            rL_diff     = (rad_history[:, 1] - rad_history[:, 0]) / rad_history[:, 1] * 100.
            rL_diff_pos = (pos_radial[ :, 1] - pos_radial[ :, 0]) / pos_radial[ :, 1] * 100.
            
            fig, axes = plt.subplots(4, sharex=True)
            
            axes[0].set_title(r'Lamor radius evaluation :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
            
            axes[0].plot(time, rad_history[:, 0], label='$r_L$ approx', lw=0.5, c='r', ls='-')
            axes[0].plot(time, rad_history[:, 1], label='$r_L$ exact' , lw=0.5, c='b', ls='-')
            axes[0].set_ylabel('$r_L$ from\n$B_0$ calc.', rotation=0, labelpad=30)
            axes[0].legend(loc='upper left')
            
            axes[1].plot(time, rL_diff , lw=0.5, c='k', ls='-')
            axes[1].set_ylabel('$r_L$ difference\n(%)', rotation=0, labelpad=30)
    
            axes[2].plot(time, pos_radial[:, 0], label='$r_L$ approx', lw=0.5, c='r', ls='-')
            axes[2].plot(time, pos_radial[:, 1], label='$r_L$ exact' , lw=0.5, c='b', ls='-')
            axes[2].set_ylabel('$r_L$ from\nposition', rotation=0, labelpad=30)
            axes[2].legend(loc='upper left')
            
            axes[3].plot(time, rL_diff_pos, lw=0.5, c='k', ls='-')
            axes[3].set_ylabel('$r_L$ difference\n(%)', rotation=0, labelpad=30)
    
            axes[3].set_xlabel('Time (s)')
            axes[3].set_xlim(0, time[-1])
        
        
        if False:
            # Plot 3D trajectory (This is just a check of the 1D approximation, since everything
            # can be more or less exact here. Yay for not needing gridpoints)
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            
            #ax.plot(pos_history[:, 0]*1e-3, pos_history[:, 1]*1e-3, pos_history[:, 2]*1e-3, marker='o')
            im = ax.scatter(pos_history[:, 0]*1e-3, pos_history[:, 1]*1e-3, pos_history[:, 2]*1e-3, c=time)
            plt.colorbar(im).set_label('Time (s)', rotation=0, labelpad=20)
            
            plt.title(r'Single Trapped Particle :: 3D Position (exact soln) :: Max $\delta \mu = $%6.4f%% :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (mu_percent, v_mag, pitch))
            
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_zlabel('z (km)')
            
            
        if False:
            ## Plots velocity/magnetic field timeseries                 ##
            ## comment out the bits you don't want to plot              ##
            ## vy,vz show cyclotron motion, v_perp shows bounce motion  ##
            fig, axes = plt.subplots(2, sharex=True)
            
            #axes[0].plot(time, vel_history[:, 1]* 1e-3, label='vy')
            #axes[0].plot(time, vel_history[:, 2]* 1e-3, label='vz')
            axes[0].plot(time, vel_perp         * 1e-3, label='v_perp')
            axes[0].plot(time, vel_para*1e-3, label='v_para')
            
            axes[0].set_ylabel('v (km)')
            axes[0].set_xlabel('t (s)')
            axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], loss_cone, init_pitch))
            axes[0].legend()
            
            axes[1].plot(time, B_magnitude,       label='|B0|')
            #axes[1].plot(time, mag_history[:, 0], label='B0x')
            #axes[1].plot(time, mag_history[:, 1], label='B0y')
            #axes[1].plot(time, mag_history[:, 2], label='B0z')
            axes[1].legend()
            axes[1].set_ylabel('t (s)')
            axes[1].set_ylabel('B (nT)')
            axes[1].set_xlim(0, time[-1])
                
        if False:
            ## Plot gyromotion of particle vx vs. vy ##
            ## Only really handles up to a few thousand gyroperiods, thanks matplotlib
            plt.title('Particle gyromotion: {} gyroperiods ({:.1f}s)'.format(max_rev, max_t))
            plt.scatter(vel_history[:, 1], vel_history[:, 2], c=time, s=1)
            plt.colorbar().set_label('Time (s)')
            plt.ylabel('vy (km/s)')
            plt.xlabel('vz (km/s)')
            plt.axis('equal')
            
        if False:
            from matplotlib.patches import Circle
            ## Plot gyromotion of particle y vs. z ##
            ## Only really handles up to a few thousand gyroperiods, thanks matplotlib
            #pt = 0  # Which particle
            
            for pt in [0, 1]:
                rL = np.sqrt(init_pos[1, pt] ** 2 + init_pos[2, pt] ** 2)
                
                rL_angle    = np.arctan2(pos_history[:, 2, pt], pos_history[:, 1, pt])
                
                plt.figure()
                plt.title('Particle gyromotion: {} gyroperiods ({:.1f}s)'.format(max_rev, max_t))
                plt.scatter(pos_history[:, 1, pt]*1e-3, pos_history[:, 2, pt]*1e-3, s=20, c='b', label='Position')
                plt.scatter(rad_history[:, pt]*1e-3*np.cos(rL_angle), 
                            rad_history[:, pt]*1e-3*np.sin(rL_angle), s=20, label='Larmor', c='r')
                plt.legend()
                #plt.colorbar().set_label('Time (s)')
                
                testCircle = Circle((0, 0), radius=rL*1e-3, color='k', ls=':', fill=False)
                plt.gca().add_artist(testCircle)
                
                plt.ylabel('y (km)')
                plt.xlabel('z (km)')
                plt.axis('equal')
                
        if False:
            ## Plot gyromotion of particle y vs. z ##
            ## Same as above, but timeseries
            
            for pt, lbl in zip([0, 1], ['1D', '3D']):
                rL = np.sqrt(init_pos[1, pt] ** 2 + init_pos[2, pt] ** 2) * 1e-3
    
                fig, ax = plt.subplots(sharex=True)
                ax.set_title('Particle gyromotion :: {} :: {} gyroperiods ({:.1f}s)'.format(lbl, max_rev, max_t))
                
                ax.plot(time,  pos_radial[:, pt]*1e-3, c='b', label='From Position')
                ax.plot(time, rad_history[:, pt]*1e-3, c='r', label='From B0 func.')
                ax.legend()
                
                ax.axhline(rL, ls=':', c='k', alpha=0.5)
                
                ax.set_ylabel('r (km)')
                ax.set_xlabel('Time (s)')
            
        if False:
            ## Plot parallel and perpendicular kinetic energies/velocities vs. time
            ## Looks primarily at conservation and totals
            fig, axes = plt.subplots(2, sharex=True)
            axes[0].set_title('Kinetic energy of single particle: Components and Conservation')
            axes[0].plot(time, KE_para/q, c='b', label=r'$KE_\parallel$')
            axes[0].plot(time, KE_perp/q, c='r', label=r'$KE_\perp$')
            axes[0].plot(time, KE_tot/q, c='k', label=r'$KE_{total}$')
            axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
            axes[0].get_yaxis().get_major_formatter().set_scientific(False)
            axes[0].set_ylabel('Energy\n(eV)', rotation=0, labelpad=20)
            axes[0].legend()
    
            percent_change = np.abs(KE_tot - init_KE.sum()) / init_KE.sum() * 100. 
            axes[1].set_title('Total change :: $\Delta KE = |KE_{init} - KE(t)|$')
            axes[1].plot(time, percent_change*1e12)
            axes[1].get_yaxis().get_major_formatter().set_useOffset(False)
            axes[1].get_yaxis().get_major_formatter().set_scientific(False)
            axes[1].set_ylabel('Percent change\n($\\times 10^{-12}$)')
            
            axes[1].set_xlim(0, time[-1])
            axes[1].set_xlabel('Time (s)')
         
            
        if False:
            # Visualises the KE drift by plotting the KE ratio and difference in space,
            # with the color representing the time
            fig, axes = plt.subplots(2, sharex=True)
            
            KE_ratio = KE_para / KE_perp  
            axes[0].set_title('Kinetic Energy ratio :: $KE_{\parallel} / KE_{\perp}$')
            im = axes[0].scatter(pos_history*1e-3, KE_ratio, c=time, s=2)
            axes[0].set_ylabel('$\\frac{KE_{\parallel}}{KE_{\perp}}$', fontsize=16, rotation=0, labelpad=20)
            axes[0].set_ylim(0.975, 1.025)
            
            KE_difference = (KE_para - KE_perp)/q
            axes[1].set_title('Kinetic Energy difference :: $KE_{\parallel} - KE_{\perp}$')
    
            axes[1].scatter(pos_history*1e-3, KE_difference, c=time, s=2)
            axes[1].set_ylabel('Difference\n(eV)', rotation=0, labelpad=30)
            
            axes[1].set_xlabel('Position (m)')
            axes[1].set_xlim(-300, 300)
            
            fig.colorbar(im, ax=axes).set_label('Time (s)', rotation=0, labelpad=20)
            
        if False:
            # Plots vx, v_perp vs. x  
            # Tests if quantities are conserved vs. position in the bottle
            fig, ax = plt.subplots(1)
            ax.set_title(r'Velocity vs. Space: v0 = [%4.1f, %4.1f, %4.1f]$v_{A,eq}^{-1}$ : %d gyroperiods (%5.2fs)' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], max_rev, max_t))
            
            for idx, lbl, mk in zip([0, 1], ['1D', '3D'], ['x', 'o']):
                ax.scatter(pos_history*1e-3, vel_history[:, 0]*1e-3, c='b', label=r'$v_\parallel$')
                ax.scatter(pos_history*1e-3, vel_perp,               c='r', label=r'$v_\perp$')
            
            ax.set_xlabel('x (km)')
            ax.set_ylabel('v (km/s)')
            ax.set_xlim(xmin*1e-3, xmax*1e-3)
            ax.legend()
    
    
        if False:
            # Invariant and parameters vs. x
            # Tests if quantities are conserved vs. position in the bottle
            fig, axes = plt.subplots(3, sharex=True)
            axes[0].plot(pos_history*1e-3, mu*1e10)
            axes[0].set_title(r'First Invariant $\mu$ for single trapped particle, v0 = [%3.1f, %3.1f, %3.1f]$v_{A,eq}^{-1}$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg, $t_{max} = %5.0fs$' % (init_vel[0, 0]/va, init_vel[1, 0]/va, init_vel[2, 0]/va, loss_cone, init_pitch, max_t))
            axes[0].set_ylabel(r'$\mu (\times 10^{-10})$', rotation=0, labelpad=20)
            axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
            axes[0].axhline(init_mu*1e10, c='k', ls=':')
            
            axes[1].plot(pos_history*1e-3, KE_perp/q)
            axes[1].set_ylabel(r'$KE_\perp (eV)$', rotation=0, labelpad=20)
    
            axes[2].plot(pos_history*1e-3, B_magnitude*1e9)
            axes[2].set_ylabel(r'$|B|$ (nT)', rotation=0, labelpad=20)
            
            axes[2].set_xlabel('Position (km)')
            axes[2].set_xlim(xmin*1e-3, xmax*1e-3)
    
    
        if True:
            # Plot for each DT, just the mu. In the loop so the figure is slowly generated.
            for ii, lbl, ls in zip([0, 1], ['1D', '3D'], ['-', ':']):
                ax.plot(time, mu[:, ii]*1e10, label='$\mu$ %s DT $\\times$ %.2f' % (lbl, dt_mult), c=dt_color[tt], ls=ls)
            ax.legend(loc='lower right')
            ax.set_ylabel('$\mu \\times 10^{10}$', rotation=0, labelpad=40)
    
    ax.set_title(r'$\mu$ and parameters for trapped particle :: Exact (3D) vs. Approx (1D) soln :: |v| = %4.1f$v_A$ :: $\alpha$ = %4.1f' % (v_mag, pitch))
    
    ax.set_xlabel('Time (s)')
    ax.set_xlim(0, time[-1])
            
            
    
    
    # Maximizes plot window, but only if you've plotted something
    # (prevents random empty plot windows from opening)
    if plt.fignum_exists(1):
        #fig.align_ylabels()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    else:
        print('No plot routine selected, no figure generated.')
    return


if __name__ == '__main__':
    call_and_plot()