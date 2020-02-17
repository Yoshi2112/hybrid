# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:17:49 2020

@author: Yoshi
"""
import pdb
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from simulation_parameters_1D import B_eq, B_xmax, a, q, mp, xmin, xmax, dx, va, loss_cone

'''
Diagnostic code that contains a stripped down version of the functions in the
hybrid code responsible for pushing the particles

velocity_update() :: Updates the velocity of a single particle using the Boris-
                    Buneman algorithm
position_update() :: Updates the position of a single particle using a simple
                    x_new = x_old + v*dt
eval_B0_particle():: Returns the magnetic field of a particle at its location,
                    analytically solved.
                    
Simulation variables imported from the simulation_parameters_1D script
'''
@nb.njit()
def eval_B0_particle(x, v):
    B0_xp    = np.zeros(3)
    B0_xp[0] = B_eq * (1 + a * x**2)    
    
    l_cyc    = q * B0_xp[0] / mp
    fac      = a * B_eq * x / l_cyc
    
    B0_xp[1] =  v[2] * fac
    B0_xp[2] = -v[1] * fac
    return B0_xp


@nb.njit()
def velocity_update(pos, vel, dt):

    for ii in nb.prange(vel.shape[1]):  
        qmi = 0.5 * dt * q / mp                                 
        
        Ep         = np.zeros(3)
        v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push
        Bp         = eval_B0_particle(pos[ii], v_minus)                     # B0 at particle location
        
        T = qmi * Bp                                                        # Vector Boris variable
        S = 2.*T / (1. + T[0] ** 2 + T[1] ** 2 + T[2] ** 2)                 # Vector Boris variable
        
        v_prime    = np.zeros(3)
        v_prime[0] = v_minus[0] + v_minus[1] * T[2] - v_minus[2] * T[1]     # Magnetic field rotation
        v_prime[1] = v_minus[1] + v_minus[2] * T[0] - v_minus[0] * T[2]
        v_prime[2] = v_minus[2] + v_minus[0] * T[1] - v_minus[1] * T[0]
                
        v_plus     = np.zeros(3)
        v_plus[0]  = v_minus[0] + v_prime[1] * S[2] - v_prime[2] * S[1]
        v_plus[1]  = v_minus[1] + v_prime[2] * S[0] - v_prime[0] * S[2]
        v_plus[2]  = v_minus[2] + v_prime[0] * S[1] - v_prime[1] * S[0]
        
        vel[:, ii] = v_plus +  qmi * Ep                                     # Second E-field half-push
    return Bp


@nb.njit()
def position_update(pos, vel, dt):
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[0, ii] * dt

        if (pos[ii] <= xmin or pos[ii] >= xmax):
            vel[0, ii] *= -1.                   # Reflect velocity
            pos[ii]    += vel[0, ii] * dt       # Get particle back in simulation space
    return


@nb.njit()
def do_particle_run(max_rev=1000):
    '''
    Contains full particle pusher including timestep checker to simulate the motion
    of a single particle in the analytic magnetic field field. No wave magnetic/electric
    fields present.
    '''
    orbit_res = 0.1
    pos       = np.array([0.])
    vel       = np.array([0.5, 1.0, 0.0]).reshape((3, 1))*va

    # Initial quantities
    init_pos = pos.copy() 
    init_vel = vel.copy()
    
    # Set timestep
    gyroperiod = (2 * np.pi * mp) / (q * B_xmax)
    ion_ts     = orbit_res * gyroperiod
    
    if vel[0, :].max() != 0.:
        vel_ts   = 0.6 * dx / vel[0, :].max()
    else:
        vel_ts = ion_ts
    
    if ion_ts < vel_ts:
        DT = ion_ts
    else:
        DT = vel_ts
    
    # Set output arrays and max time/timesteps
    max_t    = max_rev * gyroperiod
    max_inc  = int(max_t / DT) + 1
    
    time        = np.zeros((max_inc))
    pos_history = np.zeros((max_inc))
    vel_history = np.zeros((max_inc, 3))
    mag_history = np.zeros((max_inc, 3))

    # Retard velocity by half a timestep
    Bp = velocity_update(pos, vel, -0.5*DT)

    # Record initial values
    time[       0]   = 0.                       # t = 0
    pos_history[0]   = pos[0]                   # t = 0
    vel_history[0]   = vel[:, 0]                # t = -0.5
    mag_history[0]   = Bp                       # t = -0.5 (since it is evaluated only during velocity push)

    tt = 0; t_total = 0
    while tt < max_inc - 1:
        # Increment so first loop is at t = 1*DT
        tt      += 1
        t_total += DT
        
        # Update values: Velocity first, then position
        Bp = velocity_update(pos, vel, DT)
        position_update(pos, vel, DT)
        
        time[         tt] = t_total
        pos_history[  tt] = pos[0]
        vel_history[  tt] = vel[:, 0]
        mag_history[  tt] = Bp

    return init_pos, init_vel, time, pos_history, vel_history, mag_history, DT, max_t


def test_mirror_motion():
    '''
    Diagnostic code to call the particle pushing part of the hybrid and check
    that its solving ok. Runs with zero background E field and B field defined
    by the constant background field specified in the parameter script.
    '''
    max_rev = 50000
    init_pos, init_vel, time, pos_history, vel_history, mag_history, DT, max_t = do_particle_run(max_rev=max_rev)

    # Calculate parameter timeseries using recorded values
    init_vperp  = np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    init_vpara = init_vel[0]
    init_KE    = 0.5 * mp * init_vel ** 2
    init_pitch = np.arctan(init_vperp / init_vpara) * 180. / np.pi
    init_mu    = 0.5 * mp * init_vperp ** 2 / B_eq
    
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
    
    mu         = KE_perp / B_magnitude
    mu_percent = (mu.max() - mu.min()) / init_mu * 100.
    
    if True:
        # Basic mu plot with v_perp, |B| also plotted
        fig, axes = plt.subplots(4, sharex=True)
        
        axes[0].plot(time, mu*1e10, label='$\mu(t_v)$', lw=0.5, c='k')
        
        axes[0].set_title(r'First Invariant $\mu$ for single trapped particle :: DT = %7.5fs :: Max $\delta \mu = $%6.4f%%' % (DT, mu_percent))
        axes[0].set_ylabel(r'$(\times 10^{-10})$', rotation=0, labelpad=30)
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[0].axhline(init_mu*1e10, c='k', ls=':')
        
        axes[1].plot(time, vel_perp*1e-3, lw=0.5, c='k')
        axes[1].set_ylabel('$v_\perp$\n(km/s)', rotation=0, labelpad=20)
        
        axes[2].plot(time, pos_history*1e-3, lw=0.5, c='k')
        axes[2].set_ylabel('$x$\n(km)', rotation=0, labelpad=20)

        axes[3].plot(time, B_magnitude*1e9, lw=0.5, c='k')
        axes[3].set_ylabel('$|B|(t_v)$ (nT)', rotation=0, labelpad=20)

        axes[3].set_xlabel('Time (s)')
        axes[3].set_xlim(0, time[-1])
        
        
    if False:
        ## Plots velocity/mag timeseries ##
        fig, axes = plt.subplots(2, sharex=True)
        
        axes[0].plot(time, vel_history[:, 1]* 1e-3, label='vy')
        axes[0].plot(time, vel_history[:, 2]* 1e-3, label='vz')
        axes[0].plot(time, vel_perp         * 1e-3, label='v_perp')
        axes[0].plot(time, vel_para*1e-3, label='v_para')
        
        axes[0].set_ylabel('v (km)')
        axes[0].set_xlabel('t (s)')
        axes[0].set_title(r'Velocity/Magnetic Field at Particle, v0 = [%4.1f, %4.1f, %4.1f]km/s, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], loss_cone, init_pitch))
        #axes[0].set_xlim(0, None)
        axes[0].legend()
        
        axes[1].plot(time, B_magnitude,       label='|B0|')
        #axes[1].plot(time, mag_history[:, 0], label='B0x')
        #axes[1].plot(time, mag_history[:, 1], label='B0y')
        #axes[1].plot(time, mag_history[:, 2], label='B0z')
        axes[1].legend()
        axes[1].set_ylabel('t (s)')
        axes[1].set_ylabel('B (nT)')
        axes[1].set_xlim(0, None)
        
        
    if False:
        ## Plots position/velocity component timeseries ##
        fig, axes = plt.subplots(3, sharex=True)
        axes[0].set_title(r'Position/Velocity of Particle : v0 = [%3.1f, %3.1f, %3.1f]$v_A$, $\alpha_L$=%4.1f deg, $\alpha_{p,eq}$=%4.1f deg' % (init_vel[0, 0]/va, init_vel[1, 0]/va, init_vel[2, 0]/va, loss_cone, init_pitch))

        axes[0].plot(time, pos_history*1e-3)
        axes[0].set_xlabel('t (s)')
        axes[0].set_ylabel(r'x (km)')
        axes[0].axhline(xmin*1e-3, color='k', ls=':')
        axes[0].axhline(xmax*1e-3, color='k', ls=':')
        axes[0].legend()
        
        axes[1].plot(time, vel_history[:, 0]/va, label='$v_\parallel$')
        axes[1].plot(time,            vel_perp/va, label='$v_\perp$')
        axes[1].set_xlabel('t (s)')
        axes[1].set_ylabel(r'$v_\parallel$ ($v_{A,eq}^{-1}$)')
        axes[1].legend()
        
        axes[2].plot(time, vel_history[:, 1]/va, label='vy')
        axes[2].plot(time, vel_history[:, 2]/va, label='vz')
        axes[2].set_xlabel('t (s)')
        axes[2].set_ylabel(r'$v_\perp$ ($v_{A,eq}^{-1}$)')
        axes[2].legend()
        
        for ax in axes:
            ax.set_xlim(0, 100)
            
    if False:
        # Plot gyromotion of particle vx vs. vy
        plt.title('Particle gyromotion: {} gyroperiods ({:.1f}s)'.format(max_rev, max_t))
        plt.scatter(vel_history[:, 1], vel_history[:, 2], c=time)
        plt.colorbar().set_label('Time (s)')
        plt.ylabel('vy (km/s)')
        plt.xlabel('vz (km/s)')
        plt.axis('equal')
        
    if False:
        ## Plot parallel and perpendicular kinetic energies/velocities
        plt.figure()
        plt.title('Kinetic energy of single particle: Full Bottle')
        plt.plot(time, KE_para/q, c='b', label=r'$KE_\parallel$')
        plt.plot(time, KE_perp/q, c='r', label=r'$KE_\perp$')
        plt.plot(time, KE_tot /q, c='k', label=r'$KE_{total}$')
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
        plt.ylabel('Energy (eV)')
        plt.xlabel('Time (s)')
        plt.legend()
    
    if False:           
        percent = abs(KE_tot - init_KE.sum()) / init_KE.sum() * 100. 

        plt.figure()
        plt.title('Total kinetic energy change')

        plt.plot(time, percent*1e12)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
        plt.xlim(0, time[-1])
        plt.ylabel(r'Percent change ($\times 10^{-12}$)')
        plt.xlabel('Time (s)')
        
    if False:
        # Plots vx, v_perp vs. x  
        fig, ax = plt.subplots(1)
        ax.set_title(r'Velocity vs. Space: v0 = [%4.1f, %4.1f, %4.1f]$v_{A,eq}^{-1}$ : %d gyroperiods (%5.2fs)' % (init_vel[0, 0], init_vel[1, 0], init_vel[2, 0], max_rev, max_t))
        ax.plot(pos_history*1e-3, vel_history[:, 0]*1e-3, c='b', label=r'$v_\parallel$')
        ax.plot(pos_history*1e-3, vel_perp,               c='r', label=r'$v_\perp$')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('v (km/s)')
        ax.set_xlim(xmin*1e-3, xmax*1e-3)
        ax.legend()

    if False:
        # Invariant and parameters vs. x
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
    return


if __name__ == '__main__':
    test_mirror_motion()