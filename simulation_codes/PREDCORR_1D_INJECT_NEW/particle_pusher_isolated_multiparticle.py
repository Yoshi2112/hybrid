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
def eval_B0_particle(z, v):
    B0_xp    = np.zeros(3)
    B0_xp[2] = B_eq * (1 + a * z**2)    
    
    l_cyc    = q * B0_xp[2] / mp
    fac      = a * B_eq * z / l_cyc
    rL       = np.sqrt(v[0] ** 2 + v[1] ** 2) / l_cyc
        
    B0_xp[0] =  v[1] * fac
    B0_xp[1] = -v[0] * fac
    return B0_xp, rL


@nb.njit()
def velocity_update(pos, vel, dt):
    Bps = np.zeros((3, pos.shape[0]))
    rL  = np.zeros(pos.shape[0])
    for ii in nb.prange(vel.shape[1]):  
        qmi = 0.5 * dt * q / mp                                 
        
        Ep         = np.zeros(3)
        v_minus    = vel[:, ii] + qmi * Ep                                  # First E-field half-push
        Bps[:, ii],rL[ii] = eval_B0_particle(pos[ii], v_minus)             # B0 at particle location
        
        T = qmi * Bps[:, ii]                                                # Vector Boris variable
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
    return rL, Bps


@nb.njit()
def position_update(pos, vel, dt):
    for ii in nb.prange(pos.shape[0]):
        pos[ii] += vel[2, ii] * dt

        if (pos[ii] <= xmin or pos[ii] >= xmax):
            vel[2, ii] *= -1.                   # Reflect velocity
            pos[ii]    += vel[2, ii] * dt       # Get particle back in simulation space
    return


#@nb.njit()
def do_particle_run(max_rev=1000):
    '''
    Contains full particle pusher including timestep checker to simulate the motion
    of a single particle in the analytic magnetic field field. No wave magnetic/electric
    fields present.
    '''
    global v_parallel
    
    Np         = 5
    v_parallel = np.sqrt(2) * va
    init_pitch = np.linspace(loss_cone + 0.5, 85, Np) * np.pi / 180.
    vel        = np.zeros((3, Np))

    # Initialize each particle with a different pitch angle
    for ii in range(Np):
        vel[2, ii] = v_parallel
        vel[1, ii] = v_parallel * np.tan(init_pitch[ii])

    pos = np.zeros(Np)
    
    # Initial quantities
    init_pos = pos.copy() 
    init_vel = vel.copy()
    
    # Set timestep
    orbit_res  = 0.1
    gyroperiod = (2 * np.pi * mp) / (q * B_xmax)
    ion_ts     = orbit_res * gyroperiod
    
    if vel[2, :].max() != 0.:
        vel_ts   = 0.6 * dx / vel[2, :].max()
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
    pos_history = np.zeros((max_inc, Np))
    rL_history  = np.zeros((max_inc, Np))
    vel_history = np.zeros((max_inc, 3, Np))
    mag_history = np.zeros((max_inc, 3, Np))

    # Retard velocity by half a timestep
    rL, Bp = velocity_update(pos, vel, -0.5*DT)

    # Record initial values
    time[       0]   = 0.                       # t = 0
    pos_history[0]   = pos                      # t = 0
    rL_history[ 0]   = rL                       # t = 0
    vel_history[0]   = vel                      # t = -0.5
    mag_history[0]   = Bp                       # t = -0.5 (since it is evaluated only during velocity push)

    tt = 0; t_total = 0
    while tt < max_inc - 1:
        # Increment so first loop is at t = 1*DT
        tt      += 1
        t_total += DT
        
        # Update values: Velocity first, then position
        rL, Bp = velocity_update(pos, vel, DT)
        position_update(pos, vel, DT)
        
        time[         tt] = t_total
        pos_history[  tt] = pos
        rL_history[   tt] = rL
        vel_history[  tt] = vel
        mag_history[  tt] = Bp

    return init_pos, init_vel, init_pitch, time, pos_history, rL_history, vel_history, mag_history, DT, max_t


def test_mirror_motion():
    '''
    Diagnostic code to call the particle pushing part of the hybrid and check
    that its solving ok. Runs with zero background E field and B field defined
    by the constant background field specified in the parameter script.
    '''
    max_rev = 5000
    init_pos, init_vel, init_pitch, time, pos_history, rL_history, vel_history, mag_history, DT, max_t = do_particle_run(max_rev=max_rev)
    Np = init_pitch.shape[0]  
    
    # Calculate parameter timeseries using recorded values
    init_vperp = np.sqrt(init_vel[1] ** 2 + init_vel[0] ** 2)
    init_vpara = init_vel[2]
    init_KE    = 0.5 * mp * init_vel ** 2
    init_mu    = 0.5 * mp * init_vperp ** 2 / B_eq
    
    vel_perp      = np.sqrt(vel_history[:, 1] ** 2 + vel_history[:, 0] ** 2)
    vel_para      = vel_history[:, 2]
    vel_magnitude = np.sqrt(vel_history[:, 0] ** 2 + vel_history[:, 1] ** 2 + vel_history[:, 2] ** 2)
    
    B_para      = mag_history[:, 2]
    B_perp      = np.sqrt(mag_history[:, 1] ** 2 + mag_history[:, 0] ** 2)
    B_magnitude = np.sqrt(mag_history[:, 0] ** 2 + mag_history[:, 1] ** 2 + mag_history[:, 2] ** 2)
    
    KE_perp = 0.5 * mp * (vel_history[:, 1] ** 2 + vel_history[:, 0] ** 2)
    KE_para = 0.5 * mp *  vel_history[:, 2] ** 2
    KE_tot  = KE_para + KE_perp
    
    mu  = KE_perp / B_magnitude
    run = r'const. $v_\parallel$ = %5.2fkm/s :: Loss Cone = %5.2f$^\circ$'%(v_parallel*1e-3, loss_cone)
    
    if False:
        # Basic mu plot with v_perp, |B| also plotted
        fig, axes = plt.subplots(4, sharex=True)
        axes[0].set_title(r'Particle First Invariants ($\mu$) by Initial Pitch Angle, $\alpha$ :: %s' % run)
        
        for ii in range(Np):
            axes[0].plot(time, mu[:, ii]         *1e10, lw=0.8, label=r'$\alpha$ = %4.1f$^\circ$' % (init_pitch[ii] * 180 / np.pi))
            axes[1].plot(time, vel_perp[:, ii]   *1e-3, lw=0.8)
            axes[2].plot(time, B_magnitude[:, ii]*1e9 , lw=0.8)
            axes[3].plot(time, pos_history[:, ii]*1e-3, lw=0.8)
            
            #axes[0].axhline(init_mu[:, ii]*1e10, ls=':')
        axes[0].legend(loc=2, ncol=5)
        axes[0].set_ylabel(r'$\mu (\times 10^{-10})$', rotation=0, labelpad=30)
        axes[1].set_ylabel('$v_\perp$\n(km/s)', rotation=0, labelpad=20)
        axes[2].set_ylabel('$|B|(t_v)$ (nT)', rotation=0, labelpad=20)
        axes[3].set_ylabel('$x$\n(km)', rotation=0, labelpad=20)
        
        for ax in axes:
            ax.set_xlim(0, time[-1])
            ax.set_xlabel('Time (s)')
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
        
    if False:
        ## Plots velocity/mag timeseries ##
        fig, axes = plt.subplots(2, sharex=True)
        
        axes[0].plot(time, vel_history[:, 0]* 1e-3, label='vx')
        axes[0].plot(time, vel_history[:, 1]* 1e-3, label='vy')
        axes[0].plot(time, vel_perp         * 1e-3, label='v_perp')
        axes[0].plot(time, vel_para*1e-3          , label='v_para')
        
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
        
        
    if True:
        
        mirror_force =  - init_mu * 4 * B_eq * a * pos_history
        mirror_accel = mirror_force / mp
        
        root   = np.sqrt((a*rL_history*pos_history) ** 2 + 2*a*pos_history**2 + a**2*pos_history**4)
        top    = 2*B_eq*rL_history*a**2*pos_history**2
        grad_B = top/root 
        rhs    = q*B_magnitude**2 / (mp * vel_perp)
        pdb.set_trace()
    
        ## Plots position/velocity component timeseries ##
        fig, axes = plt.subplots(4, sharex=True)
        axes[0].set_title(r'Particle Positions/Velocities by initial Pitch Angle, $\alpha$ : %s' % run)

        for ii in range(Np):
            axes[0].plot(time, pos_history[:, ii]*1e-3, label=r'$\alpha$ = %4.1f$^\circ$' % (init_pitch[ii] * 180 / np.pi))
            axes[1].plot(time, vel_para[:, ii]/va)
            axes[2].plot(time, vel_perp[:, ii]/va)
            axes[3].plot(time, mirror_force[:, ii])
            
        axes[0].legend(loc=2, ncol=5)

        axes[0].axhline(xmin*1e-3, color='k', ls=':')
        axes[0].axhline(xmax*1e-3, color='k', ls=':')

        axes[0].set_ylabel(r'x (km)')
        axes[1].set_ylabel(r'$v_\parallel$ ($v_{A,eq}^{-1}$)')       
        axes[2].set_ylabel(r'$v_\perp$ ($v_{A,eq}^{-1}$)')
        axes[3].set_ylabel(r'$f_\parallel$)')
        
        for ax in axes:
            ax.set_xlim(0, time[-1])
            ax.set_xlabel('t (s)')
            
            
            
    if False:
        # Plot gyromotion of particle vx vs. vy
        plt.title('Particle gyromotion: {} gyroperiods ({:.1f}s)'.format(max_rev, max_t))
        plt.scatter(vel_history[:, 0], vel_history[:, 1], c=time)
        plt.colorbar().set_label('Time (s)')
        plt.ylabel('vx (km/s)')
        plt.xlabel('vy (km/s)')
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
        ax.plot(pos_history*1e-3, vel_history[:, 2]*1e-3, c='b', label=r'$v_\parallel$')
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