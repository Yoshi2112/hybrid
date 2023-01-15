# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:10:38 2023

@author: Yoshi
"""
'''
Field and/or particle summary outputs that plot information at a specific time,
often to an output folder, and often skipping some number of field outputs
'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from _constants import UNIT_CHARGE, PROTON_MASS, ELECTRON_MASS, ELEC_PERMITTIVITY, MAGN_PERMEABILITY, BOLTZMANN_CONSTANT, LIGHT_SPEED


def winskeSummaryPlots(Sim, save=True, skip=1):
    '''
    Plot summaries as per Winske et al. (1993)
    Really only works for Winske load files.
    0: Cold
    1: Hot
    '''  
    print('Creating Winske summary outputs...')
    np.set_printoptions(suppress=True)
    
    if Sim.num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return

    save_dir = Sim.anal_dir + '/winske_plots/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    plt.ioff()
    pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = Sim.interpolateFields2ParticleTimes()

    radperiods = Sim.particle_sim_time * Sim.gyfreq_eq
    gperiods   = Sim.particle_sim_time / Sim.gyperiod_eq
    
    # Normalize units
    qdens_norm = pqdens / (Sim.density*Sim.charge).sum()     
    BY_norm    = pby / Sim.B_eq
    BZ_norm    = pbz / Sim.B_eq
    PHI        = np.arctan2(BZ_norm, BY_norm)
    
    for ii in range(Sim.num_particle_steps):
        if ii%skip == 0:
            filename = 'winske_summary%05d.png' % ii
            fullpath = save_dir + filename
            
            if os.path.exists(fullpath):
                sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(Sim.run_num, ii))
                sys.stdout.flush()
                continue
            sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(Sim.run_num, ii))
            sys.stdout.flush()
    
            fig, axes = plt.subplots(5, figsize=(20,10), sharex=True)                  # Initialize Figure Space
    
            xp, vp, idx, psim_time, idx_start, idx_end = Sim.load_particles(ii)
            
            pos       = xp / Sim.dx
            vel       = vp / LIGHT_SPEED
            cell_B    = Sim.B_nodes/Sim.dx
            cell_E    = Sim.E_nodes/Sim.dx
            st, en    = idx_start[1], idx_end[1]
            xmin      = Sim.xmin / Sim.dx
            xmax      = Sim.xmax / Sim.dx
    
            axes[0].scatter(pos[st: en], vel[0, st: en], s=1, c='k')
            axes[1].scatter(pos[st: en], vel[1, st: en], s=1, c='k')
            axes[2].plot(cell_E, qdens_norm[ii], color='k', lw=1.0)
            axes[3].plot(cell_B, BY_norm[ii], color='k')
            axes[4].plot(cell_B, PHI[ii],     color='k')
            
            axes[0].set_ylabel('VX ', labelpad=20, rotation=0)
            axes[1].set_ylabel('VY ', labelpad=20, rotation=0)
            axes[2].set_ylabel('DN ', labelpad=20, rotation=0)
            axes[3].set_ylabel('BY ', labelpad=20)
            axes[4].set_ylabel('PHI', labelpad=20)
            fig.align_ylabels()
            
            axes[0].set_title('BEAM')
            axes[1].set_title('BEAM')
            
            axes[0].set_ylim(-2e-3, 2e-3)
            axes[1].set_ylim(-2e-3, 2e-3)
            axes[2].set_ylim(0.5, 1.5)
            axes[4].set_ylim(-np.pi, np.pi)
            
            axes[0].set_title('WINSKE RUNS :: IT={:04d} :: T={:5.2f} :: GP={:5.2f}'.format(ii, radperiods[ii], gperiods[ii]), family='monospace')
            axes[4].set_xlabel('X')
            
            for ax in axes:
                ax.set_xlim(xmin, xmax)
            
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close('all')
    print('\n')
    return


def summaryPlots(Sim, save=True, histogram=True, skip=1, ylim=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    
    To Do: Find some nice way to include species energies instead of betas
    '''  
    print('Creating summary outputs...')
    np.set_printoptions(suppress=True)
    
    if Sim.num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return
    
    if histogram == True:
        save_dir = Sim.anal_dir + '/summary_plots_histogram/'
    else:
        save_dir = Sim.anal_dir + '/summary_plots/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    plt.ioff()
    pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = Sim.interpolateFields2ParticleTimes()

    time_gperiods_particle   = Sim.particle_sim_time / Sim.gyperiod 
    time_radperiods_particle = Sim.particle_sim_time / Sim.gyfreq
    
    # Normalized change density
    qdens_norm = pqdens / (Sim.density*Sim.charge).sum()     

    B_lim   = np.array([-1.0*pby.min(), pby.max(), -1.0*pbz.min(), pbz.max()]).max() * 1e9
    vel_lim = 20
    E_lim   = np.array([-1.0*pex.min(), pex.max(), -1.0*pey.min(), pey.max(), -1.0*pez.min(), pez.max()]).max() * 1e3
    den_max = None#qdens_norm.max()
    den_min = None#2.0 - den_max

    # Set lims to ceiling values
    B_lim = np.ceil(B_lim)
    E_lim = np.ceil(E_lim)
    
    for ii in range(Sim.num_particle_steps):
        if ii%skip == 0:
            filename = 'summ%05d.png' % ii
            fullpath = save_dir + filename
            
            if os.path.exists(fullpath):
                sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(Sim.run_num, ii))
                sys.stdout.flush()
                continue
            sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(Sim.run_num, ii))
            sys.stdout.flush()
    
            fig_size = 4, 7                                                             # Set figure grid dimensions
            fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
            fig.patch.set_facecolor('w')   
            xp, vp, idx, psim_time, idx_start, idx_end = Sim.load_particles(ii)
            
            pos       = xp  
            vel       = vp / Sim.va 
    
            ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
            ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
            
            if histogram == True:
                
                vel_tr = np.sqrt(vel[1] ** 2 + vel[2] ** 2)
                
                for jj in range(Sim.Nj):
                    try:
                        num_bins = Sim.nsp_ppc[jj] // 5
                        
                        xs, BinEdgesx = np.histogram(vel[0, idx_start[jj]: idx_end[jj]], bins=num_bins)
                        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
                        ax_vx.plot(bx, xs, '-', c=Sim.temp_color[jj], drawstyle='steps', label=Sim.species_lbl[jj])
                        
                        ys, BinEdgesy = np.histogram(vel_tr[idx_start[jj]: idx_end[jj]], bins=num_bins)
                        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
                        ax_vy.plot(by, ys, '-', c=Sim.temp_color[jj], drawstyle='steps', label=Sim.species_lbl[jj])
                    except: 
                        pass
                    
                    ax_vx.set_ylabel(r'$n_{v_\parallel}$')
                    ax_vx.set_ylabel(r'$n_{v_\perp}$')
                    
                    ax_vx.set_title('Velocity distribution of each species in simulation domain')
                    ax_vy.set_xlabel(r'$v / v_A$')
                    
                    ax_vx.set_xlim(-vel_lim, vel_lim)
                    ax_vy.set_xlim(0, np.sqrt(2)*vel_lim)
                    
                    for ax, comp in zip([ax_vx, ax_vy], ['v_\parallel', 'v_\perp']):
                        ax.legend(loc='upper right')
                        ax.set_ylabel('$n_{%s}$'%comp)
                        if ylim == True:
                            ax.set_ylim(0, int(Sim.N / Sim.NX) * 10.0)
            else:
            
                for jj in reversed(range(Sim.Nj)):
                    ax_vx.scatter(pos[idx_start[jj]: idx_end[jj]], vel[0, idx_start[jj]: idx_end[jj]], s=1, c=Sim.temp_color[jj], lw=0, label=Sim.species_lbl[jj])
                    ax_vy.scatter(pos[idx_start[jj]: idx_end[jj]], vel[1, idx_start[jj]: idx_end[jj]], s=1, c=Sim.temp_color[jj], lw=0)
            
                ax_vx.legend(loc='upper right')
                ax_vx.set_title(r'Particle velocities (in $v_A$) vs. Position (x)')
                ax_vy.set_xlabel(r'Cell', labelpad=10)
                ax_vx.set_ylabel(r'$v_x$', rotation=0)
                ax_vy.set_ylabel(r'$v_y$', rotation=0)
                
                plt.setp(ax_vx.get_xticklabels(), visible=False)
                ax_vx.set_yticks(ax_vx.get_yticks()[1:])
            
                for ax in [ax_vy, ax_vx]:
                    ax.set_xlim(Sim.xmin, Sim.xmax)
                    if ylim == True:
                        ax.set_ylim(-vel_lim, vel_lim)
        
            
            ## DENSITY ##    
            ax_den  = plt.subplot2grid(fig_size, (0, 3), colspan=3)
            ax_den.plot(Sim.E_nodes, qdens_norm[ii], color='green')
                    
            ax_den.set_title('Charge Density and Fields')
            ax_den.set_ylabel(r'$\frac{\rho_c}{\rho_{c0}}$', fontsize=14, rotation=0, labelpad=20)
            
    
            ax_Ex   = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
            ax_Ex.plot(Sim.E_nodes, pex[ii]*1e3, color='red',   label=r'$E_x$')
            ax_Ex.plot(Sim.E_nodes, pey[ii]*1e3, color='cyan',  label=r'$E_y$')
            ax_Ex.plot(Sim.E_nodes, pez[ii]*1e3, color='black', label=r'$E_z$')
            ax_Ex.set_ylabel(r'$E (mV/m)$', labelpad=25, rotation=0, fontsize=14)
            
            ax_Ex.legend(loc='upper right', ncol=3)
            
            ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
            ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
            mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
            
            ax_Bx = ax_B.twinx()
            ax_Bx.plot(Sim.B_nodes, pbx[ii]*1e9, color='k', label=r'$B_x$', ls=':', alpha=0.6) 
            
            if Sim.B_eq == Sim.Bc.max():
                pass
            else:
                if ylim == True:
                    ax_Bx.set_ylim(Sim.B_eq*1e9, Sim.Bc.max()*1e9)
                
            ax_Bx.set_ylabel(r'$B_{0x} (nT)$', rotation=0, labelpad=30, fontsize=14)
            
            ax_B.plot( Sim.B_nodes, mag_B*1e9, color='g')
            ax_By.plot(Sim.B_nodes, pby[ii]*1e9, color='g',   label=r'$B_y$') 
            ax_By.plot(Sim.B_nodes, pbz[ii]*1e9, color='b',   label=r'$B_z$') 
            ax_By.legend(loc='upper right', ncol=2)
            
            ax_B.set_ylabel( r'$B_\perp (nT)$', rotation=0, labelpad=30, fontsize=14)
            ax_By.set_ylabel(r'$B_{y,z} (nT)$', rotation=0, labelpad=20, fontsize=14)
            ax_B.set_xlabel('Cell Number')
            
            # SET FIELD RANGES #
            if ylim == True:
                try:
                    ax_den.set_ylim(den_min, den_max)
                except:
                    pass
                
                try:
                    ax_Ex.set_ylim(-E_lim, E_lim)
                except:
                    pass
                
                try:
                    ax_By.set_ylim(-B_lim, B_lim)
                    ax_B.set_ylim(0, B_lim)
                except:
                    pass
            
            for ax in [ax_den, ax_Ex, ax_By]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_yticks(ax.get_yticks()[1:])
                
            for ax in [ax_den, ax_Ex, ax_By, ax_B]:
                ax.set_xlim(Sim.B_nodes[0], Sim.B_nodes[-1])
                ax.axvline(-Sim.NX//2, c='k', ls=':', alpha=0.5)
                ax.axvline( Sim.NX//2, c='k', ls=':', alpha=0.5)
                ax.grid()
                    
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0.125)
            
            ###################
            ### FIGURE TEXT ###
            ###################
            beta_per   = (2*MAGN_PERMEABILITY*BOLTZMANN_CONSTANT*Sim.Tperp*Sim.ne / Sim.B_eq**2).round(1)
            rdens      = (Sim.density / Sim.ne).round(2)
            vdrift     = (Sim.drift_v / Sim.va).round(1)
            
            if Sim.ie == 0:
                estring = 'Isothermal electrons'
            elif Sim.ie == 1:
                estring = 'Adiabatic electrons'
            else:
                'Electron relation unknown'
                        
            top  = 0.95
            gap  = 0.025
            fontsize = 12
            plt.figtext(0.855, top        , 'Simulation Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
            plt.figtext(0.855, top - 1*gap, '{}[{}]'.format(Sim.series_name, Sim.run_num), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 2*gap, '{} cells'.format(Sim.NX), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 3*gap, '{} particles'.format(Sim.N), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 4*gap, '{}'.format(estring), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 5*gap, '', fontsize=fontsize, family='monospace')
            
            plt.figtext(0.855, top - 6*gap, 'B_eq      : {:.1f}nT'.format(Sim.B_eq*1e9  ), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 7*gap, 'B_xmax    : {:.1f}nT'.format(Sim.B_xmax*1e9), fontsize=fontsize, family='monospace')
            
            plt.figtext(0.855, top - 8*gap,  'ne        : {}cc'.format(Sim.ne/1e6), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 9*gap,  'N_species : {}'.format(Sim.N_species), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 11*gap, '', fontsize=fontsize, family='monospace')
            
            #plt.figtext(0.855, top - 12*gap, r'$\beta_e$      : %.2f' % beta_e, fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 13*gap, 'dx      : {}km'.format(round(Sim.dx/1e3, 2)), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 14*gap, 'L       : {}'.format(Sim.L), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 15*gap, 'MLAT_max: $\pm$%.1f$^{\circ}$' % (Sim.theta_xmax * 180. / np.pi), fontsize=fontsize, family='monospace')
    
            plt.figtext(0.855, top - 16*gap, '', fontsize=fontsize, family='monospace')
            
            ptop  = top - 17*gap
            pside = 0.855
            plt.figtext(pside, ptop, 'Particle Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
            plt.figtext(pside, ptop - gap, ' SPECIES  ANI  XBET    VDR  RDNS', fontsize=fontsize-2, family='monospace')
            for jj in range(Sim.Nj):
                plt.figtext(pside       , ptop - (jj + 2)*gap, '{:>10}  {:>3}  {:>5}  {:>4}  {:<5}'.format(
                        Sim.species_lbl[jj], Sim.anisotropy[jj], beta_per[jj], vdrift[jj], rdens[jj]),
                        fontsize=fontsize-2, family='monospace')
     
            time_top = 0.1
            plt.figtext(0.88, time_top - 0*gap, 't_seconds   : {:>10}'.format(round(Sim.particle_sim_time[ii], 3))   , fontsize=fontsize, family='monospace')
            plt.figtext(0.88, time_top - 1*gap, 't_gperiod   : {:>10}'.format(round(time_gperiods_particle[ii], 3))  , fontsize=fontsize, family='monospace')
            plt.figtext(0.88, time_top - 2*gap, 't_radperiod : {:>10}'.format(round(time_radperiods_particle[ii], 3)), fontsize=fontsize, family='monospace')
        
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close('all')
    print('\n')
    return


def check_fields(Sim, save=True, ylim=True, skip=1):
    '''
    Plot summary plot of raw values for each field timestep
    Field values are interpolated to this point
    '''    
    if ylim == True:
        path = Sim.anal_dir + '/field_plots_all_ylim/'
    else:
        path = Sim.anal_dir + '/field_plots_all/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
        
    plt.ioff()
    
    # Convert to plot units
    Sim.bx *= 1e9; Sim.by *= 1e9; Sim.bz *= 1e9
    Sim.ex *= 1e3; Sim.ey *= 1e3; Sim.ez *= 1e3
    Sim.qdens *= 1e-6/UNIT_CHARGE;    Sim.te /= 11603.
    
    ## Initialize plots and prepare plotspace
    fontsize = 14; fsize = 12; lpad = 20

    ## Do actual plotting and saving of data
    for ii in range(bx.shape[0]):
        if ii%skip == 0:
            
            filename = 'summ%05d.png' % ii
            fullpath = path + filename
            
            if os.path.exists(fullpath):
                continue
            
            fig, axes = plt.subplots(5, ncols=3, figsize=(20,10), sharex=True)
            fig.patch.set_facecolor('w') 
    
            axes[0, 1].set_title('Time :: {:<7.4f}s'.format(ftime[ii]), fontsize=fontsize+4, family='monospace')
    
            sys.stdout.write('\rCreating summary field plots [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
            
            # Wave Fields (Plots, Labels, Lims)
            axes[0, 0].plot(cf.B_nodes / cf.dx, rD[ii], color='k', label=r'$r_D(x)$') 
            axes[1, 0].plot(cf.B_nodes / cf.dx, by[ii], color='b', label=r'$B_y$') 
            axes[2, 0].plot(cf.B_nodes / cf.dx, bz[ii], color='g', label=r'$B_z$')
            axes[3, 0].plot(cf.E_nodes / cf.dx, ey[ii], color='b', label=r'$E_y$')
            axes[4, 0].plot(cf.E_nodes / cf.dx, ez[ii], color='g', label=r'$E_z$')
            
            # Transverse Electric Field Variables (Plots, Labels, Lims)
            axes[0, 1].plot(cf.E_nodes / cf.dx, qdens[ii], color='k', label=r'$n_e$')
            axes[1, 1].plot(cf.E_nodes / cf.dx, vey[ii], color='b', label=r'$V_{ey}$')
            axes[2, 1].plot(cf.E_nodes / cf.dx, vez[ii], color='g', label=r'$V_{ez}$')
            axes[3, 1].plot(cf.E_nodes / cf.dx, jy[ii], color='b', label=r'$J_{iy}$' )
            axes[4, 1].plot(cf.E_nodes / cf.dx, jz[ii], color='g', label=r'$J_{iz}$' )
            
            # Parallel Variables (Plots, Labels, Lims)
            axes[0, 2].plot(cf.E_nodes / cf.dx, te[ii], color='k', label=r'$T_e$')
            axes[1, 2].plot(cf.E_nodes / cf.dx, vex[ii], color='r', label=r'$V_{ex}$')
            axes[2, 2].plot(cf.E_nodes / cf.dx, jx[ii], color='r', label=r'$J_{ix}$' )
            axes[3, 2].plot(cf.E_nodes / cf.dx, ex[ii], color='r', label=r'$E_x$')
            axes[4, 2].plot(cf.B_nodes / cf.dx, bx[ii], color='r', label=r'$B_{0x}$')
            
            axes[0, 0].set_title('Field outputs: {}[{}]'.format(series, run_num), fontsize=fontsize+4, family='monospace')

            axes[0, 0].set_ylabel('$r_D(x)$'     , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[1, 0].set_ylabel('$B_y$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[2, 0].set_ylabel('$B_z$\n(nT)'  , rotation=0, labelpad=lpad, fontsize=fsize)
            axes[3, 0].set_ylabel('$E_y$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
            axes[4, 0].set_ylabel('$E_z$\n(mV/m)', rotation=0, labelpad=lpad, fontsize=fsize)
        
            axes[0, 1].set_ylabel('$n_e$\n$(cm^{-1})$', fontsize=fsize, rotation=0, labelpad=lpad)
            axes[1, 1].set_ylabel('$V_{ey}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[2, 1].set_ylabel('$V_{ez}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[3, 1].set_ylabel('$J_{iy}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[4, 1].set_ylabel('$J_{iz}$'          , fontsize=fsize, rotation=0, labelpad=lpad)
            
            axes[0, 2].set_ylabel('$T_e$\n(eV)'     , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[1, 2].set_ylabel('$V_{ex}$\n(m/s)' , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[2, 2].set_ylabel('$J_{ix}$'        , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[3, 2].set_ylabel('$E_x$\n(mV/m)'   , fontsize=fsize, rotation=0, labelpad=lpad)
            axes[4, 2].set_ylabel('$B_x$\n(nT)'     , fontsize=fsize, rotation=0, labelpad=lpad)
            fig.align_labels()
            
            for ii in range(3):
                axes[4, ii].set_xlabel('Position (m/dx)')
                for jj in range(5):
                    axes[jj, ii].set_xlim(cf.B_nodes[0] / cf.dx, cf.B_nodes[-1] / cf.dx)
                    axes[jj, ii].axvline(-cf.NX//2, c='k', ls=':', alpha=0.5)
                    axes[jj, ii].axvline( cf.NX//2, c='k', ls=':', alpha=0.5)
                    axes[jj, ii].grid()
            
            if ylim == True:
                try:
                    axes[0, 0].set_ylim(rD.min(), rD.max())
                    axes[1, 0].set_ylim(by.min(), by.max())
                    axes[2, 0].set_ylim(bz.min(), bz.max())
                    axes[3, 0].set_ylim(ey.min(), ey.max())
                    axes[4, 0].set_ylim(ez.min(), ez.max())
                    
                    axes[0, 1].set_ylim(qdens.min(), qdens.max())
                    axes[1, 1].set_ylim(vey.min(), vey.max())
                    axes[2, 1].set_ylim(vez.min(), vez.max())
                    axes[3, 1].set_ylim(jy.min() , jy.max())
                    axes[4, 1].set_ylim(jz.min() , jz.max())
                    
                    axes[0, 2].set_ylim(te.min(), te.max())
                    axes[1, 2].set_ylim(vex.min(), vex.max())
                    axes[2, 2].set_ylim(jx.min(), jx.max())
                    axes[3, 2].set_ylim(ex.min(), ex.max())
                except:
                    pass
            
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0.125)
    
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')
    print('\n')
    return