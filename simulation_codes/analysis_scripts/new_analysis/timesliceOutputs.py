# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:10:38 2023

@author: Yoshi
"""
'''
Field and/or particle summary outputs that plot information at a specific time,
often to an output folder, and often skipping some number of field outputs
'''
import os
import numpy as np
import matplotlib.pyplot as plt

def winske_summary_plots(Sim, save=True):
    '''
    Plot summaries as per Winske et al. (1993)
    Really only works for Winske load files.
    0: Cold
    1: Hot
    '''  
    np.set_printoptions(suppress=True)
    
    if Sim.num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return

    path = Sim.anal_dir + '/winske_plots/'
    if not os.path.exists(path): os.makedirs(path)
    
    plt.ioff()
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = Sim.interpolate_fields_to_particle_time()

    radperiods = ptime_sec * cf.gyfreq
    gperiods   = ptime_sec / cf.gyperiod
    
    # Normalize units
    qdens_norm = pqdens / (cf.density*cf.charge).sum()     
    BY_norm    = pby / cf.B_eq
    BZ_norm    = pbz / cf.B_eq
    PHI        = np.arctan2(BZ_norm, BY_norm)
    
    for ii in range(num_particle_steps):
        filename = 'winske_summary%05d.png' % ii
        fullpath = path + filename
        
        if os.path.exists(fullpath):
            sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
            continue
        
        sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(run_num, ii))
        sys.stdout.flush()

        fig, axes = plt.subplots(5, figsize=(20,10), sharex=True)                  # Initialize Figure Space

        xp, vp, idx, psim_time, idx_start, idx_end = cf.load_particles(ii)
        
        pos       = xp / cf.dx
        vel       = vp / c
        cell_B    = cf.B_nodes/cf.dx
        cell_E    = cf.E_nodes/cf.dx
        st, en    = idx_start[1], idx_end[1]
        xmin      = cf.xmin / cf.dx
        xmax      = cf.xmax / cf.dx

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


def summary_plots(Sim, save=True, histogram=True, skip=1, ylim=True):
    '''
    Plot summary plot of raw values for each particle timestep
    Field values are interpolated to this point
    
    To Do: Find some nice way to include species energies instead of betas
    '''  
    np.set_printoptions(suppress=True)
    
    if Sim.num_particle_steps == 0:
        print('ABORT: No particle data present to create summary plots.')
        return
    
    if histogram == True:
        path = Sim.anal_dir + '/summary_plots_histogram/'
    else:
        path = Sim.anal_dir + '/summary_plots/'
        
    if os.path.exists(path) == False:                                   # Create data directory
        os.makedirs(path)
    
    plt.ioff()
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = cf.interpolate_fields_to_particle_time(num_particle_steps)

    time_seconds_particle    = ptime_sec
    time_gperiods_particle   = ptime_sec / cf.gyperiod 
    time_radperiods_particle = ptime_sec / cf.gyfreq
    
    # Normalized change density
    qdens_norm = pqdens / (cf.density*cf.charge).sum()     

    B_lim   = np.array([-1.0*pby.min(), pby.max(), -1.0*pbz.min(), pbz.max()]).max() * 1e9
    vel_lim = 20
    E_lim   = np.array([-1.0*pex.min(), pex.max(), -1.0*pey.min(), pey.max(), -1.0*pez.min(), pez.max()]).max() * 1e3
    den_max = None#qdens_norm.max()
    den_min = None#2.0 - den_max

    # Set lims to ceiling values (Nope: Field limits are still doing weird shit. It's alright.)
    B_lim = np.ceil(B_lim)
    E_lim = np.ceil(E_lim)
    
    for ii in range(num_particle_steps):
        if ii%skip == 0:
            filename = 'summ%05d.png' % ii
            fullpath = path + filename
            
            if os.path.exists(fullpath):
                sys.stdout.write('\rSummary plot already present for timestep [{}]{}'.format(run_num, ii))
                sys.stdout.flush()
                continue
            
            sys.stdout.write('\rCreating summary plot for particle timestep [{}]{}'.format(run_num, ii))
            sys.stdout.flush()
    
            fig_size = 4, 7                                                             # Set figure grid dimensions
            fig = plt.figure(figsize=(20,10))                                           # Initialize Figure Space
            fig.patch.set_facecolor('w')   
            xp, vp, idx, psim_time, idx_start, idx_end = cf.load_particles(ii)
            
            pos       = xp  
            vel       = vp / cf.va 
    
            # Count particles lost to the simulation
            N_lost = np.zeros(cf.Nj, dtype=int)
            if idx is not None:
                Nl_idx  = idx[idx < 0]  # Collect indices of those < 0
                Nl_idx += 128           # Cast from negative to positive indexes ("reactivate" particles)
                for jj in range(cf.Nj):
                    N_lost[jj] = Nl_idx[Nl_idx == jj].shape[0]
            else:
                N_lost = None
            ax_vx   = plt.subplot2grid(fig_size, (0, 0), rowspan=2, colspan=3)
            ax_vy   = plt.subplot2grid(fig_size, (2, 0), rowspan=2, colspan=3)
            
            if histogram == True:
                
                vel_tr = np.sqrt(vel[1] ** 2 + vel[2] ** 2)
                
                for jj in range(cf.Nj):
                    try:
                        num_bins = cf.nsp_ppc[jj] // 5
                        
                        xs, BinEdgesx = np.histogram(vel[0, idx_start[jj]: idx_end[jj]], bins=num_bins)
                        bx = 0.5 * (BinEdgesx[1:] + BinEdgesx[:-1])
                        ax_vx.plot(bx, xs, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
                        
                        ys, BinEdgesy = np.histogram(vel_tr[idx_start[jj]: idx_end[jj]], bins=num_bins)
                        by = 0.5 * (BinEdgesy[1:] + BinEdgesy[:-1])
                        ax_vy.plot(by, ys, '-', c=cf.temp_color[jj], drawstyle='steps', label=cf.species_lbl[jj])
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
                            ax.set_ylim(0, int(cf.N / cf.NX) * 10.0)
            else:
            
                for jj in reversed(range(cf.Nj)):
                    ax_vx.scatter(pos[idx_start[jj]: idx_end[jj]], vel[0, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0, label=cf.species_lbl[jj])
                    ax_vy.scatter(pos[idx_start[jj]: idx_end[jj]], vel[1, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj], lw=0)
            
                ax_vx.legend(loc='upper right')
                ax_vx.set_title(r'Particle velocities (in $v_A$) vs. Position (x)')
                ax_vy.set_xlabel(r'Cell', labelpad=10)
                ax_vx.set_ylabel(r'$v_x$', rotation=0)
                ax_vy.set_ylabel(r'$v_y$', rotation=0)
                
                plt.setp(ax_vx.get_xticklabels(), visible=False)
                ax_vx.set_yticks(ax_vx.get_yticks()[1:])
            
                for ax in [ax_vy, ax_vx]:
                    ax.set_xlim(-cf.xmax, cf.xmax)
                    if ylim == True:
                        ax.set_ylim(-vel_lim, vel_lim)
        
            
            ## DENSITY ##
            B_nodes  = (np.arange(cf.NC + 1) - cf.NC // 2)            # B grid points position in space
            E_nodes  = (np.arange(cf.NC)     - cf.NC // 2 + 0.5)      # E grid points position in space
    
            ax_den  = plt.subplot2grid(fig_size, (0, 3), colspan=3)
            ax_den.plot(E_nodes, qdens_norm[ii], color='green')
                    
            ax_den.set_title('Charge Density and Fields')
            ax_den.set_ylabel(r'$\frac{\rho_c}{\rho_{c0}}$', fontsize=14, rotation=0, labelpad=20)
            
    
            ax_Ex   = plt.subplot2grid(fig_size, (1, 3), colspan=3, sharex=ax_den)
            ax_Ex.plot(E_nodes, pex[ii]*1e3, color='red',   label=r'$E_x$')
            ax_Ex.plot(E_nodes, pey[ii]*1e3, color='cyan',  label=r'$E_y$')
            ax_Ex.plot(E_nodes, pez[ii]*1e3, color='black', label=r'$E_z$')
            ax_Ex.set_ylabel(r'$E (mV/m)$', labelpad=25, rotation=0, fontsize=14)
            
            ax_Ex.legend(loc='upper right', ncol=3)
            
            ax_By  = plt.subplot2grid(fig_size, (2, 3), colspan=3, sharex=ax_den)
            ax_B   = plt.subplot2grid(fig_size, (3, 3), colspan=3, sharex=ax_den)
            mag_B  = np.sqrt(pby[ii] ** 2 + pbz[ii] ** 2)
            
            ax_Bx = ax_B.twinx()
            ax_Bx.plot(B_nodes, pbx[ii]*1e9, color='k', label=r'$B_x$', ls=':', alpha=0.6) 
            
            if cf.B_eq == cf.Bc.max():
                pass
            else:
                if ylim == True:
                    ax_Bx.set_ylim(cf.B_eq*1e9, cf.Bc.max()*1e9)
                
            ax_Bx.set_ylabel(r'$B_{0x} (nT)$', rotation=0, labelpad=30, fontsize=14)
            
            ax_B.plot( B_nodes, mag_B*1e9, color='g')
            ax_By.plot(B_nodes, pby[ii]*1e9, color='g',   label=r'$B_y$') 
            ax_By.plot(B_nodes, pbz[ii]*1e9, color='b',   label=r'$B_z$') 
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
                ax.set_xlim(B_nodes[0], B_nodes[-1])
                ax.axvline(-cf.NX//2, c='k', ls=':', alpha=0.5)
                ax.axvline( cf.NX//2, c='k', ls=':', alpha=0.5)
                ax.grid()
                    
            plt.tight_layout(pad=1.0, w_pad=1.8)
            fig.subplots_adjust(hspace=0.125)
            
            ###################
            ### FIGURE TEXT ###
            ###################
            Tperp = cf.mass * cf.vth_perp ** 2 / kB
            anisotropy = (cf.vth_perp ** 2 / cf.vth_par ** 2 - 1).round(1)
            beta_per   = (2*(4e-7*np.pi)*(1.381e-23)*Tperp*cf.ne / (cf.B_eq**2)).round(1)
            #beta_e     = np.round((2*(4e-7*np.pi)*(1.381e-23)*cf.Te0*cf.ne  / (cf.B_eq**2)), 2)
            rdens      = (cf.density / cf.ne).round(2)
    
            try:
                vdrift     = (cf.velocity / cf.va).round(1)
            except:
                vdrift     = (cf.drift_v / cf.va).round(1)
            
            if cf.ie == 0:
                estring = 'Isothermal electrons'
            elif cf.ie == 1:
                estring = 'Adiabatic electrons'
            else:
                'Electron relation unknown'
                        
            top  = 0.95
            gap  = 0.025
            fontsize = 12
            plt.figtext(0.855, top        , 'Simulation Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
            plt.figtext(0.855, top - 1*gap, '{}[{}]'.format(series, run_num), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 2*gap, '{} cells'.format(cf.NX), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 3*gap, '{} particles'.format(cf.N_species.sum()), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 4*gap, '{}'.format(estring), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 5*gap, '', fontsize=fontsize, family='monospace')
            
            plt.figtext(0.855, top - 6*gap, 'B_eq      : {:.1f}nT'.format(cf.B_eq*1e9  ), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 7*gap, 'B_xmax    : {:.1f}nT'.format(cf.B_xmax*1e9), fontsize=fontsize, family='monospace')
            
            plt.figtext(0.855, top - 8*gap,  'ne        : {}cc'.format(cf.ne/1e6), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 9*gap,  'N_species : {}'.format(cf.N_species), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 10*gap, 'N_lost    : {}'.format(N_lost), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 11*gap, '', fontsize=fontsize, family='monospace')
            
            #plt.figtext(0.855, top - 12*gap, r'$\beta_e$      : %.2f' % beta_e, fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 13*gap, 'dx      : {}km'.format(round(cf.dx/1e3, 2)), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 14*gap, 'L       : {}'.format(cf.L), fontsize=fontsize, family='monospace')
            plt.figtext(0.855, top - 15*gap, 'MLAT_max: $\pm$%.1f$^{\circ}$' % (cf.theta_xmax * 180. / np.pi), fontsize=fontsize, family='monospace')
    
            plt.figtext(0.855, top - 16*gap, '', fontsize=fontsize, family='monospace')
            
            ptop  = top - 17*gap
            pside = 0.855
            plt.figtext(pside, ptop, 'Particle Parameters', fontsize=fontsize, family='monospace', fontweight='bold')
            plt.figtext(pside, ptop - gap, ' SPECIES  ANI  XBET    VDR  RDNS', fontsize=fontsize-2, family='monospace')
            for jj in range(cf.Nj):
                plt.figtext(pside       , ptop - (jj + 2)*gap, '{:>10}  {:>3}  {:>5}  {:>4}  {:<5}'.format(
                        cf.species_lbl[jj], anisotropy[jj], beta_per[jj], vdrift[jj], rdens[jj]),
                        fontsize=fontsize-2, family='monospace')
     
            time_top = 0.1
            plt.figtext(0.88, time_top - 0*gap, 't_seconds   : {:>10}'.format(round(time_seconds_particle[ii], 3))   , fontsize=fontsize, family='monospace')
            plt.figtext(0.88, time_top - 1*gap, 't_gperiod   : {:>10}'.format(round(time_gperiods_particle[ii], 3))  , fontsize=fontsize, family='monospace')
            plt.figtext(0.88, time_top - 2*gap, 't_radperiod : {:>10}'.format(round(time_radperiods_particle[ii], 3)), fontsize=fontsize, family='monospace')
        
            if save == True:
                plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close('all')
    print('\n')
    return