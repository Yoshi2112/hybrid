# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:13:58 2023

@author: Yoshi
"""

def plot_equilibrium_distribution(saveas='eqdist', histogram=True):
    '''
    Plots the initial and final 'relaxed' particle distributions to show how
    the particles were initially loaded and what their state is before the
    beginning of the simulation
    
    Plot x vs. vx, vy, vz and
    vx/vz, vy/vz, vx/vy, v_para/v_perp
    
    '''
    if cf.particle_open == 1:
        shuffled_idx = True
    else:
        shuffled_idx = False
        
    if os.path.exists(cf.data_dir + '//equil_particles//') == False:
        print('No equilibrium data to plot. Aborting.')
        return
    n_equil = len(os.listdir(cf.data_dir + '//equil_particles//'))
    
    filepath = cf.anal_dir + '//equilibrium_distributions//'
    filename = filepath + saveas
    if not os.path.exists(filepath): os.makedirs(filepath)
        
    print('Loading particles...')
    pos1, vel1, idx1, ptime1, idx_start1, idx_end1 =\
        cf.load_particles(0, shuffled_idx=shuffled_idx, preparticledata=True)
    
    pos2, vel2, idx2, ptime2, idx_start2, idx_end2 =\
        cf.load_particles(n_equil-1, shuffled_idx=shuffled_idx, preparticledata=True)
        
    vel1 /= cf.va; vel2 /= cf.va
    pos1 /= cf.dx; pos2 /= cf.dx
        
    v_perp1 = np.sqrt(vel1[1] ** 2 + vel1[2] ** 2) * np.sign(vel1[2])
    v_perp2 = np.sqrt(vel2[1] ** 2 + vel2[2] ** 2) * np.sign(vel2[2])
    
    xbins = np.linspace(cf.xmin/cf.dx, cf.xmax/cf.dx, cf.NX + 1, endpoint=True)
    
    plt.ioff()
    for jj in range(cf.Nj):
        
        # Do the calculations and plotting
        cfac  = 10  if cf.temp_type[jj] == 1 else 10
        vlim  = 5*cf.vth_perp[jj]/cf.va
        vbins = np.linspace(-vlim, vlim, 501, endpoint=True)
        print('\nGenerating plots for species', jj)
        
        if True:
            pfac = 3
            
            # Plot Spatial (vx, vy, vz, vperp) for before and after
            fig1, axes1 = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
            fig1.suptitle(cf.species_lbl[jj])
            axes1[0, 0].set_title('At initialization')
            st1 = idx_start1[jj]; en1 = idx_end1[jj]
            
            counts, xedges, yedges, im1a = axes1[0, 0].hist2d(pos1[st1:en1], vel1[0, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im1a, ax=axes1[0, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[0, 0].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2a = axes1[1, 0].hist2d(pos1[st1:en1], vel1[1, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im2a, ax=axes1[1, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[1, 0].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3a = axes1[2, 0].hist2d(pos1[st1:en1], vel1[2, st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im3a, ax=axes1[2, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[2, 0].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4a = axes1[3, 0].hist2d(pos1[st1:en1], v_perp1[st1:en1], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im4a, ax=axes1[3, 0], pad=0.015)
            cb.set_label('Counts')
            axes1[3, 0].set_ylabel('$v_\perp$', rotation=0)
            
            
            axes1[0, 1].set_title('After relaxation')
            st2 = idx_start2[jj]; en2 = idx_end2[jj]
            
            counts, xedges, yedges, im1b = axes1[0, 1].hist2d(pos2[st2:en2], vel2[0, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im1b, ax=axes1[0, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[0, 1].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2b = axes1[1, 1].hist2d(pos2[st2:en2], vel2[1, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im2b, ax=axes1[1, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[1, 1].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3b = axes1[2, 1].hist2d(pos2[st2:en2], vel2[2, st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im3b, ax=axes1[2, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[2, 1].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4b = axes1[3, 1].hist2d(pos2[st2:en2], v_perp2[st2:en2], 
                                                    bins=[xbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac/pfac)
            cb = fig1.colorbar(im4b, ax=axes1[3, 1], pad=0.015)
            cb.set_label('Counts')
            axes1[3, 1].set_ylabel('$v_\perp$', rotation=0)
           
            for mm in range(4):
                for nn in range(2):
                    axes1[mm, nn].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
                    axes1[mm, nn].set_ylim(-vlim, vlim)
                    
            fig1.subplots_adjust()
            fig1.savefig(filename + f'_spatial_sp{jj}.png')
            print('Plot saved as', filename + f'_spatial_sp{jj}.png')
        
        
        if True:
            vfac=6
            
            # Plot phase space (vx/vy, vx/vz, vy/vz, v_perp/v_para)
            fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
            fig2.suptitle(cf.species_lbl[jj])
            
            axes2[0, 0].set_title('At initialization')
            st1 = idx_start1[jj]; en1 = idx_end1[jj]
            
            counts, xedges, yedges, im1c = axes2[0, 0].hist2d(vel1[0, st1:en1], vel1[1, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im1c, ax=axes2[0, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[0, 0].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2c = axes2[1, 0].hist2d(vel1[0, st1:en1], vel1[2, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im2c, ax=axes2[1, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[1, 0].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3c = axes2[2, 0].hist2d(vel1[1, st1:en1], vel1[2, st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im3c, ax=axes2[2, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[2, 0].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4c = axes2[3, 0].hist2d(vel1[0, st1:en1], v_perp1[st1:en1], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im4c, ax=axes2[3, 0], pad=0.015)
            cb.set_label('Counts')
            axes2[3, 0].set_ylabel('$v_\perp$', rotation=0)
            
            
            axes2[0, 1].set_title('After relaxation')
            st2 = idx_start2[jj]; en2 = idx_end2[jj]
            
            counts, xedges, yedges, im1d = axes2[0, 1].hist2d(vel2[0, st2:en2], vel2[1, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im1d, ax=axes2[0, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[0, 1].set_ylabel('$v_x$', rotation=0)
            
            counts, xedges, yedges, im2d = axes2[1, 1].hist2d(vel2[0, st2:en2], vel2[2, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im2d, ax=axes2[1, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[1, 1].set_ylabel('$v_y$', rotation=0)
            
            counts, xedges, yedges, im3d = axes2[2, 1].hist2d(vel2[1, st2:en2], vel2[2, st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im3d, ax=axes2[2, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[2, 1].set_ylabel('$v_z$', rotation=0)
            
            counts, xedges, yedges, im4d = axes2[3, 1].hist2d(vel2[0, st2:en2], v_perp2[st2:en2], 
                                                    bins=[vbins, vbins], vmin=0, vmax=cf.nsp_ppc[jj]/cfac*vfac)
            cb = fig2.colorbar(im4d, ax=axes2[3, 1], pad=0.015)
            cb.set_label('Counts')
            axes2[3, 1].set_ylabel('$v_\perp$', rotation=0)
            
            for mm in range(4):
                for nn in range(2):
                    axes2[mm, nn].axis('equal')
                    
            fig2.subplots_adjust()
            fig2.savefig(filename + f'_phase_sp{jj}.png')
            print('Plot saved as', filename + f'_phase_sp{jj}.png')
            plt.close('all')
        
    return


def plot_initial_configurations(it_max=None, save=True, plot_lost=True):
    ## Count those that have been lost by the end of the simulation
    ## and plot that against initial distro phase spaces
    #
    ## Notes:
    ##  -- Why are lost particles only in the negative side of the simulation space?
    ##  -- Why is there seemingly no connection between lost particles and loss cone?
    print('Plotting particle initial phase space configs...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    savedir = cf.anal_dir + '/Particle_Loss_Analysis/Initial Particle Configuration/'

    if os.path.exists(savedir) == False:                                   # Create directories
        os.makedirs(savedir)
    
    init_pos , init_vel , init_idx , ptime1 = cf.load_particles(0)
    v_mag  = np.sqrt(init_vel[0] ** 2 + init_vel[1] ** 2 + init_vel[2] ** 2)
    v_perp = np.sign(init_vel[2]) * np.sqrt(init_vel[1] ** 2 + init_vel[2] ** 2)
    v_para = init_vel[0]
    
    plt.ioff()
    cf.temp_color[0] = 'c'
    
    plt.ioff()
    for jj in range(cf.Nj):
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
    
        # Loss cone diagram
        ax1.scatter(v_perp[cf.idx_start[jj]: cf.idx_end[jj]], v_para[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        ax1.set_title('Initial Loss Cone Distribution :: {}'.format(cf.species_lbl[jj]))
        ax1.set_ylabel('$v_\parallel$ (m/s)')
        ax1.set_xlabel('$v_\perp$ (m/s)')
        ax1.legend(loc='upper right')
        
        # v_mag vs. x
        ax2.scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], v_mag[cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        ax2.set_title('Initial Velocity vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax2.set_xlabel('Position (m)')
        ax2.set_ylabel('Velocity |v| (m/s)')
        ax2.legend(loc='upper right')
            
        # v components vs. x (3 plots)
        ax3[0].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[0, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[1].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[1, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        ax3[2].scatter(init_pos[0, cf.idx_start[jj]: cf.idx_end[jj]], init_vel[2, cf.idx_start[jj]: cf.idx_end[jj]], s=1, c=cf.temp_color[jj])
        
        ax3[0].set_ylabel('$v_x$ (m/s)')
        ax3[1].set_ylabel('$v_y$ (m/s)')
        ax3[2].set_ylabel('$v_z$ (m/s)')
        
        ax3[0].set_title('Initial Velocity Components vs. Position :: {}'.format(cf.species_lbl[jj]))
        ax3[2].set_xlabel('Position (m)')
        
        for ax in ax3:
            ax.legend(loc='upper right')
            
        if save == True:
            fig1.savefig(savedir + 'loss_velocity_space_species_{}'.format(jj))
            fig2.savefig(savedir + 'loss_position_velocity_magnitude_species_{}'.format(jj))
            fig3.savefig(savedir + 'loss_position_velocity_components_species_{}'.format(jj))
 
            plt.close('all')
        else:
            plt.show()
    return


def plot_phase_space_with_time(it_max=None, skip=1):
    ## Same plotting routines as above, just for all times, and saving output
    ## to a file
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
        
    path_cone = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_phase_space/'
    path_mag  = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_mag_vs_x/'
    path_comp = cf.anal_dir + '/Particle_Loss_Analysis/Config vs Time/velocity_components_vs_x/'
    
    for path in [path_cone, path_mag, path_comp]:
        if os.path.exists(path) == False:                                   # Create directories
            os.makedirs(path)
    
    v_max = 4.0 * np.sqrt(kB * cf.Tperp.max() / cf.mass[0]) / cf.va
    
    for ii in range(0, it_max, skip):
        print('Plotting phase space diagrams for particle output {}'.format(ii))
        pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=True)
    
        vel   /= cf.va 
        v_mag  = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        v_perp = np.sign(vel[2]) * np.sqrt(vel[1] ** 2 + vel[2] ** 2)
        v_para = vel[0]
        
        plt.ioff()
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        fig3, ax3 = plt.subplots(3, sharex=True, figsize=(15, 10))
        
        for jj in range(cf.Nj):
    
            if True:
                # Loss cone diagram
                ax1.scatter(v_perp[idx_start[jj]: idx_end[jj]], v_para[idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax1.set_title('Initial Loss Cone Distribution :: t = {:5.4f}'.format(ptime))
                ax1.set_ylabel('$v_\parallel$ (m/s)')
                ax1.set_xlabel('$v_\perp$ (m/s)')
                ax1.set_xlim(-v_max, v_max)
                ax1.set_ylim(-v_max, v_max)
            
            if False:
                # v_mag vs. x
                ax2.scatter(pos[idx_start[jj]: idx_end[jj]], v_mag[idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])       
                ax2.set_title('Initial Velocity vs. Position :: t = {:5.4f}'.format(ptime))
                ax2.set_xlabel('Position (m)')
                ax2.set_ylabel('Velocity |v| (m/s)')
                
                ax2.set_xlim(cf.xmin, cf.xmax)
                ax2.set_ylim(0, v_max)
            
            if True:
                # v components vs. x (3 plots)
                ax3[0].scatter(pos[idx_start[jj]: idx_end[jj]], vel[0, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[1].scatter(pos[idx_start[jj]: idx_end[jj]], vel[1, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                ax3[2].scatter(pos[idx_start[jj]: idx_end[jj]], vel[2, idx_start[jj]: idx_end[jj]], s=1, c=cf.temp_color[jj])
                
                ax3[0].set_ylabel('$v_x$ (m/s)')
                ax3[1].set_ylabel('$v_y$ (m/s)')
                ax3[2].set_ylabel('$v_z$ (m/s)')
                
                for ax in ax3:
                    ax.set_xlim(cf.xmin, cf.xmax)
                    ax.set_ylim(-v_max, v_max)
                
                ax3[0].set_title('Initial Velocity Components vs. Position :: t = {:5.4f}'.format(ptime))
                ax3[2].set_xlabel('Position (m)')
                       
        fig1.savefig(path_cone + 'cone%06d.png' % ii)
        fig2.savefig(path_mag  +  'mag%06d.png' % ii)
        fig3.savefig(path_comp + 'comp%06d.png' % ii)
        
        plt.close('all')
    return


def plot_vi_vs_x(it_max=None, sp=None, save=True, shuffled_idx=False, skip=1, ppd=False):
    '''
    jj can be list of species or int
    
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Issue : Bins along v changing depending on time (how to set max/min bins? Specify arrays manually)
    '''
    # Do checks on species specification
    if sp       is None: sp = np.arange(cf.Nj)
    if type(sp) is int:  sp = [sp]
        
    if ppd == True and os.path.exists(cf.data_dir + '//equil_particles//') == False:
            print('No equilibrium data to plot. Aborting.')
            return
        
    lt = ['x', 'y', 'z']
    
    if it_max is None:
        if ppd == False:
            it_max = len(os.listdir(cf.particle_dir))
        else:
            it_max = len(os.listdir(cf.data_dir + '//equil_particles//'))
            
    xbins = np.linspace(cf.xmin/cf.dx, cf.xmax/cf.dx, cf.NX + 1, endpoint=True)
            
    for ii in range(it_max):
        if ii%skip == 0:
            
            # Decide if continue
            sp_do = []
            for jj in sp:           
                if ppd == False:
                    save_dir = cf.anal_dir + '//Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'fv_vs_x_species_{}_{:05}'.format(jj, ii)
                else:
                    save_dir = cf.anal_dir + '//EQUIL_Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'EQ_fv_vs_x_species_{}_{:05}'.format(jj, ii)
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                if not os.path.exists(save_dir + filename + '.png'):
                    sp_do.append(jj)
            if len(sp_do) == 0: continue
        
            pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=shuffled_idx,
                                                                         preparticledata=ppd)
            if cf.disable_waves:
                ptime = cf.dt_particle*ii
            for jj in sp_do:  
                if ppd == False:
                    save_dir = cf.anal_dir + '//Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'fv_vs_x_species_{}_{:05}'.format(jj, ii)
                else:
                    save_dir = cf.anal_dir + '//EQUIL_Particle Spatial Distribution Histograms//Species {}//'.format(jj)
                    filename = 'EQ_fv_vs_x_species_{}_{:05}'.format(jj, ii)
                    
                print(f'Plotting particle data for species {jj}, time {ii}')
                    
                # Do the calculations and plotting
                cfac = 10  if cf.temp_type[jj] == 1 else 5
                vlim = 5*cf.vth_perp[jj]/cf.va

                # Manually specify bin edges for histogram
                vbins = np.linspace(-vlim, vlim, 101, endpoint=True)
                
                # Do the plotting
                plt.ioff()
                
                fig, axes = plt.subplots(3, figsize=(15, 10), sharex=True)
                axes[0].set_title('f(v) vs. x :: {} :: t = {:.3f}s'.format(cf.species_lbl[jj], ptime))
                
                st = idx_start[jj]
                en = idx_end[jj]
        
                for kk in range(3):
                    counts, xedges, yedges, im1 = axes[kk].hist2d(pos[st:en]/cf.dx, vel[kk, st:en]/cf.va, 
                                                            bins=[xbins, vbins],
                                                            vmin=0, vmax=cf.nsp_ppc[jj] / cfac)
        
                    cb = fig.colorbar(im1, ax=axes[kk], pad=0.015)
                    cb.set_label('Counts')
                    
                    axes[kk].set_ylim(-vlim, vlim)
                    axes[kk].set_ylabel('v{}\n($v_A$)'.format(lt[kk]), rotation=0)
                    
                axes[kk].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
                axes[kk].set_xlabel('Position (cell)')
        
                fig.subplots_adjust(hspace=0.065)
                
                if save:
                    plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                    plt.close('all')
                else:
                    plt.show()
    return


def scatterplot_velocities(it_max=None, skip=1):
    print('Plotting scatterplot of all particle velocities...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))

    save_dir = cf.anal_dir + '//velocity_scatterplots//'
    
    if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
            
    ptime_sec, pbx, pby, pbz, pex, pey, pez, pvex, pvey,\
    pvez, pte, pjx, pjy, pjz, pqdens = cf.interpolate_fields_to_particle_time(it_max)
    
    for ii in range(it_max):
        if ii%skip == 0:
            sys.stdout.write('\rPlotting particle data from p-file {}'.format(ii))
            sys.stdout.flush()
    
            filename = 'velocity_scatterplot_{:05}.png'.format(ii)
            if os.path.exists(save_dir + filename):
                print('Plot already exists, skipping...')
                continue
    
            pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii)
            v_perp = np.sqrt(vel[1, :] ** 2 + vel[2, :] ** 2) * np.sign(vel[2, :])
            # Do the plotting
            plt.ioff()
            
            fig, axes = plt.subplots(4, figsize=(15, 10), sharex=True)
            axes[0].set_title('All particles :: t = {:.3f}s :: x vs. vx'.format(ptime))
            
            for jj in range(cf.Nj):
                st, en = idx_start[jj], idx_end[jj]
                axes[0].scatter(pos[st:en]/cf.dx, vel[0, st:en]/cf.va, c=cf.temp_color[jj], s=1)
                axes[1].scatter(pos[st:en]/cf.dx, v_perp[st:en]/cf.va, c=cf.temp_color[jj], s=1)
            
            axes[0].set_ylabel('$v_\parallel$')
            axes[1].set_ylabel('$v_\perp$')
            
            axes[0].set_ylim(-15, 15)
            axes[1].set_ylim(-15, 15) 
            
            axes[-1].set_xlim(cf.xmin/cf.dx, cf.xmax/cf.dx)
            axes[-1].set_xlabel('x (dx)')
            
            axes[2].plot(cf.E_nodes/cf.dx, pjx[ii])
            axes[2].set_ylabel('$J_x$', rotation=0)
            axes[2].set_ylim(pjx.min(), pjx.max())
            
            axes[3].plot(cf.E_nodes/cf.dx, pqdens[ii])
            axes[3].set_ylabel('dens')
            axes[3].set_ylim(pqdens.min(), pqdens.max())
    
            fig.subplots_adjust(hspace=0)
            
            plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close('all')
    print('\n')
    return


def plot_vi_vs_t_for_cell(cell=None, comp=0, it_max=None, jj=1, save=True, hexbin=False):
    '''
    For each point in time
     - Collect particle information for particles near cell, plus time component
     - Store in array
     - Plot using either hexbin or hist2D
     
    Fix this: Make it better. Manually specify bins maybe? Or use 1D hist and compile at each timestep
                now that I know how to make sure bin limits are the same at each time
    '''
    if cell == None:
        cell = cf.NX//2
    cell  += cf.ND
    x_node = cf.E_nodes[cell]
    
    print('Calculating distribution vs. time...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
       
    vel_i = np.zeros(0, dtype=float)
    time  = np.zeros(0, dtype=float)
        
    # Collect all particle information for specified cell
    for ii in range(it_max):
        sys.stdout.write('\rAccruing particle data from p-file {}'.format(ii))
        sys.stdout.flush()

        pos, vel, idx, ptime= cf.load_particles(ii)
        
        f = np.zeros((0, 3))    ;   count = 0
        for ii in np.arange(cf.idx_start[jj], cf.idx_end[jj]):
            if (abs(pos[0, ii] - x_node) <= 0.5*cf.dx):
                f = np.append(f, [vel[0:3, ii]], axis=0)
                count += 1
                
        vel_i = np.append(vel_i, f[:, comp])
        time  = np.append(time, np.ones(count) * ptime)
    print('\n')

    vel_i /= cf.va

    # Do the plotting
    plt.ioff()
    
    xmin = time.min()
    xmax = time.max()
    ymin = vel_i.min()
    ymax = vel_i.max()
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("F(v) vs. t :: {} :: Cell {}".format(cf.species_lbl[jj], cell))
    
    if hexbin == True:
        im1 = ax.hexbin(time, vel_i, gridsize=50, cmap='inferno')
        ax.axis([xmin, xmax, ymin, ymax])
    else:
        im1 = ax.hist2d(time, vel_i, bins=100)
               
    #cb = fig.colorbar(im1, ax=ax)
    #cb.set_label('Counts')
    if save == True:
        save_dir = cf.anal_dir + '//Particle Distribution Histograms//Time//'
        filename = 'v{}_vs_t_cell_{}_species_{}'.format(comp, cell, jj)
        
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        
        plt.savefig(save_dir + filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print('Distribution histogram for v{}, cell {}, species {} saved'.format(comp, cell, jj))
        plt.close('all')
    else:
        plt.show()
    return


def analyse_sampleParticle_motion(it_max=None):
    '''
    Mainly looking at a few particles at a time to get a sense of the motion
    of these particles in a bottle/with waves
    '''
    # To Do:
    #   - Track bounce period of some hot/cold particles (maybe a handful each?)
    #   - Look at their magnetic moments with time
    savedir = cf.anal_dir + '/Sample_Particle_Motion/'
    
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)

    if it_max is None:
        num_particle_steps = len(os.listdir(cf.particle_dir))
    
    ptime = np.zeros(num_particle_steps)
    np.random.seed(cf.seed)
    
    # CREATE SAMPLE ARRAY :: Either equal number from each, or just from the one
    N_samples = 5
    pos, vel, idx, sim_time, idx_start, idx_end = cf.load_particles(0)    
    
    if False:
        # Collect a sample from each species
        sloc = np.zeros((cf.Nj * N_samples), dtype=int)  # Sample location (to not confuse with particle index)
        for ii in range(cf.Nj):
            sloc[ii*N_samples: (ii + 1)*N_samples] = np.random.randint(idx_start[ii], idx_end[ii], N_samples, dtype=int)
    elif True:
        # Collect a sample from just one species
        jj   = 0
        sloc = np.random.randint(idx_start[jj], idx_end[jj], N_samples, dtype=int)
    
    ## COLLECT DATA ON THESE PARTICLES
    sidx      = np.zeros((num_particle_steps, sloc.shape[0]), dtype=int)    # Sample particle index
    spos      = np.zeros((num_particle_steps, sloc.shape[0]))            # Sample particle position
    svel      = np.zeros((num_particle_steps, sloc.shape[0], 3))            # Sample particle velocity
    
    # Load up species index and particle position, velocity for samples
    for ii in range(num_particle_steps):
        pos, vel, idx, ptime[ii], idx_start, idx_end = cf.load_particles(ii)
        print('Loading sample particle data for particle file {}'.format(ii))
        for jj in range(sloc.shape[0]):
            sidx[ii, jj]    = idx[sloc[jj]]
            spos[ii, jj]    = pos[sloc[jj]]
            svel[ii, jj, :] = vel[:, sloc[jj]]

    if True:
        # Plot position/velocity (will probably have to put a catch in here for absorbed particles: ylim?)
        plt.ioff()
        fig, axes = plt.subplots(2, sharex=True, figsize=(16, 10))
        for ii in range(sloc.shape[0]):
            axes[0].plot(ptime, spos[:, ii]   , c=cf.temp_color[sidx[0, ii]])
            axes[1].plot(ptime, svel[:, ii, 0], c=cf.temp_color[sidx[0, ii]])
            
            axes[0].set_title('Sample Positions/Velocities of Particles :: Indices {}'.format(sloc))
            axes[0].set_ylabel('Position (m)')
            axes[1].set_ylabel('Velocity (m/s)') 
            
            axes[-1].set_xlabel('Time (s)')
            
        fig.savefig(savedir + 'sample.png')
        plt.close('all')
    return


def plot_particle_paths(it_max=None, nsamples=1000):    
    save_folder = cf.anal_dir + '//particle_trajectories//'
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    indexes = np.random.randint(0, cf.N, nsamples)
    Np      = len(indexes)

    spos = np.zeros((it_max, Np))
    svel = np.zeros((it_max, Np, 3))
    
    for ii in range(it_max):
        print('Loading timestep', ii, 'of', it_max)
        pos, vel, idx, ptime, idx_start, idx_end = cf.load_particles(ii, shuffled_idx=False)
        
        for index, jj in zip(indexes, range(Np)):
            spos[ii, jj]    = pos[   index]
            svel[ii, jj, :] = vel[:, index]
    
    for ii in Np:
        prt = indexes[ii]
        print('Plotting trajectory for particle', ii)
        fig, ax = plt.subplots()
        ax.set_title('Particle {} Trajectory :: Bottle {:.1f} - {:.1f} nT'.format(prt, cf.B_eq*1e9, cf.B_xmax*1e9))
        ax.scatter(spos[:, prt], svel[:, prt, 0], label='$v_x$')
        ax.scatter(spos[:, prt], svel[:, prt, 1], label='$v_y$')
        ax.scatter(spos[:, prt], svel[:, prt, 2], label='$v_z$')
        
        fig.savefig(save_folder + 'particle_{:08}'.format(ii))
        plt.close('all')
    return


def thesis_plot_mu_and_position(it_max=None, save_plot=True, save_data=True):
    '''
    Generate publication quality plot to show average mu of particles
    as well as the individual mu and locations of sample particles.
    
    Limit sample search to particles that are only within 1dx of the equator
    
    Need function to extract particle paths?
    '''
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    plot_path = cf.anal_dir + '//mu_and_position_SI//'
    if not os.path.exists(plot_path): os.makedirs(plot_path)
    save_data_file = cf.temp_dir + 'thesis_mu_position_data.npz'
    
    print('Collecting particle first invariants...')
    if it_max is None:
        it_max = len(os.listdir(cf.particle_dir))
    
    ################
    ### GET DATA ###
    ################
    if os.path.exists(save_data_file) == False or save_data == False:
        
        # Initial positions and velocities (hard coded because reasons)
        ipos, ivel, _, _, _, _ = cf.load_particles(0)
        count = np.arange(ipos.shape[0])
        print('Finding all particle indices within 1 cell of equator')
        sidx = count[np.abs(ipos) < cf.dx]
        Ns = sidx.shape[0]
        
        # Define arrays
        Bp    = np.zeros((it_max, Ns))
        mu    = np.zeros((it_max, Ns))
        pos   = np.zeros((it_max, Ns))
        vel   = np.zeros((it_max, Ns, 3))
        ptime = np.zeros(it_max)
        
        # Retrieve relevant data into arrays
        for ii in range(it_max):
            print('Loading particle information at timestep {}'.format(ii))
            this_pos, this_vel, this_idx, ptime[ii], idx_start, idx_end = cf.load_particles(ii)
            for ss in range(sidx.shape[0]):
                _idx                   = sidx[ss] 
                mu[ii, ss], Bp[ii, ss] = calc_mu(this_pos[_idx], this_vel[:, _idx], this_idx[_idx])
                pos[ii, ss]            = this_pos[_idx]
                vel[ii, ss, :]         = this_vel[:, _idx]
                
        if save_data == True and os.path.exists(save_data_file) == False:
            np.savez(save_data_file, Bp=Bp, pos=pos, vel=vel, mu=mu, ptime=ptime, sidx=sidx)
    else:
        print('Loading from file')
        data = np.load(save_data_file)
        Bp    = data['Bp']
        mu    = data['mu']
        pos   = data['pos']
        vel   = data['vel']
        ptime = data['ptime']
        sidx  = data['sidx']

    #################
    ## Do PLOTTING ##
    #################
    yfac       = 1e8
    ylabel_pad = 20
    fontsize   = 8
    tick_size  = 8
    
    mpl.rcParams['xtick.labelsize'] = tick_size 
    mpl.rcParams['ytick.labelsize'] = tick_size 

    cf.temp_color[0] = 'c'

    plt.ioff()
    for ss in range(sidx.shape[0]):
        mu_av = mu[ :, ss].mean()
        
        fig, axes = plt.subplots(2, figsize=(6.0, 4.0), sharex=True)
        
        axes[0].plot(ptime, pos[:, ss] / cf.dx)
        axes[1].plot(ptime, mu[ :, ss] * yfac)
        
        axes[0].set_ylabel('$\\frac{x}{\Delta x}$', rotation=0, fontsize=fontsize+6, labelpad=ylabel_pad)
        axes[1].set_ylabel('$\mu$\n$(\\times 10^{%d})$' % np.log10(yfac), rotation=0,
                           labelpad=ylabel_pad, fontsize=fontsize)

        axes[1].set_xlabel('Time (s)', fontsize=fontsize)
        axes[1].set_xlim(0, ptime[-1])
        axes[1].set_ylim(0.9*mu_av*yfac, 1.1*mu_av*yfac)
            
        fig.subplots_adjust(hspace=0.1)
        fig.align_ylabels()
    
        if True:
            fpath = plot_path + f'mu_position_idx_{ss:08}.png'
            fig.savefig(fpath, bbox_inches='tight', dpi=200)
            plt.close('all')
            print('1st adiabatic invariant graph for particle {} saved as {}'.format(ss, fpath))
        else:
            plt.show()
    return


def plot_max_velocity(save=True):
    
    num_particle_steps = len(os.listdir(cf.particle_dir))
    
    # Load up species index and particle position, velocity for samples
    ptime  = np.zeros(num_particle_steps)
    max_vx = np.zeros(num_particle_steps)
    max_vy = np.zeros(num_particle_steps)
    max_vz = np.zeros(num_particle_steps)
    for ii in range(num_particle_steps):
        print('Loading sample particle data for particle file {}'.format(ii))
        pos, vel, idx, ptime[ii], id1, id2 = cf.load_particles(ii)
        
        max_vx[ii] = np.abs(vel[0]).max()
        max_vy[ii] = np.abs(vel[1]).max()
        max_vz[ii] = np.abs(vel[2]).max()
        
    plt.ioff()
    fig, ax = plt.subplots()
    ax.set_title('Maximum particle velocity in simulation')
    ax.plot(ptime, max_vx, label='$v_x$', c='k')
    ax.plot(ptime, max_vy, label='$v_y$', c='r', alpha=0.5, lw=0.5)
    ax.plot(ptime, max_vz, label='$v_z$', c='b', alpha=0.5, lw=0.5)
    ax.axhline(max_vx[0], color='k', ls='--', alpha=0.5)
    ax.legend()
    
    if save == True:
        fullpath = cf.anal_dir + 'max_velocity_check' + '.png'
        plt.savefig(fullpath, facecolor=fig.get_facecolor(), edgecolor='none')
        print('t-x Plot saved')
        plt.close('all')
    else:
        plt.show()
    return